"""Discrete historical-market adapter AlphaMaster tests."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from dataset.dataset import init_data_loader
from dataset.schema import GROUP_SLICES, TOTAL_DIM
from module.alphamaster import MASTER, StandardVectorQuantizer
from trainer.train_alphamaster import AlphaMasterModule


ROOT = Path(__file__).resolve().parent.parent


def load_config(universe="csi300"):
    with (ROOT / "configs" / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    config["data"]["universe"] = universe
    return config


class IndexedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, values, index):
        self.values = values
        self.index = index

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def get_index(self):
        return self.index


class AlphaMasterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = torch.randn(6, 20, TOTAL_DIM)

    def test_default_config_is_discrete_historical_market_adapter(self):
        config = load_config()
        self.assertEqual(config["train"]["seed"], 0)
        self.assertEqual(config["train"]["learning_rate"], 8e-6)
        self.assertEqual(config["predictor"]["target_day"], 5)
        self.assertNotIn("vqvae", config)
        self.assertEqual(config["data"]["window_size"], 20)
        self.assertEqual(config["alphamaster"], {
            "d_feat": 158,
            "d_market": 63,
            "d_model": 256,
            "t_nhead": 4,
            "s_nhead": 2,
            "T_dropout_rate": 0.5,
            "S_dropout_rate": 0.5,
            "market_encoder": {
                "type": "gru",
                "input_size": 63,
                "hidden_size": 63,
                "num_layers": 1,
                "batch_first": True,
                "bidirectional": False,
                "dropout": 0,
            },
            "market_quantizer": {
                "type": "standard_vq",
                "codebook_size": 8,
                "embedding_dim": 63,
                "distance": "l2",
                "straight_through": True,
                "commitment_weight": 0.25,
            },
            "market_adapter": {
                "type": "linear",
                "input_size": 63,
                "output_size": 256,
                "bias": False,
                "zero_init": True,
            },
            "beta": {"csi300": 10, "sp500": 5},
        })

    def test_canonical_shapes_path_slices_and_zero_init_equivalence(self):
        model = AlphaMasterModule(load_config()).eval()
        self.assertFalse(any("prior" in name for name, _ in model.named_parameters()))
        stock, market, target = model._get_data(self.batch)
        self.assertEqual(stock.shape, (6, 20, 158))
        self.assertEqual(market.shape, (6, 20, 63))
        self.assertEqual(target.shape, (6,))

        captured = {}

        def save_input(name):
            return lambda module, args: captured.__setitem__(
                name, args[0].detach().clone()
            )

        def save_output(name):
            return lambda module, args, output: captured.__setitem__(
                name, output.detach().clone()
            )

        def save_vq_output(module, args, output):
            captured["vq_input"] = args[0].detach().clone()
            captured["quantized_state"] = output.quantized.detach().clone()
            captured["vq_indices"] = output.indices.detach().clone()
            captured["vq_loss"] = output.loss

        def save_adapter_input(module, args):
            captured["adapter_input"] = args[0].detach().clone()

        handles = [
            model.master.feature_gate.register_forward_pre_hook(save_input("gate")),
            model.master.market_encoder.register_forward_pre_hook(save_input("history")),
            model.master.market_encoder.register_forward_hook(save_output("market_state")),
            model.master.market_quantizer.register_forward_hook(save_vq_output),
            model.master.temporalatten.register_forward_hook(save_output("hidden")),
            model.master.market_adapter.register_forward_pre_hook(save_adapter_input),
            model.master.market_adapter.register_forward_hook(save_output("delta_weight")),
        ]
        prediction = model(stock, market)
        for handle in handles:
            handle.remove()

        self.assertEqual(captured["history"].shape, (6, 19, 63))
        self.assertEqual(captured["market_state"].shape, (6, 63))
        self.assertEqual(captured["vq_input"].shape, (6, 63))
        self.assertEqual(captured["quantized_state"].shape, (6, 63))
        self.assertEqual(captured["vq_indices"].shape, (6,))
        self.assertEqual(captured["hidden"].shape, (6, 256))
        self.assertEqual(captured["delta_weight"].shape, (6, 256))
        self.assertEqual(prediction.shape, (6,))
        torch.testing.assert_close(captured["gate"], market[:, -1, :], rtol=0, atol=0)
        torch.testing.assert_close(captured["history"], market[:, :-1, :], rtol=0, atol=0)
        self.assertEqual(model.master.market_encoder.gru.input_size, 63)
        self.assertEqual(model.master.market_encoder.gru.hidden_size, 63)
        self.assertEqual(model.master.market_encoder.gru.num_layers, 1)
        self.assertTrue(model.master.market_encoder.gru.batch_first)
        self.assertFalse(model.master.market_encoder.gru.bidirectional)
        self.assertEqual(model.master.market_encoder.gru.dropout, 0)
        self.assertEqual(model.master.market_quantizer.codebook_size, 8)
        self.assertEqual(model.master.market_quantizer.embedding_dim, 63)
        self.assertEqual(model.master.market_quantizer.embedding.weight.shape, (8, 63))
        self.assertEqual(model.master.market_quantizer.commitment_weight, 0.25)
        torch.testing.assert_close(
            captured["vq_input"], captured["market_state"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            captured["adapter_input"], captured["quantized_state"], rtol=0, atol=0
        )
        self.assertEqual(model.master.market_adapter.in_features, 63)
        self.assertEqual(model.master.market_adapter.out_features, 256)
        self.assertIsNone(model.master.market_adapter.bias)
        self.assertEqual(torch.count_nonzero(model.master.market_adapter.weight), 0)
        self.assertEqual(torch.count_nonzero(captured["delta_weight"]), 0)

        base_prediction = model.master.decoder(captured["hidden"]).squeeze(-1)
        torch.testing.assert_close(prediction, base_prediction, rtol=0, atol=0)

        changed_prior = self.batch.clone()
        changed_prior[..., GROUP_SLICES["prior"]] += 1_000_000
        changed_stock, changed_market, _ = model._get_data(changed_prior)
        torch.testing.assert_close(
            prediction, model(changed_stock, changed_market), rtol=0, atol=0
        )

    def test_007_prediction_equivalence_with_matching_backbone(self):
        source = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
        if not source.is_file():
            self.skipTest("standalone AlphaMaster checkout is unavailable")
        spec = importlib.util.spec_from_file_location("standalone_alphamaster", source)
        standalone = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(standalone)

        torch.manual_seed(31)
        reference_007 = standalone.MASTER(beta=10).eval()
        torch.manual_seed(31)
        model_012 = MASTER(beta=10).eval()
        reference_state = reference_007.state_dict()
        for name, value in reference_state.items():
            torch.testing.assert_close(
                model_012.state_dict()[name], value, rtol=0, atol=0
            )
        features = torch.randn(5, 20, 221)
        torch.testing.assert_close(
            model_012(features), reference_007(features), rtol=0, atol=0
        )

    def test_standard_vq_loss_is_finite_and_straight_through(self):
        quantizer = StandardVectorQuantizer(
            codebook_size=8, embedding_dim=63, commitment_weight=0.25
        )
        inputs = torch.randn(5, 63, requires_grad=True)
        output = quantizer(inputs)
        self.assertEqual(output.quantized.shape, inputs.shape)
        self.assertEqual(output.indices.shape, (5,))
        self.assertTrue(torch.isfinite(output.loss))
        torch.testing.assert_close(
            output.loss,
            output.codebook_loss + 0.25 * output.commitment_loss,
        )
        torch.testing.assert_close(
            output.quantized,
            quantizer.embedding(output.indices),
            rtol=0,
            atol=0,
        )
        output.quantized.sum().backward(retain_graph=True)
        torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))
        self.assertIsNone(quantizer.embedding.weight.grad)
        inputs.grad = None
        output.loss.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        self.assertIsNotNone(quantizer.embedding.weight.grad)
        self.assertGreater(quantizer.embedding.weight.grad.abs().sum().item(), 0)

    def test_market_paths_are_decoupled(self):
        model = AlphaMasterModule(load_config()).eval()
        stock, market, _ = model._get_data(self.batch)
        market = market[:1].expand_as(market).clone()
        changed_history = market.clone()
        changed_history[:, :-1, 0] += 10
        with torch.no_grad():
            original_state = model.master.market_encoder(market[:, :-1, :])[0]
            historical_state = model.master.market_encoder(
                changed_history[:, :-1, :]
            )[0]
            model.master.market_quantizer.embedding.weight.fill_(1000)
            model.master.market_quantizer.embedding.weight[0].copy_(original_state)
            model.master.market_quantizer.embedding.weight[1].copy_(historical_state)
            model.master.market_adapter.weight.copy_(
                torch.randn_like(model.master.market_adapter.weight) * 0.01
            )

        def forward_and_capture(candidate_market):
            values = {}
            handles = [
                model.master.feature_gate.register_forward_pre_hook(
                    lambda module, args: values.__setitem__("gate", args[0].detach().clone())
                ),
                model.master.market_encoder.register_forward_pre_hook(
                    lambda module, args: values.__setitem__("history", args[0].detach().clone())
                ),
                model.master.market_quantizer.register_forward_hook(
                    lambda module, args, output: values.update({
                        "market_state": args[0].detach().clone(),
                        "quantized_state": output.quantized.detach().clone(),
                        "indices": output.indices.detach().clone(),
                    })
                ),
                model.master.market_adapter.register_forward_hook(
                    lambda module, args, output: values.__setitem__(
                        "delta_weight", output.detach().clone()
                    )
                ),
            ]
            values["prediction"] = model(stock, candidate_market).detach()
            for handle in handles:
                handle.remove()
            return values

        original = forward_and_capture(market)
        historical = forward_and_capture(changed_history)
        torch.testing.assert_close(historical["gate"], original["gate"], rtol=0, atol=0)
        self.assertFalse(torch.equal(historical["market_state"], original["market_state"]))
        self.assertFalse(torch.equal(historical["indices"], original["indices"]))
        self.assertFalse(
            torch.equal(historical["quantized_state"], original["quantized_state"])
        )
        self.assertFalse(torch.equal(historical["delta_weight"], original["delta_weight"]))
        self.assertFalse(torch.equal(historical["prediction"], original["prediction"]))

        changed_current = market.clone()
        changed_current[:, -1, 0] += 100
        current = forward_and_capture(changed_current)
        torch.testing.assert_close(current["history"], original["history"], rtol=0, atol=0)
        torch.testing.assert_close(
            current["market_state"], original["market_state"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            current["quantized_state"], original["quantized_state"], rtol=0, atol=0
        )
        torch.testing.assert_close(current["indices"], original["indices"], rtol=0, atol=0)
        torch.testing.assert_close(
            current["delta_weight"], original["delta_weight"], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(current["gate"], original["gate"]))
        self.assertFalse(torch.equal(current["prediction"], original["prediction"]))

    def test_cross_section_shares_quantized_regime_and_dynamic_weight(self):
        model = AlphaMasterModule(load_config()).eval()
        stock, market, _ = model._get_data(self.batch)
        market = market[:1].expand_as(market).clone()
        with torch.no_grad():
            model.master.market_adapter.weight.normal_(std=0.01)
        captured = {}
        handles = [
            model.master.market_encoder.register_forward_hook(
                lambda module, args, output: captured.__setitem__(
                    "market_state", output.detach().clone()
                )
            ),
            model.master.market_quantizer.register_forward_hook(
                lambda module, args, output: captured.update({
                    "quantized_state": output.quantized.detach().clone(),
                    "indices": output.indices.detach().clone(),
                })
            ),
            model.master.market_adapter.register_forward_hook(
                lambda module, args, output: captured.__setitem__(
                    "delta_weight", output.detach().clone()
                )
            ),
        ]
        prediction = model(stock, market)
        for handle in handles:
            handle.remove()
        self.assertEqual(prediction.shape, (6,))
        torch.testing.assert_close(
            captured["market_state"],
            captured["market_state"][:1].expand_as(captured["market_state"]),
            rtol=1e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            captured["quantized_state"],
            captured["quantized_state"][:1].expand_as(captured["quantized_state"]),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.equal(
            captured["indices"], captured["indices"][:1].expand_as(captured["indices"])
        ))
        torch.testing.assert_close(
            captured["delta_weight"],
            captured["delta_weight"][:1].expand_as(captured["delta_weight"]),
            rtol=1e-5,
            atol=2e-7,
        )

    def test_vq_gru_and_adapter_receive_gradients_and_update(self):
        model = AlphaMasterModule(load_config()).train()
        stock, market, target = model._get_data(self.batch)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        initial_parameters = {
            "adapter": model.master.market_adapter.weight.detach().clone(),
            "codebook": model.master.market_quantizer.embedding.weight.detach().clone(),
            "gru": model.master.market_encoder.gru.weight_ih_l0.detach().clone(),
        }
        prediction, vq_output = model(stock, market, return_vq_output=True)
        loss = model.loss_fn(prediction, target) + vq_output.loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = {
            "adapter": model.master.market_adapter.weight.grad,
            "codebook": model.master.market_quantizer.embedding.weight.grad,
            "gru": model.master.market_encoder.gru.weight_ih_l0.grad,
        }
        for name, gradient in gradients.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(gradient)
                self.assertGreater(gradient.abs().sum().item(), 0)
        optimizer.step()
        updated_parameters = {
            "adapter": model.master.market_adapter.weight,
            "codebook": model.master.market_quantizer.embedding.weight,
            "gru": model.master.market_encoder.gru.weight_ih_l0,
        }
        for name, parameter in updated_parameters.items():
            with self.subTest(parameter=name):
                self.assertFalse(torch.equal(initial_parameters[name], parameter))

    def test_csi300_and_sp500_forward_with_expected_beta(self):
        for universe, beta in (("csi300", 10), ("sp500", 5)):
            with self.subTest(universe=universe):
                model = AlphaMasterModule(load_config(universe)).eval()
                stock, market, _ = model._get_data(self.batch)
                self.assertEqual(model.master.feature_gate.t, beta)
                self.assertEqual(model(stock, market).shape, (6,))

    def test_daily_sampler_emits_one_complete_cross_section(self):
        dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-03",
                                "2023-01-04", "2023-01-03", "2023-01-04"])
        instruments = ["A", "A", "B", "B", "C", "C"]
        index = pd.MultiIndex.from_arrays(
            [dates, instruments], names=["datetime", "instrument"]
        )
        dataset = IndexedTensorDataset(self.batch, index)
        loader, batches = init_data_loader(dataset, shuffle=False)
        self.assertEqual(batches, 2)
        emitted = loader.batch_sampler.ordered_indices()
        self.assertEqual(emitted.tolist(), [0, 2, 4, 1, 3, 5])
        for positions in loader.batch_sampler:
            batch_dates = index[positions].get_level_values("datetime")
            self.assertEqual(batch_dates.nunique(), 1)
            self.assertEqual(
                set(index[positions].get_level_values("instrument")),
                {"A", "B", "C"},
            )

    def test_checkpoint_load_is_strict_and_prediction_is_preserved(self):
        config = load_config()
        model = AlphaMasterModule(config).eval()
        stock, market, _ = model._get_data(self.batch)
        expected_prediction = model(stock, market)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ckpt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = AlphaMasterModule.load_strict_checkpoint(path, config).eval()
            torch.testing.assert_close(
                expected_prediction, restored(stock, market), rtol=0, atol=0
            )

            state = model.state_dict()
            state.pop(next(iter(state)))
            torch.save({"state_dict": state}, path)
            with self.assertRaises(RuntimeError):
                AlphaMasterModule.load_strict_checkpoint(path, config)

            base_011_state = {
                name: value for name, value in model.state_dict().items()
                if not name.startswith("master.market_quantizer.")
            }
            torch.save({"state_dict": base_011_state}, path)
            with self.assertRaises(RuntimeError):
                AlphaMasterModule.load_strict_checkpoint(path, config)


if __name__ == "__main__":
    unittest.main()
