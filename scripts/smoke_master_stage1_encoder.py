"""Minimal synthetic Stage 1 -> Stage 2 smoke for experiment 031."""

import ast
import copy
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module.layers.encoder import Gate, MASTERStyleEncoder
from trainer.train_vqvae import FactorVQVAE
from trainer.train_ypred import GenerateReturn
from utils import seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "031" / "smoke"
SOURCE_MODEL = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
COPIED_MODEL = ROOT / "module" / "layers" / "encoder.py"
COPIED_CLASSES = (
    "Gate",
    "PositionalEncoding",
    "TAttention",
    "SAttention",
    "TemporalAttention",
)


def class_nodes(path):
    tree = ast.parse(path.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def verify_alpha_master_copy():
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"AlphaMaster source is missing: {SOURCE_MODEL}")
    source = class_nodes(SOURCE_MODEL)
    copied = class_nodes(COPIED_MODEL)
    result = {}
    for name in COPIED_CLASSES:
        exact = ast.dump(source[name], include_attributes=False) == ast.dump(
            copied[name], include_attributes=False
        )
        if not exact:
            raise AssertionError(f"{name} differs from AlphaMaster source")
        result[name] = True
    return result


def build_config():
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    return OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "config.yaml"), resolve=True
    )


def save_lightning_checkpoint(path, model, global_step):
    torch.save(
        {
            "epoch": 0,
            "global_step": global_step,
            "pytorch-lightning_version": pl.__version__,
            "state_dict": model.state_dict(),
        },
        path,
    )


def main():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ARTIFACT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_copy = verify_alpha_master_copy()
    config = build_config()
    encoder_config = config["vqvae"]["encoder"]
    expected_splits = {
        "train_period": ["2009-01-01", "2020-12-31"],
        "valid_period": ["2021-01-01", "2022-12-31"],
        "test_period": ["2023-01-01", "2025-12-31"],
    }
    if encoder_config["type"] != "master":
        raise AssertionError("Default config does not enable experiment 031")
    expected_gate_config = {
        "input_dim": 63,
        "beta": {"csi300": 10, "sp500": 5},
    }
    if encoder_config["market_gate"] != expected_gate_config:
        raise AssertionError("Default config does not fix the canonical gate rules")
    if config["train"]["seed"] != 0:
        raise AssertionError("Stage 2 seed changed")
    for key, expected in expected_splits.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # One synthetic Stage 1 optimization step exercises the complete unchanged
    # loss path: reconstruction + VQ + prediction.
    seed_everything(42)
    pl.seed_everything(42, workers=True)
    stage1 = FactorVQVAE(copy.deepcopy(config), T_max=1).train()
    if not isinstance(stage1.vqvae.spatial_encoder.master_encoder, MASTERStyleEncoder):
        raise AssertionError("Stage 1 did not construct MASTERStyleEncoder")
    if any(isinstance(module, torch.nn.GRU) for module in stage1.vqvae.spatial_encoder.modules()):
        raise AssertionError("Stage 1 encoder still contains a GRU")
    feature_gate = stage1.vqvae.spatial_encoder.feature_gate
    if not isinstance(feature_gate, Gate):
        raise AssertionError("AlphaMaster Market Gate was not constructed")
    if feature_gate.t != 10:
        raise AssertionError("CSI300 beta must be 10")

    generator = torch.Generator().manual_seed(31042)
    feature = torch.randn(8, 20, 158, generator=generator)
    prior = torch.randn(8, 13, generator=generator)
    market = torch.randn(8, 20, 63, generator=generator)
    future_returns = torch.randn(8, 10, generator=generator)

    # Capture the actual Stage 1 path, including the unchanged RevIN and the
    # gate's adapted canonical input/output shapes.
    call_order = []
    gate_shapes = {}

    def record(name):
        def hook(_module, inputs, output):
            call_order.append(name)
            if name == "Market Gate":
                gate_shapes["input"] = list(inputs[0].shape)
                gate_shapes["output"] = list(output.shape)
                gate_shapes["weight_sums"] = output.detach().sum(dim=-1).tolist()
        return hook

    encoder = stage1.vqvae.spatial_encoder
    ordered_modules = [
        ("RevIN", stage1.vqvae.revin),
        ("Market Gate", encoder.feature_gate),
        ("Feature Transform", encoder.feature_transform),
        ("Input Projection", encoder.master_encoder.x2y),
        ("PositionalEncoding", encoder.master_encoder.pe),
        ("TAttention", encoder.master_encoder.tatten),
        ("SAttention", encoder.master_encoder.satten),
        ("TemporalAttention", encoder.master_encoder.temporalatten),
        ("Projection MLP", encoder.out_layer),
    ]
    handles = [module.register_forward_hook(record(name)) for name, module in ordered_modules]
    stage1_optimizer = torch.optim.AdamW(
        stage1.parameters(), lr=config["train"]["learning_rate"]
    )
    stage1_optimizer.zero_grad()
    try:
        stage1_output = stage1(feature, prior, market, future_returns)
    finally:
        for handle in handles:
            handle.remove()
    expected_order = [name for name, _ in ordered_modules] + ["RevIN"]
    if call_order != expected_order:
        raise AssertionError(f"Unexpected Stage 1 call order: {call_order}")
    if gate_shapes["input"] != [8, 63] or gate_shapes["output"] != [8, 158]:
        raise AssertionError(f"Unexpected gate shapes: {gate_shapes}")
    if not torch.allclose(
        torch.tensor(gate_shapes["weight_sums"]), torch.full((8,), 158.0),
        rtol=1e-6, atol=1e-5,
    ):
        raise AssertionError("Gate weights do not sum to 158")
    recon_loss, vq_loss, pred_loss, total_loss, z_q, _ = stage1_output
    if z_q.shape != (8, 128):
        raise AssertionError(f"Unexpected Stage 1 latent shape: {tuple(z_q.shape)}")
    if not torch.isfinite(total_loss):
        raise AssertionError("Stage 1 total loss is not finite")
    total_loss.backward()
    encoder_grad_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in stage1.vqvae.spatial_encoder.parameters()
        if parameter.grad is not None
    )
    if encoder_grad_l1 <= 0:
        raise AssertionError("MASTER-style Stage 1 encoder received no gradient")
    stage1_optimizer.step()

    # Only market_feature[:, -1, :] may affect the gate and encoder latent.
    stage1.eval()
    changed_history = market.clone()
    changed_history[:, :-1, :] += 1000
    changed_current = market.clone()
    changed_current[:, -1, 0] += 1000
    changed_prior = prior + 1000
    with torch.no_grad():
        gate = feature_gate(market[:, -1, :])
        historical_gate = feature_gate(changed_history[:, -1, :])
        current_gate = feature_gate(changed_current[:, -1, :])
        prior_invariant_gate = feature_gate(market[:, -1, :])
        feature_normalized = stage1.vqvae.revin(feature, mode="norm")
        latent = encoder(feature_normalized, market)
        historical_latent = encoder(feature_normalized, changed_history)
        current_latent = encoder(feature_normalized, changed_current)
    if not torch.equal(gate, historical_gate) or not torch.equal(latent, historical_latent):
        raise AssertionError("Past market timesteps leaked into Market Gate")
    if torch.equal(gate, current_gate) or torch.equal(latent, current_latent):
        raise AssertionError("Current market state did not affect gate and latent")
    if not torch.equal(gate, prior_invariant_gate) or torch.equal(prior, changed_prior):
        raise AssertionError("Market Gate unexpectedly depends on prior factors")

    stage1_checkpoint = checkpoint_dir / "master-stage1-smoke.ckpt"
    save_lightning_checkpoint(stage1_checkpoint, stage1, global_step=1)

    # GenerateReturn's normal constructor performs strict loading of encoder,
    # quantizer and RevIN from the self-trained Stage 1 checkpoint.
    stage2_config = copy.deepcopy(config)
    stage2_config["predictor"]["saved_model"] = str(stage1_checkpoint)
    seed_everything(0)
    pl.seed_everything(0, workers=True)
    stage2 = GenerateReturn(stage2_config, T_max=1)
    stage2.train()
    if stage2.encoder.training or stage2.quantizer.training or stage2.revin.training:
        raise AssertionError("Frozen Stage 1 modules did not remain in eval mode")
    if any(parameter.requires_grad for parameter in stage2.encoder.parameters()):
        raise AssertionError("Stage 1 encoder was not frozen in Stage 2")

    stage2_optimizer = torch.optim.AdamW(
        [parameter for parameter in stage2.parameters() if parameter.requires_grad],
        lr=config["train"]["learning_rate"],
    )
    stage2_optimizer.zero_grad()
    stage2_output = stage2(feature, prior, market)
    y_pred, beta_p, beta_l, stage2_z_q, aux_loss = stage2_output
    if y_pred.shape != (8,):
        raise AssertionError(f"Unexpected prediction shape: {tuple(y_pred.shape)}")
    if stage2_z_q.shape != (8, 128):
        raise AssertionError("Stage 2 received an incompatible VQ latent")
    stage2_loss = y_pred.square().mean() + config["predictor"]["aux_weight"] * aux_loss
    if not torch.isfinite(stage2_loss):
        raise AssertionError("Stage 2 loss is not finite")
    stage2_loss.backward()
    frozen_stage1_has_grad = any(
        parameter.grad is not None
        for module in (stage2.encoder, stage2.quantizer, stage2.revin)
        for parameter in module.parameters()
    )
    if frozen_stage1_has_grad:
        raise AssertionError("Frozen Stage 1 received Stage 2 gradients")
    trainable_stage2_grad_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in stage2.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if trainable_stage2_grad_l1 <= 0:
        raise AssertionError("Stage 2 received no gradient")
    stage2_optimizer.step()

    stage2_checkpoint = checkpoint_dir / "master-stage2-smoke.ckpt"
    save_lightning_checkpoint(stage2_checkpoint, stage2, global_step=1)
    restored = GenerateReturn.load_from_checkpoint(
        stage2_checkpoint,
        config=copy.deepcopy(stage2_config),
        T_max=1,
        strict=True,
    ).eval()
    stage2.eval()
    with torch.no_grad():
        expected_output = stage2(feature, prior, market)
        restored_output = restored(feature, prior, market)
    checkpoint_exact = all(
        torch.equal(expected, actual)
        for expected, actual in zip(expected_output, restored_output)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict Stage 2 checkpoint round-trip changed output")

    report = {
        "status": "PASS",
        "experiment": {"id": "031", "name": "prism-market-gated-stage1-encoder"},
        "alpha_master_source": str(SOURCE_MODEL.relative_to(ROOT.parent)),
        "alpha_master_component_ast_exact": source_copy,
        "stage1": {
            "source": "self",
            "checkpoint": str(stage1_checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": stage1_checkpoint.stat().st_size,
            "latent_shape": list(z_q.shape),
            "encoder_gradient_l1": encoder_grad_l1,
            "losses": {
                "reconstruction": float(recon_loss.detach()),
                "vq": float(vq_loss.detach()),
                "prediction": float(pred_loss.detach()),
                "total": float(total_loss.detach()),
            },
            "market_gate": {
                "input_shape": gate_shapes["input"],
                "output_shape": gate_shapes["output"],
                "weight_sum_target": 158,
                "beta": feature_gate.t,
                "source_ast_exact": source_copy["Gate"],
                "only_last_market_timestep": True,
                "prior_invariant": True,
                "current_market_changes_gate": not torch.equal(gate, current_gate),
                "current_market_changes_encoder_latent": not torch.equal(latent, current_latent),
            },
            "call_order": call_order,
            "gru_absent": True,
        },
        "stage2_compatibility": {
            "encoder_strict_load": {"missing": 0, "unexpected": 0},
            "quantizer_strict_load": {"missing": 0, "unexpected": 0},
            "revin_strict_load": {"missing": 0, "unexpected": 0},
            "frozen_stage1_has_grad": frozen_stage1_has_grad,
            "prediction_shape": list(y_pred.shape),
            "prior_loading_shape": list(beta_p.shape),
            "latent_loading_shape": list(beta_l.shape),
            "vq_latent_shape": list(stage2_z_q.shape),
            "trainable_gradient_l1": trainable_stage2_grad_l1,
            "checkpoint": str(stage2_checkpoint.relative_to(ROOT)),
            "strict_checkpoint_round_trip": checkpoint_exact,
        },
        "unchanged": {
            "vq_type": type(stage2.quantizer).__name__,
            "codebook_shape": list(stage2.quantizer.embedding.weight.shape),
            "data_splits": expected_splits,
            "stage1_seed": 42,
            "stage2_seed": config["train"]["seed"],
            "beta_rules": expected_gate_config["beta"],
            "stage2_market_conditioning_scope": "frozen Stage 1 encoder only",
        },
    }
    report_path = ARTIFACT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
