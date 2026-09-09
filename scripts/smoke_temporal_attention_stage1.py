"""Minimal synthetic Stage 1 -> Stage 2 smoke for experiment 028."""

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module.layers.encoder import CrossAssetTransformerEncoder
from trainer.train_vqvae import FactorVQVAE
from trainer.train_ypred import GenerateReturn
from utils import seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "028" / "smoke"
SOURCE_MODEL = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
COPIED_MODEL = ROOT / "module" / "layers" / "encoder.py"
ALPHA_MASTER_CLASSES = (
    "PositionalEncoding",
    "TAttention",
    "TemporalAttention",
)


def class_nodes(source):
    tree = ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def verify_source_fidelity():
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"AlphaMaster source is missing: {SOURCE_MODEL}")
    source = class_nodes(SOURCE_MODEL.read_text())
    copied = class_nodes(COPIED_MODEL.read_text())
    alpha_exact = {}
    for name in ALPHA_MASTER_CLASSES:
        exact = ast.dump(source[name], include_attributes=False) == ast.dump(
            copied[name], include_attributes=False
        )
        if not exact:
            raise AssertionError(f"{name} differs from AlphaMaster source")
        alpha_exact[name] = True

    baseline_source = subprocess.run(
        ["git", "show", "main:module/layers/encoder.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    baseline = class_nodes(baseline_source)
    cross_asset_exact = ast.dump(
        baseline["CrossAssetTransformerEncoder"], include_attributes=False
    ) == ast.dump(copied["CrossAssetTransformerEncoder"], include_attributes=False)
    if not cross_asset_exact:
        raise AssertionError("CrossAssetTransformerEncoder differs from main")
    return alpha_exact, cross_asset_exact


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

    alpha_exact, cross_asset_exact = verify_source_fidelity()
    config = build_config()
    encoder_config = config["vqvae"]["encoder"]
    expected_splits = {
        "train_period": ["2009-01-01", "2020-12-31"],
        "valid_period": ["2021-01-01", "2022-12-31"],
        "test_period": ["2023-01-01", "2025-12-31"],
    }
    if encoder_config["type"] != "temporal-attention":
        raise AssertionError("Default config does not enable experiment 028")
    if config["train"]["seed"] != 0:
        raise AssertionError("Stage 2 seed changed")
    for key, expected in expected_splits.items():
        if config["data"][key] != expected:
            raise AssertionError(f"Data split changed for {key}")

    # One synthetic optimization step exercises the complete unchanged Stage 1
    # objective: reconstruction + VQ + pred_weight * prediction.
    seed_everything(42)
    pl.seed_everything(42, workers=True)
    stage1 = FactorVQVAE(copy.deepcopy(config), T_max=1).train()
    encoder = stage1.vqvae.spatial_encoder
    if any(isinstance(module, torch.nn.GRU) for module in encoder.modules()):
        raise AssertionError("Stage 1 temporal encoder still contains a GRU")
    if not isinstance(encoder.cross_asset_transformer, CrossAssetTransformerEncoder):
        raise AssertionError("Original CrossAssetTransformer is absent")
    if any(type(module).__name__ == "SAttention" for module in encoder.modules()):
        raise AssertionError("SAttention was introduced")
    if any("gate" in name.lower() for name, _ in encoder.named_modules()):
        raise AssertionError("Market Gate was introduced")

    shape_trace = {}

    def record_shape(name):
        def hook(_module, inputs, output):
            shape_trace[name] = {
                "input": list(inputs[0].shape),
                "output": list(output.shape),
            }
        return hook

    hooks = [
        encoder.temporal_encoder.temporal_attention.register_forward_hook(
            record_shape("tattention")
        ),
        encoder.temporal_encoder.temporal_aggregation.register_forward_hook(
            record_shape("temporal_aggregation")
        ),
        encoder.cross_asset_transformer.register_forward_hook(
            record_shape("cross_asset_transformer")
        ),
    ]

    generator = torch.Generator().manual_seed(28042)
    feature = torch.randn(8, 20, 158, generator=generator)
    prior = torch.randn(8, 13, generator=generator)
    future_returns = torch.randn(8, 10, generator=generator)
    stage1_optimizer = torch.optim.AdamW(
        stage1.parameters(), lr=config["train"]["learning_rate"]
    )
    stage1_optimizer.zero_grad()
    stage1_output = stage1(feature, prior, future_returns)
    for hook in hooks:
        hook.remove()
    recon_loss, vq_loss, pred_loss, total_loss, z_q, _ = stage1_output
    expected_trace = {
        "tattention": {"input": [8, 20, 128], "output": [8, 20, 128]},
        "temporal_aggregation": {"input": [8, 20, 128], "output": [8, 128]},
        "cross_asset_transformer": {"input": [8, 128], "output": [8, 128]},
    }
    if shape_trace != expected_trace:
        raise AssertionError(f"Unexpected encoder data flow: {shape_trace}")
    if z_q.shape != (8, 128):
        raise AssertionError(f"Unexpected Stage 1 latent shape: {tuple(z_q.shape)}")
    if not torch.isfinite(total_loss):
        raise AssertionError("Stage 1 total loss is not finite")
    total_loss.backward()
    encoder_grad_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    )
    if encoder_grad_l1 <= 0:
        raise AssertionError("Stage 1 encoder received no gradient")
    stage1_optimizer.step()

    stage1_checkpoint = checkpoint_dir / "temporal-attention-stage1-smoke.ckpt"
    save_lightning_checkpoint(stage1_checkpoint, stage1, global_step=1)

    # The normal Stage 2 constructor must strict-load this experiment's self
    # Stage 1 encoder, quantizer, and RevIN without changing Stage 2 logic.
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
    stage2_output = stage2(feature, prior)
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

    stage2_checkpoint = checkpoint_dir / "temporal-attention-stage2-smoke.ckpt"
    save_lightning_checkpoint(stage2_checkpoint, stage2, global_step=1)
    restored = GenerateReturn.load_from_checkpoint(
        stage2_checkpoint,
        config=copy.deepcopy(stage2_config),
        T_max=1,
        strict=True,
    ).eval()
    stage2.eval()
    with torch.no_grad():
        expected_output = stage2(feature, prior)
        restored_output = restored(feature, prior)
    checkpoint_exact = all(
        torch.equal(expected, actual)
        for expected, actual in zip(expected_output, restored_output)
    )
    if not checkpoint_exact:
        raise AssertionError("Strict Stage 2 checkpoint round-trip changed output")

    report = {
        "status": "PASS",
        "experiment": {"id": "028", "name": "prism-temporal-attention-stage1"},
        "source_fidelity": {
            "alpha_master_source": str(SOURCE_MODEL.relative_to(ROOT.parent)),
            "alpha_master_component_ast_exact": alpha_exact,
            "cross_asset_transformer_ast_exact_to_main": cross_asset_exact,
        },
        "stage1": {
            "source": "self",
            "checkpoint": str(stage1_checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": stage1_checkpoint.stat().st_size,
            "latent_shape": list(z_q.shape),
            "encoder_data_flow": shape_trace,
            "encoder_gradient_l1": encoder_grad_l1,
            "losses": {
                "reconstruction": float(recon_loss.detach()),
                "vq": float(vq_loss.detach()),
                "prediction": float(pred_loss.detach()),
                "total": float(total_loss.detach()),
            },
            "cross_asset_transformer_preserved": True,
            "gru_absent": True,
            "sattention_absent": True,
            "market_gate_absent": True,
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
        },
    }
    report_path = ARTIFACT_ROOT / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
