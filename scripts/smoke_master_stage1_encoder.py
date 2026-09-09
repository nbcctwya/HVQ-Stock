"""Minimal synthetic Stage 1 -> Stage 2 smoke for experiment 030."""

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

from module.layers.encoder import MASTERStyleEncoder, TemporalAttention
from trainer.train_vqvae import FactorVQVAE
from trainer.train_ypred import GenerateReturn
from utils import seed_everything


ARTIFACT_ROOT = ROOT / "artifacts" / "030" / "smoke"
SOURCE_MODEL = ROOT.parent / "AlphaMaster" / "src" / "alphamaster" / "model.py"
COPIED_MODEL = ROOT / "module" / "layers" / "encoder.py"
COPIED_CLASSES = (
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
        raise AssertionError("Default config does not enable experiment 030")
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
    if any(isinstance(module, TemporalAttention) for module in stage1.vqvae.spatial_encoder.modules()):
        raise AssertionError("Learned TemporalAttention still participates in Stage 1")
    if any("gate" in name.lower() for name, _ in stage1.vqvae.spatial_encoder.named_modules()):
        raise AssertionError("Market Gate was introduced")

    generator = torch.Generator().manual_seed(27042)
    feature = torch.randn(8, 20, 158, generator=generator)
    prior = torch.randn(8, 13, generator=generator)
    future_returns = torch.randn(8, 10, generator=generator)
    call_order = []
    spatial_output = None
    pooled_output = None

    def record(name):
        def hook(_module, _inputs, _output):
            call_order.append(name)
        return hook

    def capture_spatial_output(_module, _inputs, output):
        nonlocal spatial_output
        spatial_output = output.detach().clone()

    def capture_pooled_output(_module, inputs):
        nonlocal pooled_output
        pooled_output = inputs[0].detach().clone()

    spatial_encoder = stage1.vqvae.spatial_encoder
    ordered_modules = [
        ("Input Projection", spatial_encoder.master_encoder.x2y),
        ("PositionalEncoding", spatial_encoder.master_encoder.pe),
        ("TAttention", spatial_encoder.master_encoder.tatten),
        ("SAttention", spatial_encoder.master_encoder.satten),
        ("Projection MLP", spatial_encoder.out_layer),
    ]
    handles = [module.register_forward_hook(record(name)) for name, module in ordered_modules]
    handles.append(spatial_encoder.master_encoder.satten.register_forward_hook(capture_spatial_output))
    handles.append(spatial_encoder.out_layer.register_forward_pre_hook(capture_pooled_output))
    stage1_optimizer = torch.optim.AdamW(
        stage1.parameters(), lr=config["train"]["learning_rate"]
    )
    stage1_optimizer.zero_grad()
    try:
        stage1_output = stage1(feature, prior, future_returns)
    finally:
        for handle in handles:
            handle.remove()
    expected_order = [name for name, _ in ordered_modules]
    if call_order != expected_order:
        raise AssertionError(f"Unexpected encoder call order: {call_order}")
    if spatial_output.shape != (8, 20, 128):
        raise AssertionError(f"Unexpected pre-pooling shape: {tuple(spatial_output.shape)}")
    if pooled_output.shape != (8, 128):
        raise AssertionError(f"Unexpected post-pooling shape: {tuple(pooled_output.shape)}")
    if not torch.equal(pooled_output, spatial_output.mean(dim=1)):
        raise AssertionError("Temporal aggregation is not exactly mean(dim=1)")
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
        "experiment": {"id": "030", "name": "prism-mean-pooling-stage1-encoder"},
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
            "market_gate_absent": True,
            "gru_absent": True,
            "temporal_attention_absent": True,
            "call_order": call_order,
            "pre_pooling_shape": list(spatial_output.shape),
            "post_pooling_shape": list(pooled_output.shape),
            "mean_pooling_exact": True,
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
