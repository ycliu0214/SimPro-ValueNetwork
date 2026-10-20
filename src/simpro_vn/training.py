"""SimPro-ValueNetwork supervised prefix training, fine-tuning, and three-stage curricula."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from .bundles import resolve_bundle
from .checkpoints import checkpoint_metadata
from .model import RecurrentValueNetwork
from .representations import REPRESENTATIONS, get_splitter
from .scaling import PROPERTY_NAMES, PropertyScaler
from .settings import normalize_setting
from .tokenization import SurrogateChemicalTokenizer
from .training_data import EncodedDataset, collate_examples, prepare_data


@dataclass
class TrainingConfig:
    dataset_path: str
    output_dir: str
    mode: str = "finetune"
    checkpoint_path: str | None = None
    resume_from: str | None = None
    representation: str = "smiles"
    min_frequency: int | str = 5000
    depth: int = 6
    text_column: str | None = None
    split_column: str = "Split_Set"
    group_column: str | None = None
    tokenizer_prefix: str | None = None
    scaler_path: str | None = None
    skip_invalid: bool = False
    epochs: int = 10
    batch_size: int = 32
    accumulation_steps: int = 1
    learning_rate: float = 2e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    schedule: str = "discounted"
    gamma: float = 0.95
    precision: str = "fp32"
    device: str = "auto"
    seed: int = 42
    log_every_steps: int = 100
    d_model: int = 768
    nhead: int = 12
    dim_feedforward: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 128
    warmup_epochs: int = 2
    linear_epochs: int = 10
    discounted_epochs: int = 10
    stage3_batch_size: int = 1024
    curriculum_depths: tuple = (2, 4, 6)

    @classmethod
    def from_json(cls, path):
        path = Path(path).expanduser().resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        for key in ("dataset_path", "output_dir", "checkpoint_path", "resume_from", "tokenizer_prefix", "scaler_path"):
            if values.get(key) is not None:
                value = Path(values[key]).expanduser()
                values[key] = str(value if value.is_absolute() else path.parent / value)
        return cls(**values)

    def validate(self):
        for name in ("dataset_path", "output_dir", "checkpoint_path", "resume_from", "tokenizer_prefix", "scaler_path"):
            if getattr(self, name) is not None:
                setattr(self, name, str(Path(getattr(self, name)).expanduser().resolve()))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.mode not in ("finetune", "scratch", "curriculum"):
            raise ValueError("mode must be finetune, scratch, or curriculum")
        if self.representation not in REPRESENTATIONS or self.depth not in (2, 4, 6):
            raise ValueError("Select a supported representation and depth 2, 4, or 6")
        self.min_frequency = normalize_setting(self.min_frequency)
        for name in ("epochs", "batch_size", "accumulation_steps", "d_model", "nhead",
                     "dim_feedforward", "max_seq_len", "warmup_epochs", "linear_epochs",
                     "discounted_epochs", "stage3_batch_size", "log_every_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead or self.d_model % 2 or self.max_seq_len < 2:
            raise ValueError("d_model must be even and divisible by nhead; max_seq_len must be >= 2")
        for name in ("learning_rate", "min_learning_rate", "clip_norm", "gamma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_learning_rate > self.learning_rate or self.gamma > 1:
            raise ValueError("min_learning_rate must not exceed learning_rate; gamma must be <= 1")
        if self.mode == "curriculum" and self.min_learning_rate > 1e-5:
            raise ValueError("Curriculum min_learning_rate must not exceed the warm-up rate 1e-5")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Require 0 <= dropout < 1 and finite weight_decay >= 0")
        if self.schedule not in ("linear", "discounted") or self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Select linear/discounted schedule and fp32/fp16/bf16 precision")
        if self.mode == "finetune" and not (self.checkpoint_path or self.resume_from):
            raise ValueError("finetune requires checkpoint_path or resume_from")
        if self.checkpoint_path and self.resume_from:
            raise ValueError("Use checkpoint_path for initialization or resume_from for continuation, not both")
        if self.mode in ("scratch", "curriculum") and self.checkpoint_path:
            raise ValueError("scratch/curriculum initialize randomly; use finetune to load weights")
        if self.mode == "curriculum" and self.resume_from:
            raise ValueError("Resume individual phases with their saved phase_config.json")
        if not self.curriculum_depths or len(set(self.curriculum_depths)) != len(self.curriculum_depths) or any(
            d not in (2, 4, 6) for d in self.curriculum_depths
        ):
            raise ValueError("curriculum_depths must be unique choices from 2, 4, 6")


def temporal_weights(lengths, width, schedule="linear", gamma=0.95):
    """Return unnormalized weights; padding is always zero (positions are zero-based)."""
    if schedule not in ("linear", "discounted") or not 0 < gamma <= 1:
        raise ValueError("Require linear/discounted schedule and 0 < gamma <= 1")
    if lengths.ndim != 1 or torch.any(lengths < 1) or torch.any(lengths > width):
        raise ValueError("lengths must be a vector between 1 and sequence width")
    positions = torch.arange(width, device=lengths.device)[None, :]
    valid = positions < lengths[:, None]
    if schedule == "linear":
        weights = 0.1 + 0.9 * positions.float() / (lengths[:, None] - 1).clamp_min(1)
    else:
        weights = gamma ** (lengths[:, None] - 1 - positions).clamp_min(0).float()
    return weights.masked_fill(~valid, 0.0)


def temporal_loss(predictions, targets, lengths, schedule="linear", gamma=0.95, reduction="mean"):
    """Mean over properties, normalized per molecule, then mean over molecules."""
    if predictions.ndim != 3 or targets.shape != (predictions.size(0), predictions.size(2)):
        raise ValueError("Expected predictions [B,T,P] and targets [B,P]")
    if lengths.shape != (predictions.size(0),):
        raise ValueError("One length is required per molecule")
    weights = temporal_weights(lengths, predictions.size(1), schedule, gamma)
    valid = weights > 0
    # Mask before squaring so unused padded predictions cannot contaminate the loss.
    errors = (predictions.float() - targets.float()[:, None]).masked_fill(~valid[:, :, None], 0).square().mean(-1)
    losses = (errors * weights).sum(1) / (weights.sum(1) + 1e-8)
    if reduction == "none":
        return losses
    if reduction != "mean":
        raise ValueError("reduction must be mean or none")
    return losses.mean()


def _atomic_save(payload, path):
    # Only called inside a newly created run directory owned by this invocation.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _load(path):
    return torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)


def _signature(config):
    ignored = {"dataset_path", "output_dir", "checkpoint_path", "resume_from", "device",
               "tokenizer_prefix", "scaler_path"}
    # JSON canonicalization also normalizes tuple/list settings.
    return json.loads(json.dumps({k: v for k, v in asdict(config).items() if k not in ignored}))


def _new_model(config, tokenizer, scaler, fingerprint, initializer=None, allow_depth_transfer=False):
    if initializer:
        checkpoint = _load(initializer)
        metadata, _ = checkpoint_metadata(initializer, checkpoint)
        for name, expected in (("schema_version", 1), ("architecture", "recurrent_depth_transformer"),
                               ("representation", config.representation), ("min_frequency", config.min_frequency),
                               ("bundle_fingerprint", fingerprint), ("nhead", config.nhead),
                               ("property_names", list(PROPERTY_NAMES))):
            if name in metadata and metadata[name] != expected:
                raise ValueError(f"Checkpoint {name} does not match training configuration")
        if not allow_depth_transfer and metadata.get("depth", config.depth) != config.depth:
            raise ValueError("Checkpoint depth does not match training depth")
        state = checkpoint.get("model_state_dict", checkpoint)
        vocab_size, d_model = state["embedding.weight"].shape
        if vocab_size != tokenizer.vocab_size or state["value_head.3.weight"].shape[0] != len(PROPERTY_NAMES):
            raise ValueError("Checkpoint vocabulary or target count is incompatible")
        dimensions = {"vocab_size": vocab_size, "d_model": d_model,
                      "num_targets": state["value_head.3.weight"].shape[0],
                      "max_seq_len": state["pos_encoder.pe"].shape[1],
                      "dim_feedforward": state["shared_transformer_block.linear1.weight"].shape[0]}
        for name, actual in dimensions.items():
            if name in metadata and metadata[name] != actual:
                raise ValueError(f"Checkpoint tensor shapes differ from its model description: {name}")
        special_ids = {"[PAD]": tokenizer.pad_id, "[BOS]": tokenizer.bos_id, "[EOS]": tokenizer.eos_id}
        if "special_token_ids" in metadata and metadata["special_token_ids"] != special_ids:
            raise ValueError("Checkpoint special token IDs differ from the selected tokenizer")
        model = RecurrentValueNetwork(vocab_size, d_model, config.nhead,
                                      state["shared_transformer_block.linear1.weight"].shape[0],
                                      config.dropout, len(PROPERTY_NAMES), state["pos_encoder.pe"].shape[1])
        model.load_state_dict(state, strict=True)
        return model
    return RecurrentValueNetwork(tokenizer.vocab_size, config.d_model, config.nhead,
                                 config.dim_feedforward, config.dropout, len(PROPERTY_NAMES), config.max_seq_len)


def _metadata(config, model, fingerprint, tokenizer):
    return {"schema_version": 1, "architecture": "recurrent_depth_transformer",
            "representation": config.representation, "min_frequency": config.min_frequency,
            "depth": config.depth, "bundle_fingerprint": fingerprint, "nhead": config.nhead,
            "dropout": config.dropout, "property_names": list(PROPERTY_NAMES),
            "max_seq_len": model.pos_encoder.pe.size(1), "d_model": model.d_model,
            "dim_feedforward": model.shared_transformer_block.linear1.out_features,
            "vocab_size": model.embedding.num_embeddings, "num_targets": len(PROPERTY_NAMES),
            "special_token_ids": {"[PAD]": tokenizer.pad_id, "[BOS]": tokenizer.bos_id, "[EOS]": tokenizer.eos_id}}


def _device(config):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Training supports cpu or cuda devices")
    if device.type == "cuda" and (not torch.cuda.is_available() or
                                  (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ValueError("The selected CUDA device is unavailable")
    if config.precision != "fp32" and device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA; use fp32 on CPU")
    if config.precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support bf16; select fp16 or fp32")
    return device


def _phase(config, directory, tokenizer, scaler, fingerprint, summary, data_dir,
           initializer=None, allow_depth_transfer=False, max_epochs=None):
    directory = Path(directory)
    directory.mkdir()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    device = _device(config)
    model = _new_model(config, tokenizer, scaler, fingerprint, config.resume_from or initializer, allow_depth_transfer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_learning_rate)
    if hasattr(torch.amp, "GradScaler"):
        amp_scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    else:
        amp_scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")
    generator = torch.Generator().manual_seed(config.seed)
    start, best, history, best_checkpoint = 0, float("inf"), [], None
    metadata = _metadata(config, model, fingerprint, tokenizer)
    _json(directory / "phase_config.json", asdict(config))
    if config.resume_from:
        resumed = _load(config.resume_from)
        if resumed.get("checkpoint_type") != "training_last":
            raise ValueError("resume_from requires last.pt with optimizer and RNG state")
        if resumed["signature"] != _signature(config) or resumed["data_fingerprint"] != summary["fingerprint"]:
            raise ValueError("Resume requires unchanged data, split, schedule, epochs, and training settings")
        if resumed["device_type"] != device.type:
            raise ValueError("Resume requires the same device type; initialize a new fine-tuning run to change it")
        start, best, history = resumed["epoch"], resumed["best_val_loss"], resumed["history"]
        if start >= config.epochs:
            raise ValueError("This run already completed; start a new fine-tuning run to train further")
        best_checkpoint = resumed["best_checkpoint"]
        saved_metadata = best_checkpoint.get("simpro_vn", {})
        if (any(key in metadata and value != metadata[key] for key, value in saved_metadata.items()) or
                best_checkpoint.get("val_loss") != best or
                best_checkpoint.get("data_fingerprint") != summary["fingerprint"]):
            raise ValueError("The best-model snapshot in last.pt is inconsistent")
        _atomic_save(best_checkpoint, directory / "best.pt")
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        amp_scaler.load_state_dict(resumed["amp_scaler_state_dict"])
        torch.set_rng_state(resumed["torch_rng_state"])
        generator.set_state(resumed["loader_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(resumed["cuda_rng_states"])
    train = EncodedDataset(data_dir / "train.jsonl")
    val = EncodedDataset(data_dir / "val.jsonl")
    collate = partial(collate_examples, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=False, collate_fn=collate, num_workers=0,
                            generator=torch.Generator().manual_seed(config.seed))
    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16

    def autocast():
        return torch.autocast("cuda", dtype=dtype) if config.precision != "fp32" else nullcontext()

    def batch_loss(batch):
        src, targets, lengths = (value.to(device) for value in batch)
        with autocast():
            predictions = model(src, padding_mask=src.eq(tokenizer.pad_id), inference_depth=config.depth)
            losses = temporal_loss(predictions, targets, lengths, config.schedule, config.gamma, reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite loss; inspect labels and precision settings")
        return losses

    try:
        stop = config.epochs if max_epochs is None else min(config.epochs, start + max_epochs)
        for epoch in range(start, stop):
            model.train()
            train_sum, train_count, skipped_updates, updates = 0.0, 0, 0, 0
            iterator = iter(train_loader)
            while True:
                group = []
                for _ in range(config.accumulation_steps):
                    batch = next(iterator, None)
                    if batch is None:
                        break
                    group.append(batch)
                if not group:
                    break
                count = sum(len(batch[1]) for batch in group)
                optimizer.zero_grad(set_to_none=True)
                for batch in group:
                    losses = batch_loss(batch)
                    amp_scaler.scale(losses.sum() / count).backward()
                    train_sum += losses.detach().sum().item()
                    train_count += losses.numel()
                amp_scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
                if config.precision != "fp16" and not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradients; inspect labels and learning rate")
                previous_scale = amp_scaler.get_scale()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                skipped_updates += int(amp_scaler.get_scale() < previous_scale)
                updates += 1
                if updates == 1 or updates % config.log_every_steps == 0:
                    print(json.dumps({"event": "train_progress", "epoch": epoch + 1,
                                      "optimizer_groups": updates, "molecules": train_count,
                                      "total_molecules": len(train), "mean_loss": train_sum / train_count}), flush=True)
            model.eval()
            val_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    losses = batch_loss(batch)
                    val_sum += losses.sum().item()
                    val_count += losses.numel()
            row = {"epoch": epoch + 1, "train_loss": train_sum / train_count, "val_loss": val_sum / val_count,
                   "learning_rate": optimizer.param_groups[0]["lr"], "skipped_fp16_updates": skipped_updates}
            history.append(row)
            scheduler.step()
            common = {"model_state_dict": model.state_dict(), "simpro_vn": metadata,
                      "epoch": epoch + 1, "val_loss": row["val_loss"], "data_fingerprint": summary["fingerprint"]}
            if row["val_loss"] < best:
                best = row["val_loss"]
                # Clone the best model so later optimizer steps cannot mutate it.
                best_checkpoint = {**common, "checkpoint_type": "model_best", "model_state_dict": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }}
                _atomic_save(best_checkpoint, directory / "best.pt")
            _atomic_save({**common, "checkpoint_type": "training_last", "best_val_loss": best,
                          "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                          "amp_scaler_state_dict": amp_scaler.state_dict(), "signature": _signature(config),
                          "history": history, "best_checkpoint": best_checkpoint,
                          "device_type": device.type, "torch_rng_state": torch.get_rng_state(),
                          "loader_rng_state": generator.get_state(),
                          "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
                         directory / "last.pt")
            _json(directory / "history.json", history)
            print(json.dumps(row), flush=True)
        inference_config = {"checkpoint_path": "best.pt", "representation": config.representation,
                            "min_frequency": config.min_frequency, "depth": config.depth, "nhead": config.nhead}
        if config.tokenizer_prefix:
            inference_config["tokenizer_prefix"] = str(Path(config.tokenizer_prefix).expanduser().resolve())
        if config.scaler_path:
            inference_config["scaler_path"] = str(Path(config.scaler_path).expanduser().resolve())
        _json(directory / "inference_config.json", inference_config)
        return {"directory": str(directory), "best_checkpoint": str(directory / "best.pt"),
                "last_checkpoint": str(directory / "last.pt"), "best_val_loss": best,
                "completed_epochs": len(history), "history": history}
    finally:
        train.close()
        val.close()


def run_training(config, *, max_epochs=None):
    """Create a fresh run. max_epochs bounds a single-phase invocation for smoke runs."""
    if isinstance(config, (str, Path)):
        config = TrainingConfig.from_json(config)
    config.validate()
    _device(config)
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 1 or config.mode == "curriculum"):
        raise ValueError("max_epochs must be positive and is only supported for a single phase")
    output = Path(config.output_dir).expanduser().resolve()
    protected = Path(__file__).resolve().parents[2] / "weights"
    if output == protected or protected in output.parents:
        raise ValueError("Training outputs must be outside the project's pretrained weights directory")
    # exist_ok=False prevents overwriting checkpoints or an earlier run.
    output.mkdir(parents=True, exist_ok=False)
    config.output_dir = str(output)
    _json(output / "training_config.json", asdict(config))
    prefix, scaler_path, fingerprint = resolve_bundle(config.representation, config.min_frequency,
                                                     config.tokenizer_prefix, config.scaler_path)
    tokenizer = SurrogateChemicalTokenizer(prefix, get_splitter(config.representation))
    scaler = PropertyScaler(scaler_path)
    initializer = config.resume_from or config.checkpoint_path
    # Read shape metadata on CPU; no optimizer starts until all data pass preflight.
    reference = _new_model(config, tokenizer, scaler, fingerprint, initializer)
    max_seq_len = reference.pos_encoder.pe.size(1)
    del reference
    summary = prepare_data(config, tokenizer, scaler, max_seq_len, output)
    _json(output / "environment.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                         "cuda_available": torch.cuda.is_available(), "seed": config.seed})
    if config.mode != "curriculum":
        result = _phase(config, output / "model", tokenizer, scaler, fingerprint, summary,
                        output / "data", initializer=config.checkpoint_path, max_epochs=max_epochs)
    else:
        values = asdict(config)
        warm = TrainingConfig(**{**values, "mode": "scratch", "depth": 2, "epochs": config.warmup_epochs,
                                 "learning_rate": 1e-5, "schedule": "linear"})
        warm.validate()
        warm_result = _phase(warm, output / "warmup_d2", tokenizer, scaler, fingerprint, summary, output / "data")
        branches = {}
        for depth in config.curriculum_depths:
            linear = TrainingConfig(**{**values, "mode": "finetune", "depth": depth,
                                       "checkpoint_path": warm_result["best_checkpoint"], "epochs": config.linear_epochs,
                                       "learning_rate": 1e-4, "schedule": "linear"})
            linear.validate()
            linear_result = _phase(linear, output / f"linear_d{depth}", tokenizer, scaler, fingerprint, summary,
                                   output / "data", initializer=warm_result["best_checkpoint"], allow_depth_transfer=True)
            discounted = TrainingConfig(**{**values, "mode": "finetune", "depth": depth,
                                           "checkpoint_path": linear_result["best_checkpoint"],
                                           "epochs": config.discounted_epochs, "batch_size": config.stage3_batch_size,
                                           "learning_rate": 2e-5, "schedule": "discounted"})
            discounted.validate()
            final = _phase(discounted, output / f"discounted_d{depth}", tokenizer, scaler, fingerprint, summary,
                           output / "data", initializer=linear_result["best_checkpoint"])
            branches[str(depth)] = {"linear": linear_result, "discounted": final}
        result = {"warmup": warm_result, "branches": branches}
    _json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train or fine-tune a molecular prefix value network")
    parser.add_argument("--config", required=True, type=Path, help="Training JSON; paths resolve relative to this file")
    parser.add_argument("--max-epochs", type=int, help="Run at most this many epochs of a single phase, retaining resume state")
    args = parser.parse_args()
    print(json.dumps(run_training(args.config, max_epochs=args.max_epochs), indent=2))


if __name__ == "__main__":
    main()
