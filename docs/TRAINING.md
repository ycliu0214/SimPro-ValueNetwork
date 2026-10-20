# Training and Fine-Tuning

This document covers optional learning and custom-model examples. The primary use is pretrained inference; ordinary prediction does not require labels, training, or fine-tuning.

Train from random initialization, fine-tune pretrained models, or run a three-stage curriculum. All modes share the model and temporally weighted loss. Saved models work with the prediction API and CLI.

## Installation and Entry Points

```bash
pip install ".[chemistry]"
simpro-vn-train --help
simpro-vn-train --config configs/training/finetune_smiles.json
python -m simpro_vn.training --config configs/training/finetune_smiles.json
```

Training uses PyTorch and tokenizers; the chemistry extra supplies SAFE/SELFIES dependencies. The data extra is needed for label calculation. See the Colab examples for [fine-tuning](../examples/colab_finetuning.ipynb), [three-stage training](../examples/colab_training.ipynb), and [comprehensive validation](../examples/colab_comprehensive_validation.ipynb).

Training configurations and notebooks default to the [300-row smoke dataset](../examples/data/README.md), so no additional CSV is required for workflow checks. Change the dataset path for production fine-tuning; smoke data do not establish prediction accuracy.

## Data Format

Use a UTF-8, comma-delimited CSV with one complete molecule per row and:

| Column | Meaning |
| --- | --- |
| `NEW_Isomeric_SMILES`, `SMARTS`, `SAFE`, `SELFIES` | The selected representation determines the input column; override with `text_column` |
| 17 properties | Names and units follow [property definitions](PROPERTIES.md); supply unstandardized labels |
| `Split_Set` | Train, Val (or Validation), Test; test rows are excluded from optimization and model selection |
| `Parent_Group` | Automatically checked when present; override with `group_column` |

Split Train/Val beforehand, keeping a parent's stereoisomers together. The preparation example supplies parent groups. Training rejects exact-string and supplied-group overlap between Train and Val. Without a group column, only exact strings are checked; equivalent SMILES, scaffolds, and stereoisomers are not inferred. Establish chemical validity and conformer quality during [data preparation](DATA_PREPARATION.md).

All Train/Val rows undergo label, token, round-trip, and length checks before optimization. Invalid rows stop execution by default; `skip_invalid=true` explicitly skips and records them in `data/rejections.jsonl`. Invalid splits and overlap always stop execution. Rows are encoded into disk-backed caches rather than retained as CSV objects in RAM. Reserve space for caches, the SQLite group index, and checkpoints; `/content/` is recommended for Colab outputs.

## Fixed Preprocessing

All four representations and five tokenization settings are supported. Every mode retains the selected tokenizer, mapping, and scaler; none retrains tokenization or refits scaling. “From scratch” refers to network initialization. Labels are converted to z-scores with the fixed scaler. Encodings include BOS/EOS; padding is excluded from loss and oversized strings are rejected.

## Temporal Weights and Loss

Let `T_i` be valid sequence length, with positions `t = 1, ..., T_i` including BOS/EOS. The complete molecule's 17 labels supervise every valid prefix, using only the specified final recurrent depth.

```text
linear:     w(i,t) = 0.1 + 0.9 * (t - 1) / (T_i - 1)
            w(i,1) = 0.1 when T_i = 1
discounted: w(i,t) = gamma ** (T_i - t), gamma = 0.95

error(i,t) = mean_over_17_properties((prediction_z(i,t) - target_z(i)) ** 2)
loss(i)    = sum_t(w(i,t) * error(i,t)) / (sum_t(w(i,t)) + 1e-8)
batch_loss = mean_over_molecules(loss(i))
```

Linear weights rise from 0.1 to 1.0 using each molecule's own valid length, not the batch maximum. Both schedules normalize within each molecule, preventing longer sequences from contributing more merely because they contain more tokens. Discounted training remains supervised regression.

## Training Modes

| Mode | Purpose | Configuration |
| --- | --- | --- |
| `finetune` | Load a checkpoint and update all parameters at a smaller learning rate | [finetune_smiles.json](../configs/training/finetune_smiles.json) |
| `scratch` | Randomly initialize and train one depth with the selected schedule | [scratch_smiles.json](../configs/training/scratch_smiles.json) |
| `curriculum` | Warm up, branch by depth, then apply discounted training | [curriculum_smiles.json](../configs/training/curriculum_smiles.json) |

Each phase creates a fresh Adam optimizer and cosine scheduler, with a default minimum learning rate of `1e-6` and gradient norm limit of 1.0. Transitions use the preceding phase's lowest-validation-loss `best.pt`. Depth 2/4/6 branches independently start from the same warm-up weights. Training and validation share the fixed depth and temporal schedule; validation disables dropout. Epoch losses are averaged by molecule count, and validation loss selects the best checkpoint.

| Phase | Depth | Default epochs | Initial LR | Schedule |
| --- | --- | --- | --- | --- |
| Warm-up | 2 | 2 | 1e-5 | linear |
| Branch training | Each of 2, 4, 6 | 10 | 1e-4 | linear |
| Discounted training | Same branch depth | 10 | 2e-5 | discounted |

`TrainingConfig.stage3_batch_size` defaults to 1,024; the supplied Colab configuration and notebooks use 8 to assess memory requirements first. Changes to batch size, accumulation, or epochs change optimization trajectories and do not guarantee reproduction of published weights. Standalone fine-tuning defaults to discounted supervision; set `schedule` explicitly for linear supervision.

## Python API

```python
from simpro_vn.training import TrainingConfig, run_training

config = TrainingConfig(
    dataset_path="examples/data/value_network_smoke.csv",
    output_dir="training_runs/finetune_001",
    checkpoint_path="weights/smiles/value_network_best_smiles_bpe_min_freq_5000_d6.pt",
    representation="smiles", min_frequency=5000, depth=6,
    epochs=10, batch_size=8, learning_rate=2e-5,
)
result = run_training(config)

from simpro_vn import ValueNetworkPredictor
predictor = ValueNetworkPredictor.from_config(
    result["directory"] + "/inference_config.json"
)
print(predictor.predict("CCO"))
```

JSON paths resolve relative to the configuration file; relative paths passed to `TrainingConfig` resolve from the working directory. Fine-tuning derives network dimensions and context length from the checkpoint; `nhead` must match and `dropout` is configurable. `d_model`, `dim_feedforward`, and `max_seq_len` configure randomly initialized networks only. New training checkpoints verify representation, tokenization, depth, and preprocessing metadata when loaded.

## Saving and Resuming

The output directory must not exist and must be outside the project's `weights/` directory. Initialization weights are read-only. Only `best.pt` and `last.pt` inside the newly created run are updated. Failed preflight retains the run directory for inspection; use a new directory after correcting errors.

```text
training_runs/<run>/
  training_config.json
  environment.json
  data/{train.jsonl,val.jsonl,groups.sqlite,rejections.jsonl,summary.json}
  model/                         # Single phase
    best.pt
    last.pt
    phase_config.json
    inference_config.json
    history.json
  result.json
```

Curricula use phase directories such as `warmup_d2/`, `linear_d2/`, and `discounted_d2/`. `best.pt` contains the model and selection metrics. `last.pt` additionally stores optimizer, scheduler, AMP scaler, epoch, data fingerprint, random states, and the best-model snapshot, making it larger. Both support inference; only `last.pt` supports resuming.

New checkpoints describe architecture, dimensions, attention heads, depth, special token IDs, property order, and bundle fingerprints. Published weights are identified by SHA256; other training checkpoints use embedded `simpro_vn` metadata, checked against tensor shapes and tokenizer. An unknown raw state_dict only permits shape checks, which cannot establish its depth or preprocessing identity. See [MODEL_INTERFACE.md](MODEL_INTERFACE.md).

```python
from dataclasses import replace

# Bound this invocation while preserving the planned scheduler horizon.
partial = run_training(config, max_epochs=1)
continued = run_training(replace(
    config,
    checkpoint_path=None,
    resume_from=partial["last_checkpoint"],
    output_dir="training_runs/finetune_001_resumed",
))
```

This resume illustration requires a previously unused directory for its initial run. The CLI equivalent is `--max-epochs 1`. Resuming requires unchanged data, splits, total planned epochs, training settings, and device type. The best snapshot inside last.pt reconstructs best in the new directory and continues model selection without depending on the companion best file's update state. To train beyond a completed schedule, start a new fine-tuning run from its best checkpoint.

A curriculum does not automatically resume its full pipeline. Load a phase's `phase_config.json`, clear `checkpoint_path`, set `resume_from` and a new `output_dir`, and resume that phase individually. Explicitly initialize subsequent phases from the resulting best checkpoint.

## Memory and Validation

`batch_size` controls molecules per forward pass; `accumulation_steps` controls batches per optimizer update. Incomplete final groups still update, normalized by their actual molecule count. Precision defaults to fp32; CUDA supports fp16 and hardware-supported bf16. History records fp16 overflow skips; nonfinite loss or fp32/bf16 gradients stop training.

`log_every_steps` defaults to progress output every 100 optimizer groups, plus the first group. history.json records epoch train/validation losses, learning rates, and fp16 overflow skips.

Training uses a single device and does not include distributed training, automatic hyperparameter search, or automatic dataset generation. CUDA, package versions, and hardware can affect numerical results even with identical settings. See [TRAINING_VALIDATION.md](TRAINING_VALIDATION.md) for validation procedures and coverage.

Use [Comprehensive Colab Validation](../examples/colab_comprehensive_validation.ipynb) to check installation, CLI, Python APIs, and training together. Its production-model fine-tuning uses four records to verify saving, reload, and resume; it does not establish full-training results or accuracy.
