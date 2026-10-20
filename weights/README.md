# Pretrained Weights

The 60 best pretrained checkpoints listed in [manifest.json](manifest.json) are licensed under **Apache-2.0**. See [Pretrained Checkpoint License](WEIGHTS_LICENSE.md), the complete [LICENSE](../LICENSE), and [Third-Party Notices](../THIRD_PARTY_NOTICES.md). Dataset permissions and third-party software licenses remain separate.

There are 60 best models: four representations × five tokenization settings × depths 2/4/6, totaling approximately 1.97 GB.

```text
weights/
  smiles/
  smarts/
  safe/
  selfies/
  manifest.json
  model_descriptions.json
  WEIGHTS_LICENSE.md
```

## Filenames

```text
<representation>/value_network_best_<representation>_bpe_min_freq_<setting>_d<depth>.pt
<representation>/value_network_best_<representation>_no_bpe_d<depth>.pt
```

Each representation includes 5000, 10000, 20000, 50000, and no_bpe at depths 2/4/6. Configurations select the corresponding best weights.

## Release Archives

Each archive contains 15 checkpoints for one representation:

- `SimPro-ValueNetwork-smiles-weights.zip`.
- `SimPro-ValueNetwork-smarts-weights.zip`.
- `SimPro-ValueNetwork-safe-weights.zip`.
- `SimPro-ValueNetwork-selfies-weights.zip`.

Extract the selected archive into the project root containing pyproject.toml. Its members use `weights/<representation>/...` paths. LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, this checkpoint declaration, the inventory/descriptions, and SHA256SUMS.txt accompany each archive. Download all four representations for the comprehensive 60-model validation. Ordinary inference only needs its selected checkpoint and matching package assets.

## Loading

Use `checkpoint_path` or CLI `--checkpoint` for any readable location. Representation, setting, depth, and preprocessing assets must match; matching tensor shapes do not make models interchangeable.

`manifest.json` records paths, sizes, and SHA256 checksums. .pt/.pth files are excluded from ordinary Git tracking and may be distributed through releases or external storage.

model_descriptions.json records architecture, depth, attention heads, special token IDs, property order, and preprocessing fingerprints for all 60 models. The package includes the same descriptions and identifies published bytes by SHA256 to validate configuration. It neither modifies .pt files nor requires specific filenames. See [Model Descriptions and Tensor Interfaces](../docs/MODEL_INTERFACE.md).

## Fine-Tuned Weights

Training requires a new output directory outside this directory. best.pt stores the lowest-validation-loss model; last.pt also contains resume state. Both load in the prediction API/CLI, and inference_config.json records model settings. Training does not modify initialization checkpoints.

Resume from last.pt into another new output directory. Its best-model snapshot reconstructs the preceding best checkpoint. See [Training](../docs/TRAINING.md).
