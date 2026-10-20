# Preparing User Data

Prediction only needs strings in the selected representation. Geometry and property labels are needed for preparing fine-tuning data. This workflow is an optional learning example.

## Starting from SMILES

```text
SMILES → validation → stereo enumeration → 3D optimization
       → 17 properties → fixed-bundle compatibility → Train/Val split
```

1. Read CSV, parse SMILES, and clear atom-map numbers.
2. Enumerate unspecified stereochemistry when completeness is uncertain, capped at 32 per input without intentionally flipping specified stereo. This does not identify experimental isomers or guarantee exhaustive enumeration.
3. Optimize every retained molecule, including those without stereocenters or with specified stereo. The example uses AddHs, embedding seed 42, and MMFF94 with 1,000 iterations; only converged results are accepted.
4. Calculate all 17 labels from those optimized conformers; model predictions are not labels.
5. Check fixed tokenizer/mapping/scaler compatibility and context length.
6. Keep stereoisomers of one parent together when splitting Train/Val.

## Colab Example

Use [prepare_finetuning_dataset.ipynb](../examples/prepare_finetuning_dataset.ipynb) with `finetuning_data.py`. This SMILES example uses CPU RDKit MMFF94.

```bash
pip install ".[data]"
```

Set `PROJECT_DIR`, `INPUT_CSV`, `SMILES_COLUMN`, model setting, and checkpoint path. `INPUT_CSV=None` uses built-in samples. CSV reading defaults to 100 rows; use `MAX_INPUT_ROWS=None` for the full input.

## Fixed Preprocessing

Tokenizer/mapping replacement or retraining and scaler refitting are unsupported. Custom paths must point to identical bundle copies. Every row must encode, round trip, fit the context limit, and have 17 finite labels yielding finite z-scores.

Excessive length, unknown tokens, optimization failures, and missing labels are logged as rejections. No truncation, vocabulary expansion, or zero-label imputation occurs. `Max_Absolute_ZScore` is a distribution diagnostic, not an accuracy guarantee.

## Outputs and Next Steps

`prepared_datasets/<timestamp>/` contains expanded inputs, optimized conformers, label CSVs, rejection records, settings, and `finetuning_dataset.csv`. Final labels remain in property units with `Split_Set`; use the model's scaler for standardization.

At least two independent parent groups are required. The example generates single conformers, not guaranteed global minima or complete geometry screening. Use its CSV in the [fine-tuning notebook](../examples/colab_finetuning.ipynb); see [TRAINING.md](TRAINING.md) for training settings and checkpoint storage.

Already-labeled data can use the compatibility check after aligning [property definitions and units](PROPERTIES.md). Different conformers and calculation settings can produce different 3D labels.

Run the small complete workflow in [Comprehensive Colab Validation](../examples/colab_comprehensive_validation.ipynb), covering stereoisomer expansion, 3D optimization, 17 properties, compatibility, parent splitting, and per-stage rejection logs.
