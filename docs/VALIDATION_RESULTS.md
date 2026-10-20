# Recorded Functional Validation

SimPro-ValueNetwork 0.1.0 completed its comprehensive Colab functional validation on 2026-10-20. The executed notebook and machine-readable reports were reviewed together. This page summarizes the results without workstation paths, Drive layouts, runtime identifiers, or raw notebook outputs.

## Environment

- Python 3.13.15.
- PyTorch 2.11.0+cu130 with CUDA available.
- Package imported from a normal installation, outside the project source tree.
- Separate wheel installation validated in a temporary venv with shared dependencies.

## Results

| Check | Recorded result |
| --- | --- |
| Overall | PASS; all 19 required phases present |
| Functional tests | 50 passed; 0 failures, 0 errors, 0 skips |
| Released model matrix | All 60 representation/tokenization/depth combinations covered |
| Matrix CLI checks | 182 passed |
| Matrix Python API checks | 360 passed |
| Checkpoint identity | All 60 published checkpoints identified by SHA256 |
| Installation | Wheel build, venv creation, wheel installation, assets/API, CLI help, training help, and CLI prediction passed |
| Integration | Tensor/ID/cache interfaces, fixed-score gradients, and CPU/CUDA agreement passed |
| Linear temporal loss | Expected weights, hand-calculated loss/gradients, and curriculum integration passed |
| CUDA training precision | fp32, fp16, and bf16 tests passed |
| Data preparation | Four input records expanded to five accepted records; no stage rejections |
| Scratch training | CLI training, saving, and prediction reload passed |
| Production-model fine-tuning | One epoch completed, then resumed to epoch two; best/last reload passed |
| Original weight preservation | All 63 local checkpoint files unchanged, including three local archive checkpoints |

The tested production fine-tuning used two Train and two Val records. Its validation loss was 0.45380449 at epoch one and 0.45524389 at epoch two, so best remained epoch one. This verifies checkpoint selection and resume behavior; it does not demonstrate improved predictive accuracy.

## Scope

These are functional and consistency results, not a new scientific accuracy evaluation or a full-dataset retraining. They do not establish compatibility with every operating system, dependency version, or GPU. Public notebooks are clean, unexecuted examples; use [Comprehensive Colab Validation](../examples/colab_comprehensive_validation.ipynb) to generate results for another environment. Research results are documented separately in [Research Results](RESEARCH_RESULTS.md).
