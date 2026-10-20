# SimPro-ValueNetwork — Comprehensive Validation

Use [colab_comprehensive_validation.ipynb](../examples/colab_comprehensive_validation.ipynb) to check installation, console commands, Python APIs, data preparation, and training examples in one ordered run. It checks all 60 released checkpoints by default and contains no prefilled execution results.

## Procedure

1. Open a fresh Colab runtime, preferably with a GPU, and upload/extract the full project to `/content/SimPro-ValueNetwork` or another chosen location.
2. Open the notebook, set `PROJECT_DIR`, and optionally set `WEIGHTS_DIR` to another root while preserving representation subdirectories and catalog filenames.
3. Select Run all. Normal installation and CLI commands use `!`, followed by Python API cells. Do not rely on imports from an earlier runtime or source-path shortcuts.
4. Inspect the final `summary.json` and download `reports.zip`. Copy new training subdirectories separately to your own persistent storage if needed.

Google Drive is optional; enable `USE_GOOGLE_DRIVE` and supply your own paths. No released weights are downloaded and no private Drive layout is required. `PROJECT_DIR` must contain `pyproject.toml`; the weight root must contain `smiles/`, `smarts/`, `safe/`, and `selfies/`.

Isolated installation uses `venv --without-pip` and the current pip's `--python` option (pip 22.3 or newer), without calling ensurepip. After applying a fix, rerun cells in order from the first setup cell to create a new RUN_DIR, avoiding existing installation/functional_tests directories and previous FAIL records.

## Coverage

| Check | Details |
| --- | --- |
| Normal installation | Install the project with its data extra, import the installed copy, and check console/module entrypoints |
| Wheel | Temporary venv outside the source tree, bundled assets, and CLI/API reload agreement, reusing installed dependencies |
| Functional tests | Discover every unittest file and record counts, failures, errors, and skip reasons |
| Files and descriptions | Verify 60 checkpoints, 40 tokenizer/mapping files, scaler, descriptions, and installed copies |
| Released-model CLI | 182 matrix checks, plus direct console examples for complete strings, prefixes, trajectories, and file input |
| Released-model API | 360 checks, giving 542 combined matrix checks covering tensors/IDs, batches, relative paths, prefixes, caches, reset, and rejection |
| Integration | Standardization, padding masks, tensor caches, fixed-score gradient behavior, and one CPU/GPU comparison when CUDA is selected |
| Data preparation | Small SMILES input, stereo expansion, 3D optimization, 17 properties, bundle compatibility, parent grouping, and rejection logs |
| Training and fine-tuning | A real-label subset, tiny scratch training via CLI, production-model Python fine-tuning/resume, and best/last reload |
| Preservation and reports | New output directories and before/after hashes of every original .pt |

Curriculum depth branches and CUDA precision tests use tiny models. Production fine-tuning selects two Train and two Val rows from the bundled dataset with a two-epoch schedule split across interruption and resume. It does not reproduce full-dataset training. Data preparation supports SMILES; other representations are covered by inference and training compatibility tests.

## Interpretation

- `PASS`: every required phase completed with no skipped capabilities.
- `PASS_WITH_SKIPS`: executed checks passed, but capabilities such as CUDA/bf16 remain unverified; inspect `skipped_capabilities`.
- `FAIL` or `IN_PROGRESS`: a failure or incomplete run cannot establish complete validation.

A failed phase stops execution and records completed phases and its exception; an assertion or interrupted installation may leave an in-progress report. Inspect the failing cell and logs, then rerun in a fresh runtime/new directory after fixes. Do not relax tolerances just to obtain PASS. Runtime depends on hardware and storage; CPU can be slow. For GPU OOM, first use a fresh runtime and ensure other models are not occupying memory.

`reports.zip` excludes .pt/.pth and contains reports, configurations, small data, and logs. Colab temporary files disappear with the runtime; download what you need. See [Research Results](RESEARCH_RESULTS.md) and [Reproduction Inputs](REPRODUCING_RESULTS.md) for scientific results and full analysis figure reproduction.
