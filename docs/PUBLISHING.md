# Repository and Release Contents

The source repository contains inference code, fixed preprocessing assets, configuration files, documentation, tests, and five clean example notebooks. Model binaries are supplied separately as release assets.

## Commit to the Source Repository

| Path | Contents |
| --- | --- |
| `README.md`, `pyproject.toml`, `.gitignore`, `.gitattributes` | Introduction, installation metadata, exclusions, and byte-preserving data checkout rules |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` | Apache-2.0 text, included-code attribution, and dependency/data scope |
| `src/simpro_vn/` | Python package, model architecture, inference/training APIs, CLI, and bundled JSON assets |
| `configs/` | All inference configurations and three training examples |
| `tests/` | Functional inference, training, resume, and CUDA checks |
| `docs/` | User documentation, research figures, model catalog, validation results, and publishing guidance |
| `weights/README.md` | Checkpoint layout and loading instructions |
| `weights/WEIGHTS_LICENSE.md` | Apache-2.0 grant for the 60 manifest-listed checkpoints |
| `weights/manifest.json`, `weights/model_descriptions.json` | Checkpoint integrity and model descriptions |
| `examples/*.py` | Prediction, data preparation, and validation helpers used by the notebooks |
| `examples/data/` | The 300-record smoke CSV and its explanation |

Retain these five notebooks:

- `examples/colab_inference.ipynb`.
- `examples/prepare_finetuning_dataset.ipynb`.
- `examples/colab_training.ipynb`.
- `examples/colab_finetuning.ipynb`.
- `examples/colab_comprehensive_validation.ipynb`.

Do not remove validation helper scripts or tests: the comprehensive notebook uses them. Tokenizer/mapping/scaler files and both model-description copies are required assets, not disposable duplicates.

## Distribute Separately

The 60 best pretrained checkpoints are licensed under Apache-2.0 and belong in release assets rather than ordinary source commits. Four archives, one for each representation, allow selective downloads. Preserve checkpoint filenames and extract into the project root to retain `weights/<representation>/` paths. Every archive includes LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, weights/WEIGHTS_LICENSE.md, inventory/descriptions, and member SHA256 checksums. Publish the archive SHA256SUMS.txt alongside the four ZIPs. No automatic download service is required.

The source archive intentionally omits .pt files. Downloading source alone therefore does not supply pretrained models. Release asset names and links should be added to the README when the release exists; do not publish placeholder download links.

## Keep Local

- `weights/archive/`: earlier development checkpoints, not part of the 60-model release.
- `experimental/`: unsupported MOLCANO code with no public inference model.
- Overlapping notebooks `colab_validation.ipynb`, `colab_package_validation.ipynb`, and `colab_training_validation.ipynb`; the comprehensive notebook covers their checks.
- Executed notebooks, raw validation ZIPs, training outputs, analysis caches, full datasets, and prepared-data outputs.
- Credentials, `.env` files, SSH keys, local editor state, build products, and temporary runtime files.

These files may remain in the local checkout but are excluded from the source archive and ignore rules. The earlier validated notebooks and reports should be retained privately as evidence; [Validation Results](VALIDATION_RESULTS.md) provides the public summary.

## Before Publishing

Inspect the staged file list rather than uploading the entire working folder. A .gitignore affects Git, but does not filter files manually selected in a browser upload. The prepared source ZIP contains only the intended source files and should be extracted before committing through GitHub's browser interface; uploading the ZIP alone does not create a browsable source repository.

The original project code and the 60 manifest-listed checkpoints are licensed under Apache-2.0. Keep LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, and weights/WEIGHTS_LICENSE.md in source and wheel distributions, and include them with model archives. Setuptools 77.0.3 or newer is required to build the declared SPDX/license-file metadata. Publication of the processed dataset sample still requires confirming upstream redistribution permissions. See [Third-Party Notices](../THIRD_PARTY_NOTICES.md). Funding acknowledgments are intentionally retained.
