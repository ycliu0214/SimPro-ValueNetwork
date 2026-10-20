# Third-Party Software and Data Notices

The project's original source code is licensed under [Apache-2.0](LICENSE). That declaration does not relicense external packages or third-party material. This inventory was reviewed on 2026-10-20 against the primary upstream license sources linked below.

## Included Adaptation

`src/simpro_vn/representations/safe_core.py` includes modified SAFE converter logic from the [SAFE project](https://github.com/datamol-io/safe), including the `SAFEConverter` structure and slicing SMARTS. The upstream code is licensed under [Apache-2.0](https://github.com/datamol-io/safe/blob/main/LICENSE). Local modifications replace datamol calls with RDKit helpers and provide project-specific exceptions, splitting, and conversion wrappers. Attribution is recorded in [NOTICE](NOTICE), and the Apache license text is included in [LICENSE](LICENSE).

This is an included code adaptation, not merely an installed dependency. The exact historical upstream revision used during earlier development is not recorded. This attribution identifies the matched upstream implementation and does not claim that the local converter is the latest upstream version. No SAFE-GPT weights or SAFE training dataset are included in the source archive; their separate licenses do not describe SimPro-ValueNetwork's own checkpoints.

## Separately Installed Dependencies

The source archive and the project's Python wheel do not bundle the following libraries' source trees or distribution binaries. pip resolves them separately. The selected package versions, their dependencies, and any bundled native components retain their own licenses and notices.

| Package | Use | Upstream license reference |
| --- | --- | --- |
| PyTorch (`torch`) | Model inference and training | [BSD-style license and notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Hugging Face Tokenizers (`tokenizers`) | Loading and running the fixed tokenizers | [Apache-2.0](https://github.com/huggingface/tokenizers/blob/main/LICENSE) |
| RDKit (`rdkit`) | Chemistry helpers, representations, and optional label calculation | [BSD-3-Clause](https://github.com/rdkit/rdkit/blob/master/license.txt) |
| SELFIES (`selfies`) | SELFIES splitting and conversion | [Apache-2.0](https://github.com/aspuru-guzik-group/selfies/blob/master/LICENSE) |
| NumPy (`numpy`) | Optional chemistry/data operations | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| pandas (`pandas`) | Optional CSV preparation | [BSD-3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| FreeSASA (`freesasa`) | Optional surface-area label calculation | [MIT; upstream project](https://freesasa.github.io/) |
| tqdm (`tqdm`) | Optional preparation progress bars | [MPL-2.0 and MIT notices applying to respective contributions](https://github.com/tqdm/tqdm/blob/master/LICENCE) |
| setuptools | Build backend only | [MIT](https://github.com/pypa/setuptools/blob/main/LICENSE) |

RDKit's installed Contrib SA scorer is imported through `rdkit.Contrib.SA_Score.sascorer`; its implementation and scoring data are not copied into this repository. See the [scorer source and notices](https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py). Other RDKit data used through installed APIs, including filter catalogs, remain part of the installed RDKit distribution.

pip and venv are used by the installation validation workflow; Google Colab supplies the optional `google.colab` integration. These tools are not vendored. Installing `tokenizers` can also install transitive packages such as `huggingface-hub`; that does not add a model-hosting or automatic-checkpoint-download feature to this project.

This is an inventory of direct declared dependencies and the identified included adaptation, not a complete license bill of materials for every environment. The project does not pin one immutable transitive dependency graph. Inspect the license files shipped with the actual versions installed when redistributing an environment, container, executable, or dependency bundle. A source-only release and a container containing all dependencies have different distributed contents.

## Datasets and Model Weights

The study uses ZINC250k and MOSES. Original datasets are not included in the source archive; `examples/data/value_network_smoke.csv` is a 300-record, processed sample with molecular strings and calculated labels. Its redistribution terms must be checked against the actual upstream dataset versions and acquisition sources before publishing the sample. The source-code Apache declaration does not establish or replace upstream data permissions.

Upstream references: [MOSES](https://github.com/molecularsets/moses) and [ZINC](https://zinc.docking.org/). These references identify data sources, not a grant of rights over every derivative file.

The source archive excludes pretrained .pt files. The 60 SimPro-ValueNetwork checkpoints identified by `weights/manifest.json` are explicitly licensed under Apache-2.0 in [Pretrained Checkpoint License](weights/WEIGHTS_LICENSE.md). Model archives include this declaration, LICENSE, and applicable notices. This grant applies to this project's identified checkpoints; it does not relicense upstream datasets, the processed sample CSV, or third-party pretrained models.
