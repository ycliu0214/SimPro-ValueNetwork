# Installation and Interface Validation

For package-only checks, follow these Colab cells. PROJECT_DIR points to the project root containing pyproject.toml. No sys.path.insert() is used. These commands use temporary tiny checkpoints and do not require released model weights.

```python
from pathlib import Path
from uuid import uuid4
import sys
PROJECT_DIR = Path('/content/SimPro-ValueNetwork')
REPORT_DIR = Path('/content/validation_runs') / ('installation_' + uuid4().hex[:8])
```

```python
!{sys.executable} -m pip install "{PROJECT_DIR}[chemistry]"
!{sys.executable} -m unittest discover -s "{PROJECT_DIR}/tests" -p test_inference_interfaces.py -v
!{sys.executable} "{PROJECT_DIR}/examples/validate_installation.py" --project-dir "{PROJECT_DIR}" --output-dir "{REPORT_DIR}"
```

Confirm unittest reports OK and REPORT_DIR/report.json has status PASS. Colab shell commands may not stop later cells after errors; cell completion alone is not a pass. The notebook separately checks command exit codes.

## Coverage

- Normal installation builds a wheel, without editable installation.
- An additional temporary venv installs the wheel and verifies the actual import location.
- API, CLI, and both CLI help commands run outside the project directory.
- Installed assets include 40 tokenizer/mapping JSONs, the scaler, and descriptions of 60 models.
- Tiny checkpoints verify reload and tensor/dictionary/ID/incremental agreement.
- Tests cover conflicting descriptions, wrong depth, masks, IDs, padding, and fixed-score tensor gradient behavior.

The temporary venv reuses existing dependencies such as PyTorch to avoid large downloads. It verifies an independent simpro_vn installation, not every dependency version or a blank operating system. The script does not download published weights, train production models, or modify weights/. Logs are saved in the selected new directory.

The venv is created with `--without-pip`; the runtime's pip then targets its interpreter using `--python` to install the wheel, without depending on ensurepip. This requires pip 22.3 or newer and retains the isolated-installation check. Environment-creation failures and other exceptions record FAIL in report.json, with command stdout/stderr saved separately. Use a new output directory when retrying; rerun the notebook's first setup cell to create one.

For released weights, use [colab_comprehensive_validation.ipynb](../examples/colab_comprehensive_validation.ipynb). Loading checks include SHA256 identification; batch checks include tensor/ID API comparisons. Public notebooks contain no saved execution outputs; results must come from the current run.

For all 60 released models, data preparation, and training in one run, use [Comprehensive Validation](COMPREHENSIVE_VALIDATION.md) and its [Colab notebook](../examples/colab_comprehensive_validation.ipynb).
