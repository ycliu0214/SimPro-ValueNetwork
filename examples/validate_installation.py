"""Build a wheel, install it separately, and validate imports/API/CLI outside the source tree.

The temporary venv reuses installed dependencies, while simpro_vn itself must
resolve to the newly installed wheel. No published checkpoint is required.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


SMOKE_CODE = r'''
import importlib.metadata
import json
from pathlib import Path
import sys
import torch
import simpro_vn
from simpro_vn import ValueNetworkPredictor
from simpro_vn.bundles import ASSETS, resolve_bundle
from simpro_vn.model import RecurrentValueNetwork
from simpro_vn.representations import get_splitter
from simpro_vn.scaling import PROPERTY_NAMES
from simpro_vn.tokenization import SurrogateChemicalTokenizer

environment = Path(sys.prefix).resolve()
module = Path(simpro_vn.__file__).resolve()
assert environment in module.parents, f"Package was not loaded from the isolated wheel installation: {module}"
distribution = importlib.metadata.distribution('simpro_vn')
assert environment in Path(distribution.locate_file('')).resolve().parents or Path(distribution.locate_file('')).resolve() == environment
assert len(list((ASSETS / 'tokenizers').glob('*.json'))) == 40
assert (ASSETS / 'value_network_scaler.json').is_file()
catalog = json.loads((ASSETS / 'pretrained_models.json').read_text())
assert len(catalog['models']) == 60
prefix, _, fingerprint = resolve_bundle('smiles', 5000)
tokenizer = SurrogateChemicalTokenizer(prefix, get_splitter('smiles'))
torch.manual_seed(42)
model = RecurrentValueNetwork(tokenizer.vocab_size, 8, 2, 16, 0., 17, 16)
metadata = {'representation': 'smiles', 'min_frequency': 5000, 'depth': 2, 'nhead': 2,
            'property_names': list(PROPERTY_NAMES), 'bundle_fingerprint': fingerprint}
checkpoint = Path.cwd() / 'installation_smoke.pt'
torch.save({'model_state_dict': model.state_dict(), 'simpro_vn': metadata}, checkpoint)
predictor = ValueNetworkPredictor(checkpoint_path=checkpoint, depth=2, device='cpu')
records = predictor.predict(['CCO', 'C'])
assert len(records) == 2 and len(records[0]) == 17
values = predictor.predict_tensor(['CCO', 'C'])
assert values.shape == (2, 17) and not values.requires_grad
ids = torch.tensor([predictor.encode('CCO')])
torch.testing.assert_close(predictor.predict_ids(ids), predictor.predict_tensor('CCO'))
torch.testing.assert_close(predictor.new_session().step_tensor(ids[0])[-1:], predictor.predict_ids(ids))
assert predictor.describe()['nhead'] == 2
(Path.cwd() / 'api_results.json').write_text(json.dumps(predictor.predict('CCO')))
print(json.dumps({'status': 'PASS', 'package': str(module), 'checkpoint': str(checkpoint),
                  'properties': len(PROPERTY_NAMES), 'bundled_models': len(catalog['models'])}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for installation-check reports")
    args = parser.parse_args()
    project = args.project_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (project / "pyproject.toml").is_file():
        raise FileNotFoundError("project-dir must contain pyproject.toml")
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    records = []

    def save_report(status, error=None):
        summary = {"status": status, "checks": records,
                   "scope": "Isolated simpro_vn wheel; dependencies reused from the current environment"}
        if error is not None:
            summary["error"] = error
        (output / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    save_report("IN_PROGRESS")

    def run(name, command, cwd):
        result = subprocess.run([str(v) for v in command], cwd=cwd, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        (output / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (output / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        records.append({"check": name, "returncode": result.returncode})
        save_report("IN_PROGRESS" if result.returncode == 0 else "FAIL")
        print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
        if result.returncode:
            print(result.stdout, result.stderr, flush=True)
            result.check_returncode()
        return result

    try:
        with tempfile.TemporaryDirectory(prefix="simpro_install_") as temporary:
            work = Path(temporary)
            wheels = work / "wheels"
            wheels.mkdir()
            run("build_wheel", [sys.executable, "-m", "pip", "wheel", project, "--no-deps",
                                "--no-build-isolation", "--wheel-dir", wheels], work)
            wheel = next(wheels.glob("simpro_vn-*.whl"))
            # Avoid ensurepip: Colab may not provide its bundled pip bootstrap.
            environment = work / "venv"
            run("create_venv", [sys.executable, "-m", "venv", "--without-pip",
                                "--system-site-packages", environment], work)
            bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
            python = bin_dir / ("python.exe" if os.name == "nt" else "python")
            # Host pip explicitly targets the venv; never uninstall the host copy.
            run("install_wheel", [sys.executable, "-m", "pip", "--python", python,
                                  "install", "--no-index", "--no-deps", "--ignore-installed", wheel], work)
            script = work / "smoke.py"
            script.write_text(SMOKE_CODE, encoding="utf-8")
            run("installed_api_and_assets", [python, script], work)
            cli = bin_dir / ("simpro-vn.exe" if os.name == "nt" else "simpro-vn")
            training_cli = bin_dir / ("simpro-vn-train.exe" if os.name == "nt" else "simpro-vn-train")
            run("installed_cli_help", [cli, "--help"], work)
            run("installed_training_cli_help", [training_cli, "--help"], work)
            result = run("installed_cli_prediction", [cli, "--checkpoint", work / "installation_smoke.pt",
                                                      "--depth", "2", "--text", "CCO", "--device", "cpu"], work)
            predicted = json.loads(result.stdout)
            expected = json.loads((work / "api_results.json").read_text())[0]
            if len(predicted) != 1 or any(abs(predicted[0][k] - expected[k]) > 1e-5 for k in expected):
                raise AssertionError("Installed CLI/API predictions differ")
    except Exception:
        save_report("FAIL", traceback.format_exc())
        raise
    summary = save_report("PASS")
    print(json.dumps(summary, indent=2))
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
