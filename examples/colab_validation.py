"""Colab validation helpers. Importing this file does not run any checks.

Run CLI checks first, then Python API checks from colab_validation.ipynb.
No test framework or training/evaluation dataset is required.
"""

import csv
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


PROPERTIES = (
    "MW", "LogP", "HBD", "HBA", "ROB", "TPSA", "AROM", "QED", "SAS",
    "ALERTS", "Asphericity", "NPR1", "NPR2", "SNDM", "3D_Polar_SA",
    "3D_NonPolar_SA", "Total_SASA",
)
SAMPLES = {
    "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
    "smarts": ["[#6]-[#6]-[#8]", "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1", "[#6](=[#8])-[#8]"],
    "safe": ["CCO", "c1ccccc1", "S6(=O)(=O)c1cccc8c1.C5C4=O.CN6C.C38=O.C17CC1.O35.N47"],
    "selfies": ["[C][C][O]", "[C][=C][C][=C][C][=C][Ring1][=Branch1]", "[C][C][=Branch1][C][=O][O]"],
}
PREFIXES = {"smiles": "CC", "smarts": "[#6]-[#6]", "safe": "CC", "selfies": "[C][C]"}


def stem_for(rep, setting):
    return f"{rep}_no_bpe" if setting == "no_bpe" else f"{rep}_bpe_min_freq_{setting}"


class ColabValidation:
    def __init__(self, project_dir, weights_dir, output_dir, *, representations,
                 settings, depths, device="auto", rtol=1e-4, atol=1e-4,
                 cli_timeout=300):
        self.project = Path(project_dir).resolve()
        self.weights = Path(weights_dir).resolve()
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.device, self.rtol, self.atol = device, rtol, atol
        self.cli_timeout = cli_timeout
        self.cases = [
            {"rep": rep, "setting": setting, "depth": depth,
             "name": f"{rep}_{setting}_d{depth}",
             "checkpoint": self.weights / rep / f"value_network_best_{stem_for(rep, setting)}_d{depth}.pt"}
            for rep in representations for setting in settings for depth in depths
        ]
        if not self.cases:
            raise ValueError("Select at least one model")
        self.records = []
        self.cli_results = {}
        self.console = None
        self.stds = None
        self.env = os.environ.copy()
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.env["TOKENIZERS_PARALLELISM"] = "false"

    def record(self, stage, model, check, action):
        started = time.perf_counter()
        row = {"stage": stage, "model": model, "check": check}
        try:
            detail = action()
            row.update(status="PASS", detail=detail)
        except Exception:
            row.update(status="FAIL", detail=traceback.format_exc())
        row["seconds"] = round(time.perf_counter() - started, 3)
        self.records.append(row)
        self.save()
        print(f'[{row["status"]}] {stage} | {model} | {check}', flush=True)
        if row["status"] == "FAIL":
            print(row["detail"], flush=True)
        return row["status"] == "PASS"

    def save(self):
        data = {"python": sys.version, "device_requested": self.device,
                "selected_models": [c["name"] for c in self.cases],
                "rtol": self.rtol, "atol": self.atol, "records": self.records}
        (self.output / "report.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        with (self.output / "report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "model", "check", "status", "seconds", "detail"])
            writer.writeheader()
            for row in self.records:
                writer.writerow({**row, "detail": json.dumps(row["detail"], ensure_ascii=False)})

    def preflight(self):
        """Check files and installed CLI without importing the prediction API."""
        self.console = shutil.which("simpro-vn")
        if self.console is None:
            raise RuntimeError("simpro-vn entry point is absent; rerun the installation cell")
        missing = []
        for case in self.cases:
            prefix = self.project / "src/simpro_vn/assets/tokenizers" / stem_for(case["rep"], case["setting"])
            files = [case["checkpoint"], Path(str(prefix) + "_tokenizer.json"),
                     Path(str(prefix) + "_mapping.json"),
                     self.project / "configs" / f'{case["name"]}.json']
            missing.extend(str(p) for p in files if not p.is_file())
        scaler = self.project / "src/simpro_vn/assets/value_network_scaler.json"
        if not scaler.is_file():
            missing.append(str(scaler))
        if missing:
            raise FileNotFoundError("Missing files:\n" + "\n".join(sorted(set(missing))))
        stats = json.loads(scaler.read_text(encoding="utf-8"))
        self.stds = [float(stats[name]["std"]) for name in PROPERTIES]
        versions = {name: importlib.metadata.version(name)
                    for name in ("simpro_vn", "torch", "tokenizers", "selfies", "rdkit")}
        (self.output / "environment.json").write_text(
            json.dumps({"python": sys.version, "executable": sys.executable,
                        "console": self.console, "packages": versions}, indent=2), encoding="utf-8",
        )
        print(f"Preflight complete: {len(self.cases)} models. CLI: {self.console}")
        return versions

    def _run(self, command, label, expect_success=True):
        result = subprocess.run(
            [str(v) for v in command], cwd=self.output, env=self.env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=self.cli_timeout,
        )
        (self.output / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (self.output / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if expect_success and result.returncode != 0:
            raise RuntimeError(f"Exit {result.returncode}: {result.stderr}\n{result.stdout}")
        if not expect_success and result.returncode == 0:
            raise AssertionError("Invalid CLI arguments unexpectedly succeeded")
        return result

    def _base(self, case):
        return [self.console, "--checkpoint", case["checkpoint"], "--representation", case["rep"],
                "--min-frequency", str(case["setting"]), "--depth", str(case["depth"]),
                "--device", self.device]

    @staticmethod
    def _rows(rows, count=None):
        assert isinstance(rows, list) and rows, "Expected a nonempty list of predictions"
        if count is not None:
            assert len(rows) == count, f"Expected {count} predictions, got {len(rows)}"
        for row in rows:
            assert set(row) == set(PROPERTIES), "Property names do not match the 17 outputs"
            assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in row.values()), "Nonfinite output"

    def _close(self, actual, expected):
        self._rows(actual)
        self._rows(expected, len(actual))
        maximum = 0.0
        for a, b in zip(actual, expected):
            for name, std in zip(PROPERTIES, self.stds):
                difference = abs(a[name] - b[name]) / std
                maximum = max(maximum, difference)
                tolerance = self.atol + self.rtol * abs(b[name] / std)
                assert difference <= tolerance, (
                    f"{name}: standardized difference={difference:.9g}, tolerance={tolerance:.9g}"
                )
        return maximum

    def run_cli(self):
        """Run console commands first; each model gets three real CLI invocations."""
        if self.console is None:
            raise RuntimeError("Run preflight first")
        def help_check():
            result = self._run([self.console, "--help"], "console_help")
            assert "--checkpoint" in result.stdout and "no_bpe" in result.stdout
            result = self._run([sys.executable, "-m", "simpro_vn", "--help"], "module_help")
            assert "--trajectory" in result.stdout
        self.record("CLI", "entrypoints", "console and python -m help", help_check)

        def invalid_args():
            for label, command in [
                ("missing_arguments", [self.console]),
                ("invalid_depth", [self.console, "--checkpoint", "unused.pt", "--depth", "0", "--text", "CCO"]),
                ("conflicting_input", [self.console, "--checkpoint", "unused.pt", "--text", "CCO", "--input-file", "unused.txt"]),
            ]:
                result = self._run(command, label, expect_success=False)
                assert result.returncode == 2, f"Expected argparse exit 2, got {result.returncode}"
        self.record("CLI", "arguments", "invalid arguments are rejected", invalid_args)

        for index, case in enumerate(self.cases, 1):
            name, rep = case["name"], case["rep"]
            print(f'\nCLI model {index}/{len(self.cases)}: {name}', flush=True)
            self.cli_results[name] = {}

            def batch_check():
                input_path, output_path = self.output / f"{name}.input.txt", self.output / f"{name}.batch.json"
                input_path.write_text("\n".join(SAMPLES[rep]) + "\n", encoding="utf-8")
                self._run(self._base(case) + ["--input-file", input_path, "--output", output_path,
                                             "--batch-size", "2"], f"{name}.batch")
                rows = json.loads(output_path.read_text(encoding="utf-8"))
                self._rows(rows, 3)
                self.cli_results[name]["batch"] = rows
            self.record("CLI", name, "file input, JSON output, variable length batch", batch_check)

            def prefix_check():
                result = self._run(self._base(case) + ["--text", PREFIXES[rep], "--text", "", "--prefix"],
                                   f"{name}.prefix")
                rows = json.loads(result.stdout)
                self._rows(rows, 2)
                self.cli_results[name]["prefix"] = rows
            self.record("CLI", name, "repeated text, prefix and empty prefix, stdout JSON", prefix_check)

            def trajectory_check():
                result = self._run(self._base(case) + ["--text", SAMPLES[rep][0], "--trajectory"],
                                   f"{name}.trajectory")
                trajectories = json.loads(result.stdout)
                assert len(trajectories) == 1
                traj = trajectories[0]
                self._rows(traj["predictions"], len(traj["token_ids"]))
                assert len(traj["tokens"]) == len(traj["token_ids"])
                assert traj["tokens"][0] == "[BOS]" and traj["tokens"][-1] == "[EOS]"
                if "batch" in self.cli_results[name]:
                    self._close(traj["predictions"][-1:], self.cli_results[name]["batch"][:1])
                self.cli_results[name]["trajectory"] = traj
            self.record("CLI", name, "trajectory output and final prediction", trajectory_check)
        (self.output / "cli_predictions.json").write_text(
            json.dumps(self.cli_results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

    @staticmethod
    def _raises(exception, action):
        try:
            action()
        except exception:
            return
        raise AssertionError(f"Expected {exception.__name__}")

    def run_api(self):
        """Run Python-cell API after CLI, including numerical and stateful checks."""
        if not any(row["stage"] == "CLI" for row in self.records):
            raise RuntimeError("Run the CLI phase before the API phase")
        import torch
        from simpro_vn import ValueNetworkPredictor

        for index, case in enumerate(self.cases, 1):
            name, rep = case["name"], case["rep"]
            print(f'\nAPI model {index}/{len(self.cases)}: {name}', flush=True)
            holder = {}

            def load_check():
                # Load via a config in a directory containing spaces; paths are
                # relative and the caller can keep weights outside the project.
                folder = self.output / "custom config paths"
                folder.mkdir(exist_ok=True)
                config_path = folder / f"{name}.json"
                prefix = self.project / "src/simpro_vn/assets/tokenizers" / stem_for(rep, case["setting"])
                config = {"checkpoint_path": os.path.relpath(case["checkpoint"], folder),
                          "representation": rep, "min_frequency": case["setting"],
                          "depth": case["depth"], "device": self.device,
                          "tokenizer_prefix": os.path.relpath(prefix, folder),
                          "scaler_path": os.path.relpath(self.project / "src/simpro_vn/assets/value_network_scaler.json", folder)}
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                predictor = ValueNetworkPredictor.from_config(config_path)
                holder["predictor"] = predictor
                assert not predictor.model.training
                assert predictor.depth == case["depth"]
                description = predictor.describe()
                assert description["verification"] == "sha256", "Expected a published checkpoint matching the packaged SHA256 catalog"
                assert description["representation"] == rep and description["depth"] == case["depth"]
                expected = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
                if self.device == "auto" and not torch.cuda.is_available():
                    expected = "cpu"
                assert predictor.device.type == torch.device(expected).type
                return {"device": str(predictor.device), "vocab_size": predictor.tokenizer.vocab_size,
                        "max_seq_len": predictor.max_seq_len, "verification": description["verification"]}
            if not self.record("API", name, "custom paths and relative config, model load, eval mode", load_check):
                holder.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            predictor = holder["predictor"]

            def batch_check():
                batch = predictor.predict(SAMPLES[rep], batch_size=2)
                singles = [predictor.predict(t)[0] for t in SAMPLES[rep]]
                self._rows(batch, 3)
                maximum = self._close(batch, singles)
                self._close(batch, predictor.predict(SAMPLES[rep], batch_size=1))
                assert predictor.predict([]) == []
                maximum = max(maximum, self._close(batch, predictor.predict(SAMPLES[rep])))
                tensor = predictor.predict_tensor(SAMPLES[rep], batch_size=2)
                assert tensor.device.type == predictor.device.type and not tensor.requires_grad
                if predictor.device.index is not None:
                    assert tensor.device.index == predictor.device.index
                tensor_rows = [dict(zip(PROPERTIES, row)) for row in tensor.cpu().tolist()]
                maximum = max(maximum, self._close(batch, tensor_rows))
                ids = torch.tensor([predictor.encode(SAMPLES[rep][0])])
                id_rows = [dict(zip(PROPERTIES, row)) for row in predictor.predict_ids(ids).cpu().tolist()]
                maximum = max(maximum, self._close(batch[:1], id_rows))
                return {"max_standardized_difference": maximum}
            self.record("API", name, "single, batch, padding, batch size and repeatability", batch_check)

            def cli_parity():
                cli = self.cli_results.get(name, {})
                assert {"batch", "prefix", "trajectory"} <= set(cli), "CLI results incomplete; inspect CLI failures first"
                differences = [self._close(predictor.predict(SAMPLES[rep], batch_size=2), cli["batch"]),
                               self._close(predictor.predict([PREFIXES[rep], ""], complete=False), cli["prefix"])]
                traj = predictor.predict_trajectory(SAMPLES[rep][0])
                assert traj["token_ids"] == cli["trajectory"]["token_ids"]
                differences.append(self._close(traj["predictions"], cli["trajectory"]["predictions"]))
                return {"max_standardized_difference": max(differences)}
            self.record("API", name, "CLI and Python API numerical agreement", cli_parity)

            def prefix_check():
                self._rows(predictor.predict([PREFIXES[rep], ""], complete=False), 2)
                assert predictor.encode("", complete=False) == [predictor.tokenizer.bos_id]
                complete_ids = predictor.encode(SAMPLES[rep][0])
                prefix_ids = predictor.encode(SAMPLES[rep][0], complete=False)
                assert complete_ids == prefix_ids + [predictor.tokenizer.eos_id]
                for text in SAMPLES[rep]:
                    ids = predictor.encode(text)
                    readable = predictor.tokenizer.readable_tokens(ids)
                    assert "".join(readable[1:-1]) == text, "Tokenization did not round trip"
                return {"bos_id": predictor.tokenizer.bos_id, "eos_id": predictor.tokenizer.eos_id}
            self.record("API", name, "prefix, empty prefix, EOS and tokenization preservation", prefix_check)

            def cache_check():
                maximum = 0.0
                for text, complete in [(t, True) for t in SAMPLES[rep]] + [(PREFIXES[rep], False), ("", False)]:
                    traj = predictor.predict_trajectory(text, complete=complete)
                    self._rows(traj["predictions"], len(traj["token_ids"]))
                    maximum = max(maximum, self._close(traj["predictions"][-1:], predictor.predict(text, complete=complete)))
                    session = predictor.new_session()
                    for chunk_size in (1, 3):
                        session.reset()
                        rows = []
                        ids = traj["token_ids"]
                        for start in range(0, len(ids), chunk_size):
                            rows.extend(session.step(ids[start:start + chunk_size]))
                        maximum = max(maximum, self._close(rows, traj["predictions"]))
                        if complete:
                            self._raises(ValueError, lambda: session.step(predictor.tokenizer.eos_id))
                    session.reset()
                    assert session.caches is None and session.length == 0 and not session.finished
                return {"max_standardized_difference": maximum}
            self.record("API", name, "trajectory, token/chunk KV-cache, EOS, reset", cache_check)

            def error_check():
                tok = predictor.tokenizer
                chemical_id = predictor.encode(SAMPLES[rep][0])[1]
                self._raises(ValueError, lambda: predictor.predict(SAMPLES[rep], batch_size=0))
                self._raises(TypeError, lambda: predictor.encode(None))
                self._raises(ValueError, lambda: predictor.encode("not a molecule"))
                self._raises(ValueError, lambda: predictor.tokenizer.readable_tokens([tok.vocab_size]))
                for ids in ([], [tok.pad_id], [tok.vocab_size], [chemical_id],
                            [tok.bos_id, tok.bos_id], [tok.bos_id, tok.eos_id, chemical_id]):
                    self._raises(ValueError, lambda ids=ids: predictor.new_session().step(ids))
                # Length rejection happens before running any oversized forward pass.
                self._raises(ValueError, lambda: predictor.new_session().step(
                    [tok.bos_id] + [chemical_id] * predictor.max_seq_len))
            self.record("API", name, "invalid inputs and session length are rejected", error_check)

            del predictor
            holder.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def summary(self, *, include_api=True):
        self.save()
        failed = [r for r in self.records if r["status"] == "FAIL"]
        print(f'\nPASS: {len(self.records) - len(failed)} | FAIL: {len(failed)}')
        print(f"Reports: {self.output}")
        expected_cli = 2 + 3 * len(self.cases)
        expected_api = 6 * len(self.cases) if include_api else 0
        cli_count = sum(r["stage"] == "CLI" for r in self.records)
        api_count = sum(r["stage"] == "API" for r in self.records)
        print(f"CLI checks: {cli_count}/{expected_cli}; API checks: {api_count}/{expected_api}")
        for row in failed:
            print(f'FAIL: {row["stage"]} | {row["model"]} | {row["check"]}')
        assert cli_count == expected_cli and api_count == expected_api, "Run incomplete; see load failures or unexecuted cells"
        assert not failed, "Failures recorded; do not loosen tolerances before investigating"
        print("All selected checks passed. This checks inference behavior, not scientific prediction accuracy or speed.")


def main():
    """Run the released-model matrix from an installed-package shell command."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Previously unused report directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("cli", "all"), default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = ColabValidation(args.project_dir, args.weights_dir, args.output_dir,
                             representations=("smiles", "smarts", "safe", "selfies"),
                             settings=(5000, 10000, 20000, 50000, "no_bpe"), depths=(2, 4, 6),
                             device=args.device)
    runner.preflight()
    runner.run_cli()
    if args.phase == "all":
        runner.run_api()
    runner.summary(include_api=args.phase == "all")


if __name__ == "__main__":
    main()
