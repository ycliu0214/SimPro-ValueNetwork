"""Small CPU integration tests; synthetic labels do not measure model accuracy."""

from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from simpro_vn import ValueNetworkPredictor
from simpro_vn.bundles import ASSETS
from simpro_vn.scaling import PROPERTY_NAMES, PropertyScaler
from simpro_vn.training import TrainingConfig, run_training


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "dataset.csv"
        self.write_data()
        self.config = TrainingConfig(str(self.data), str(self.root / "scratch"), mode="scratch",
                                     epochs=2, batch_size=2, d_model=8, nhead=2, dim_feedforward=16,
                                     dropout=0., max_seq_len=32, device="cpu", depth=2)

    def write_data(self, change=None):
        stats = PropertyScaler(ASSETS / "value_network_scaler.json")
        rows = []
        strings = ["CCO", "CCN", "CCC", "CCCC", "CCCO"]
        selfies = ["[C][C][O]", "[C][C][N]", "[C][C][C]", "[C][C][C][C]", "[C][C][C][O]"]
        smarts = ["[#6]-[#6]-[#8]", "[#6]-[#6]-[#7]", "[#6]-[#6]-[#6]",
                  "[#6]-[#6]-[#6]-[#6]", "[#6]-[#6]-[#6]-[#8]"]
        for i, text in enumerate(strings):
            row = {"NEW_Isomeric_SMILES": text, "SELFIES": selfies[i], "SAFE": text, "SMARTS": smarts[i],
                   "Parent_Group": f"parent_{i}", "Split_Set": "Train" if i < 3 else "Val"}
            row.update({name: float(mean) + float(std) * (i + 1) / 10
                        for name, mean, std in zip(PROPERTY_NAMES, stats.mean, stats.std)})
            rows.append(row)
        if change:
            change(rows)
        with self.data.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_scratch_finetune_reload_and_source_preservation(self):
        trained = run_training(self.config)
        source = trained["best_checkpoint"]
        before = digest(source)
        tuned = run_training(replace(self.config, mode="finetune", output_dir=str(self.root / "tuned"),
                                      checkpoint_path=source, epochs=1, schedule="discounted"))
        self.assertEqual(before, digest(source))
        self.assertNotEqual(digest(source), digest(tuned["best_checkpoint"]))
        initial = torch.load(source, weights_only=True, map_location="cpu")["model_state_dict"]
        updated = torch.load(tuned["last_checkpoint"], weights_only=True, map_location="cpu")["model_state_dict"]
        self.assertFalse(torch.equal(initial["embedding.weight"], updated["embedding.weight"]))
        for name in ("best_checkpoint", "last_checkpoint"):
            predictor = ValueNetworkPredictor(checkpoint_path=tuned[name], depth=2, device="cpu")
            self.assertEqual(len(predictor.predict("CCO")[0]), 17)
        api = ValueNetworkPredictor.from_config(Path(tuned["directory"]) / "inference_config.json")
        self.assertEqual(api.depth, 2)
        self.assertTrue((Path(tuned["directory"]) / "history.json").is_file())

    def test_refuses_existing_output_and_pretrained_weight_directory(self):
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "best.pt"
        marker.write_bytes(b"preserve this file")
        with self.assertRaises(FileExistsError):
            run_training(replace(self.config, output_dir=str(existing)))
        self.assertEqual(marker.read_bytes(), b"preserve this file")
        from simpro_vn import training
        protected = Path(training.__file__).resolve().parents[2] / "weights" / "training_safety_probe"
        with self.assertRaises(ValueError):
            run_training(replace(self.config, output_dir=str(protected)))
        self.assertFalse(protected.exists())

    def test_resume_matches_uninterrupted_cpu_run(self):
        full = run_training(self.config)
        part = run_training(replace(self.config, output_dir=str(self.root / "partial")), max_epochs=1)
        resumed = run_training(replace(self.config, output_dir=str(self.root / "resumed"), resume_from=part["last_checkpoint"]))
        a = torch.load(full["last_checkpoint"], weights_only=True, map_location="cpu")
        b = torch.load(resumed["last_checkpoint"], weights_only=True, map_location="cpu")
        self.assertEqual(a["history"], b["history"])
        for key in a["model_state_dict"]:
            torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)

    def test_resume_rejects_changed_data(self):
        part = run_training(self.config, max_epochs=1)
        self.write_data(lambda rows: rows[0].update(MW=1.))
        with self.assertRaisesRegex(ValueError, "unchanged data"):
            run_training(replace(self.config, output_dir=str(self.root / "changed"), resume_from=part["last_checkpoint"]))

    def test_last_checkpoint_contains_best_snapshot_for_resume(self):
        part = run_training(self.config, max_epochs=1)
        best = torch.load(part["best_checkpoint"], weights_only=True, map_location="cpu")
        last = torch.load(part["last_checkpoint"], weights_only=True, map_location="cpu")
        self.assertEqual(best["val_loss"], last["best_checkpoint"]["val_loss"])
        Path(part["best_checkpoint"]).unlink()
        resumed = run_training(replace(self.config, output_dir=str(self.root / "resume_from_last_only"),
                                        resume_from=part["last_checkpoint"]))
        self.assertTrue(Path(resumed["best_checkpoint"]).is_file())

    def test_three_stage_branches_and_fixed_depths(self):
        result = run_training(replace(self.config, mode="curriculum", output_dir=str(self.root / "curriculum"),
                                       warmup_epochs=1, linear_epochs=1, discounted_epochs=1, stage3_batch_size=2))
        self.assertEqual(set(result["branches"]), {"2", "4", "6"})
        for depth, branch in result["branches"].items():
            for stage in ("linear", "discounted"):
                checkpoint = torch.load(branch[stage]["best_checkpoint"], weights_only=True, map_location="cpu")
                self.assertEqual(checkpoint["simpro_vn"]["depth"], int(depth))
                config = json.loads((Path(branch[stage]["directory"]) / "phase_config.json").read_text())
                self.assertEqual(config["schedule"], stage)
                state = torch.load(branch[stage]["last_checkpoint"], weights_only=True, map_location="cpu")
                steps = {int(v["step"].item()) for v in state["optimizer_state_dict"]["state"].values()}
                self.assertEqual(steps, {2})
            api = ValueNetworkPredictor(checkpoint_path=branch["discounted"]["best_checkpoint"], depth=int(depth))
            self.assertEqual(len(api.predict("CCO")), 1)

    def test_invalid_labels_fail_before_any_checkpoint_is_saved(self):
        self.write_data(lambda rows: rows[0].update(MW="nan"))
        with self.assertRaisesRegex(ValueError, "CSV row"):
            run_training(self.config)
        self.assertFalse(list(Path(self.config.output_dir).rglob("*.pt")))

    def test_best_checkpoint_is_selected_by_validation_loss(self):
        result = run_training(self.config)
        state = torch.load(result["best_checkpoint"], weights_only=True, map_location="cpu")
        self.assertEqual(state["val_loss"], min(row["val_loss"] for row in result["history"]))
        self.assertEqual(state["val_loss"], result["best_val_loss"])

    def test_resume_rejects_changed_training_settings(self):
        part = run_training(self.config, max_epochs=1)
        with self.assertRaisesRegex(ValueError, "unchanged data"):
            run_training(replace(self.config, output_dir=str(self.root / "changed_settings"),
                                  resume_from=part["last_checkpoint"], batch_size=1))

    def test_unsupported_tokens_and_excess_length_are_rejected(self):
        for text, directory in (("not a molecule", "bad_token"), ("C" * 100, "too_long")):
            self.write_data(lambda rows: rows[0].update(NEW_Isomeric_SMILES=text))
            with self.subTest(text=text), self.assertRaises(ValueError):
                run_training(replace(self.config, output_dir=str(self.root / directory),
                                      max_seq_len=2 if directory == "too_long" else 32))

    def test_skip_invalid_is_explicit_and_logged(self):
        self.write_data(lambda rows: rows[0].update(MW="nan"))
        result = run_training(replace(self.config, skip_invalid=True, epochs=1))
        summary = json.loads((Path(self.config.output_dir) / "data" / "summary.json").read_text())
        self.assertEqual(summary["rejected"], 1)
        self.assertTrue(Path(result["best_checkpoint"]).is_file())

    def test_invalid_molecular_string_is_logged_and_skipped_when_enabled(self):
        self.write_data(lambda rows: rows[0].update(NEW_Isomeric_SMILES="not a molecule"))
        result = run_training(replace(self.config, skip_invalid=True, epochs=1))
        data_dir = Path(self.config.output_dir) / "data"
        summary = json.loads((data_dir / "summary.json").read_text())
        self.assertEqual((summary["Train"], summary["Val"], summary["rejected"]), (2, 2, 1))
        rejected = [json.loads(line) for line in (data_dir / "rejections.jsonl").read_text().splitlines()]
        self.assertEqual(rejected[0]["csv_row"], 2)
        self.assertIn("Failed to split SMILES", rejected[0]["reason"])
        self.assertTrue(Path(result["best_checkpoint"]).is_file())

    def test_invalid_strings_stop_or_skip_for_all_representations(self):
        columns = {"smiles": "NEW_Isomeric_SMILES", "smarts": "SMARTS", "safe": "SAFE", "selfies": "SELFIES"}
        for representation, column in columns.items():
            self.write_data(lambda rows: rows[0].update({column: "not a molecule"}))
            with self.subTest(representation=representation, handling="stop"), self.assertRaisesRegex(ValueError, "CSV row 2"):
                run_training(replace(self.config, representation=representation,
                                      output_dir=str(self.root / (representation + "_stop"))))
            with self.subTest(representation=representation, handling="skip"):
                output = self.root / (representation + "_skip")
                run_training(replace(self.config, representation=representation, output_dir=str(output),
                                      skip_invalid=True, epochs=1))
                summary = json.loads((output / "data" / "summary.json").read_text())
                self.assertEqual((summary["Train"], summary["Val"], summary["rejected"]), (2, 2, 1))

    def test_skip_invalid_does_not_swallow_unexpected_runtime_errors(self):
        from unittest.mock import patch
        with patch("simpro_vn.representations.smiles_core._smiles_splitter.tokenize", side_effect=RuntimeError("internal failure")):
            with self.assertRaisesRegex(RuntimeError, "internal failure"):
                run_training(replace(self.config, skip_invalid=True, epochs=1))

    def test_parent_overlap_is_never_silently_skipped(self):
        self.write_data(lambda rows: rows[3].update(Parent_Group=rows[0]["Parent_Group"]))
        with self.assertRaisesRegex(ValueError, "Train/Val overlap"):
            run_training(replace(self.config, skip_invalid=True))

    def test_missing_validation_is_rejected(self):
        self.write_data(lambda rows: [row.update(Split_Set="Train") for row in rows])
        with self.assertRaisesRegex(ValueError, "Both Train and Val"):
            run_training(self.config)

    def test_test_split_is_not_used_for_optimization(self):
        self.write_data(lambda rows: rows[-1].update(Split_Set="Test", MW="not a label"))
        run_training(replace(self.config, epochs=1))
        summary = json.loads((Path(self.config.output_dir) / "data" / "summary.json").read_text())
        self.assertEqual((summary["Train"], summary["Val"], summary["Test"]), (3, 1, 1))

    def test_checkpoint_metadata_rejects_incompatible_predictions(self):
        result = run_training(replace(self.config, epochs=1))
        for options in ({"depth": 6}, {"depth": 2, "min_frequency": 10000}, {"depth": 2, "nhead": 4}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ValueNetworkPredictor(checkpoint_path=result["best_checkpoint"], **options)

    def test_gradient_accumulation_includes_partial_final_group(self):
        result = run_training(replace(self.config, epochs=1, batch_size=1, accumulation_steps=2))
        state = torch.load(result["last_checkpoint"], weights_only=True, map_location="cpu")
        steps = {int(value["step"].item()) for value in state["optimizer_state_dict"]["state"].values()}
        self.assertEqual(steps, {2})

    def test_all_representations_and_no_bpe(self):
        for representation, setting in (("smiles", "no_bpe"), ("smarts", 5000), ("safe", 5000), ("selfies", 5000)):
            with self.subTest(representation=representation):
                result = run_training(replace(self.config, representation=representation, min_frequency=setting,
                                               output_dir=str(self.root / representation), epochs=1))
                self.assertTrue(Path(result["best_checkpoint"]).is_file())

    def test_cli_training_then_cli_prediction(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(asdict(replace(self.config, epochs=1))), encoding="utf-8")
        trained = subprocess.run([sys.executable, "-m", "simpro_vn.training", "--config", str(path)],
                                 text=True, capture_output=True, timeout=180)
        self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
        predicted = subprocess.run([sys.executable, "-m", "simpro_vn", "--checkpoint",
                                    str(Path(self.config.output_dir) / "model" / "best.pt"),
                                    "--depth", "2", "--text", "CCO", "--device", "cpu"],
                                   text=True, capture_output=True, timeout=90)
        self.assertEqual(predicted.returncode, 0, predicted.stdout + predicted.stderr)
        self.assertEqual(len(json.loads(predicted.stdout)[0]), 17)
        invalid = subprocess.run([sys.executable, "-m", "simpro_vn", "--checkpoint",
                                  str(Path(self.config.output_dir) / "model" / "best.pt"),
                                  "--depth", "2", "--text", "not a molecule", "--device", "cpu"],
                                 text=True, capture_output=True, timeout=90)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("Failed to split SMILES", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
