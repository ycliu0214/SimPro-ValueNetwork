"""Optional GPU smoke tests for full and mixed precision training."""

import csv
from pathlib import Path
import tempfile
import unittest

import torch

from simpro_vn import ValueNetworkPredictor
from simpro_vn.bundles import ASSETS
from simpro_vn.scaling import PROPERTY_NAMES, PropertyScaler
from simpro_vn.training import TrainingConfig, run_training


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaTrainingTests(unittest.TestCase):
    def check_precision(self, precision):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "data.csv"
            scaler = PropertyScaler(ASSETS / "value_network_scaler.json")
            with dataset.open("w", newline="") as handle:
                fields = ["NEW_Isomeric_SMILES", "Split_Set", *PROPERTY_NAMES]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, text in enumerate(("CCO", "CCN", "CCCC")):
                    writer.writerow({"NEW_Isomeric_SMILES": text, "Split_Set": "Val" if index == 2 else "Train",
                                     **{name: float(mean) for name, mean in zip(PROPERTY_NAMES, scaler.mean)}})
            result = run_training(TrainingConfig(str(dataset), str(root / "run"), mode="scratch", depth=2,
                                                 epochs=3, batch_size=1, accumulation_steps=2, d_model=32,
                                                 nhead=4, dim_feedforward=64, dropout=0., max_seq_len=32,
                                                 precision=precision, device="cuda"))
            state = torch.load(result["last_checkpoint"], weights_only=True, map_location="cpu")
            self.assertTrue(state["optimizer_state_dict"]["state"], "No optimizer update completed")
            self.assertTrue(all(row["skipped_fp16_updates"] >= 0 for row in result["history"]))
            api = ValueNetworkPredictor(checkpoint_path=result["best_checkpoint"], depth=2, device="cuda")
            self.assertEqual(len(api.predict("CCO")[0]), 17)
            values = api.predict_tensor("CCO")
            self.assertEqual(values.device.type, "cuda")
            self.assertFalse(values.requires_grad)
            torch.testing.assert_close(api.predict_ids(torch.tensor([api.encode("CCO")])), values)

    def test_fp32(self):
        self.check_precision("fp32")

    def test_fp16(self):
        self.check_precision("fp16")

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "bf16 is unavailable")
    def test_bf16(self):
        self.check_precision("bf16")


if __name__ == "__main__":
    unittest.main()
