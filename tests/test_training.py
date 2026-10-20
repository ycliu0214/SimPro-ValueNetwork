"""CPU tests for temporal supervision, masking, gradients, and configuration."""

import tempfile
import unittest
from pathlib import Path

import torch

from simpro_vn.model import RecurrentValueNetwork
from simpro_vn.training import TrainingConfig, temporal_loss, temporal_weights


class TemporalSupervisionTests(unittest.TestCase):
    def test_linear_is_per_molecule(self):
        weights = temporal_weights(torch.tensor([1, 2, 4]), 5, "linear")
        torch.testing.assert_close(weights, torch.tensor([
            [.1, 0, 0, 0, 0], [.1, 1, 0, 0, 0], [.1, .4, .7, 1, 0],
        ]))

    def test_discounted_endpoints_and_padding(self):
        weights = temporal_weights(torch.tensor([1, 3]), 4, "discounted", .5)
        torch.testing.assert_close(weights, torch.tensor([[1., 0, 0, 0], [.25, .5, 1, 0]]))

    def test_hand_calculated_loss_and_gradients(self):
        prediction = torch.tensor([[[1.], [2.], [100.]], [[2.], [3.], [4.]]], requires_grad=True)
        targets = torch.zeros(2, 1)
        loss = temporal_loss(prediction, targets, torch.tensor([2, 3]))
        expected = ((.1 * 1 + 1 * 4) / 1.1 + (.1 * 4 + .55 * 9 + 1 * 16) / 1.65) / 2
        self.assertAlmostEqual(loss.item(), expected, places=5)
        loss.backward()
        self.assertEqual(prediction.grad[0, 2, 0].item(), 0)
        self.assertGreater(prediction.grad[1, 2, 0].item(), 0)

    def test_equal_contribution_per_molecule_and_property(self):
        prediction = torch.ones(2, 7, 17)
        losses = temporal_loss(prediction, torch.zeros(2, 17), torch.tensor([2, 7]), reduction="none")
        torch.testing.assert_close(losses, torch.ones(2))

    def test_padding_does_not_change_loss(self):
        lengths = torch.tensor([2])
        targets = torch.zeros(1, 17)
        base = torch.randn(1, 2, 17)
        padded = torch.cat([base, torch.full((1, 3, 17), float("nan"))], dim=1)
        torch.testing.assert_close(temporal_loss(base, targets, lengths), temporal_loss(padded, targets, lengths))

    def test_invalid_schedule_and_lengths(self):
        for lengths, width, schedule, gamma in [([0], 2, "linear", .95), ([3], 2, "linear", .95),
                                                ([1], 2, "invalid", .95), ([1], 2, "discounted", 0)]:
            with self.subTest(lengths=lengths, schedule=schedule, gamma=gamma), self.assertRaises(ValueError):
                temporal_weights(torch.tensor(lengths), width, schedule, gamma)

    def test_causal_mask_prevents_future_leakage(self):
        torch.manual_seed(42)
        model = RecurrentValueNetwork(12, 8, 2, 16, 0., 17, 8).eval()
        a, b = torch.tensor([[1, 3, 4, 2]]), torch.tensor([[1, 3, 8, 2]])
        with torch.no_grad():
            for depth in (2, 4, 6):
                left, right = model(a, inference_depth=depth), model(b, inference_depth=depth)
                torch.testing.assert_close(left[:, :2], right[:, :2])

    def test_all_supported_depths_backpropagate(self):
        for depth in (2, 4, 6):
            with self.subTest(depth=depth):
                model = RecurrentValueNetwork(12, 8, 2, 16, 0., 17, 8)
                ids = torch.tensor([[1, 3, 2, 0], [1, 4, 5, 2]])
                predictions = model(ids, padding_mask=ids.eq(0), inference_depth=depth)
                temporal_loss(predictions, torch.ones(2, 17), torch.tensor([3, 4])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)

    def test_padding_preserves_valid_model_predictions(self):
        model = RecurrentValueNetwork(12, 8, 2, 16, 0., 17, 8).eval()
        single = torch.tensor([[1, 3, 2]])
        batch = torch.tensor([[1, 3, 2, 0, 0], [1, 4, 5, 6, 2]])
        with torch.no_grad():
            for depth in (2, 4, 6):
                a = model(single, inference_depth=depth)
                b = model(batch, padding_mask=batch.eq(0), inference_depth=depth)
                torch.testing.assert_close(a[0], b[0, :3])

    def test_all_twenty_bundles_accept_training_inputs(self):
        from simpro_vn.bundles import resolve_bundle
        from simpro_vn.representations import get_splitter
        from simpro_vn.tokenization import SurrogateChemicalTokenizer
        from simpro_vn.settings import TOKENIZATION_SETTINGS
        samples = {"smiles": "CCO", "smarts": "[#6]-[#6]-[#8]", "safe": "CCO", "selfies": "[C][C][O]"}
        for representation, text in samples.items():
            for setting in TOKENIZATION_SETTINGS:
                with self.subTest(representation=representation, setting=setting):
                    prefix, _, _ = resolve_bundle(representation, setting)
                    tokenizer = SurrogateChemicalTokenizer(prefix, get_splitter(representation))
                    ids = tokenizer.encode(text)
                    self.assertEqual((ids[0], ids[-1]), (tokenizer.bos_id, tokenizer.eos_id))
                    self.assertEqual("".join(tokenizer.readable_tokens(ids)[1:-1]), text)

    def test_representation_decode_errors_are_value_errors(self):
        from simpro_vn.representations.smiles_core import SMILESDecodeError, smiles_split
        from simpro_vn.representations.smarts_core import SMARTSDecodeError, smarts_split
        from simpro_vn.representations.safe_core import SAFEDecodeError, safe_split
        from simpro_vn.representations.selfies_core import SELFIESDecodeError, selfies_split
        for error_type in (SMILESDecodeError, SMARTSDecodeError, SAFEDecodeError, SELFIESDecodeError):
            with self.subTest(error_type=error_type):
                self.assertTrue(issubclass(error_type, ValueError))
        from simpro_vn.bundles import resolve_bundle
        from simpro_vn.tokenization import SurrogateChemicalTokenizer
        for representation, splitter in (("smiles", smiles_split), ("smarts", smarts_split),
                                          ("safe", safe_split), ("selfies", selfies_split)):
            with self.subTest(representation=representation):
                prefix, _, _ = resolve_bundle(representation, 5000)
                tokenizer = SurrogateChemicalTokenizer(prefix, splitter)
                with self.assertRaises(ValueError):
                    tokenizer.encode("not a molecule")

    def test_splitters_preserve_unexpected_errors(self):
        from unittest.mock import patch
        from simpro_vn.representations import smiles_core, smarts_core, safe_core, selfies_core
        for module, name in ((smiles_core, "smiles"), (smarts_core, "smarts")):
            splitter = getattr(module, "_" + name + "_splitter")
            with self.subTest(representation=name), patch.object(splitter, "tokenize", side_effect=RuntimeError("internal failure")):
                with self.assertRaisesRegex(RuntimeError, "internal failure"):
                    getattr(module, name + "_split")("CCO")
        for owner, attribute, splitter in ((safe_core.SAFESplitter, "tokenize", safe_core.safe_split),
                                           (selfies_core.sf, "split_selfies", selfies_core.selfies_split)):
            with self.subTest(splitter=splitter.__name__), patch.object(owner, attribute, side_effect=RuntimeError("internal failure")):
                with self.assertRaisesRegex(RuntimeError, "internal failure"):
                    splitter("CCO")


class ConfigurationTests(unittest.TestCase):
    def test_config_paths_are_relative_to_json(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"dataset_path": "data.csv", "output_dir": "output", "mode": "scratch"}))
            config = TrainingConfig.from_json(path)
            self.assertEqual(Path(config.dataset_path), Path(temp) / "data.csv")

    def test_invalid_configuration(self):
        for values in ({"epochs": 0}, {"dropout": 1}, {"gamma": 0}, {"d_model": 7},
                       {"precision": "unknown"}, {"learning_rate": 1e-8}, {"curriculum_depths": [2, 2]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig("data.csv", "output", mode="scratch", **values).validate()


if __name__ == "__main__":
    unittest.main()
