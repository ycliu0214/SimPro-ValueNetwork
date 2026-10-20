"""Small CPU checks for model descriptions, ID tensors, and fixed tensor scoring."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn.utils.rnn import pad_sequence

from simpro_vn import ValueNetworkPredictor, identify_checkpoint
from simpro_vn.bundles import ASSETS, resolve_bundle
from simpro_vn.model import RecurrentValueNetwork
from simpro_vn.representations import get_splitter
from simpro_vn.scaling import PROPERTY_NAMES
from simpro_vn.tokenization import SurrogateChemicalTokenizer


class InferenceInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "small.pt"
        prefix, _, fingerprint = resolve_bundle("smiles", 5000)
        tokenizer = SurrogateChemicalTokenizer(prefix, get_splitter("smiles"))
        torch.manual_seed(42)
        model = RecurrentValueNetwork(tokenizer.vocab_size, 8, 2, 16, 0., 17, 16)
        self.state = model.state_dict()
        self.metadata = {"representation": "smiles", "min_frequency": 5000, "depth": 2,
                         "nhead": 2, "property_names": list(PROPERTY_NAMES), "bundle_fingerprint": fingerprint,
                         "d_model": 8, "dim_feedforward": 16, "num_targets": 17, "max_seq_len": 16,
                         "vocab_size": tokenizer.vocab_size,
                         "special_token_ids": {"[PAD]": tokenizer.pad_id, "[BOS]": tokenizer.bos_id, "[EOS]": tokenizer.eos_id}}
        torch.save({"model_state_dict": self.state, "simpro_vn": self.metadata}, self.path)
        self.predictor = ValueNetworkPredictor(checkpoint_path=self.path, depth=2, device="cpu")

    def test_describe_is_complete_and_independent(self):
        info = self.predictor.describe()
        self.assertEqual((info["d_model"], info["nhead"], info["depth"], info["max_seq_len"]), (8, 2, 2, 16))
        self.assertEqual(info["verification"], "metadata")
        self.assertEqual(info["property_names"], list(PROPERTY_NAMES))
        info["property_names"].clear()
        self.assertEqual(len(self.predictor.describe()["property_names"]), 17)

    def test_tensor_prediction_matches_dictionary_api(self):
        texts = ["CCO", "CCCC", "C"]
        tensor = self.predictor.predict_tensor(texts, batch_size=2)
        records = self.predictor.predict(texts, batch_size=2)
        expected = torch.tensor([[r[name] for name in PROPERTY_NAMES] for r in records])
        torch.testing.assert_close(tensor, expected)
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, self.predictor.device)

    def test_standardized_and_empty_predictions(self):
        z = self.predictor.predict_tensor("CCO", standardized=True)
        units = self.predictor.predict_tensor("CCO")
        torch.testing.assert_close(self.predictor.scaler.inverse_transform(z), units)
        self.assertEqual(self.predictor.predict_tensor([]).shape, (0, 17))
        with self.assertRaises(ValueError):
            self.predictor.predict_tensor("CCO", batch_size=0)

    def test_ids_mask_and_trajectory(self):
        texts = ["CCO", "C"]
        sequences = [torch.tensor(self.predictor.encode(t)) for t in texts]
        ids = pad_sequence(sequences, batch_first=True, padding_value=self.predictor.tokenizer.pad_id)
        mask = ids.ne(self.predictor.tokenizer.pad_id)
        values = self.predictor.predict_ids(ids, attention_mask=mask.long())
        torch.testing.assert_close(values, self.predictor.predict_tensor(texts))
        all_values = self.predictor.predict_ids(ids, output_all_tokens=True)
        self.assertEqual(all_values.shape, (2, ids.size(1), 17))
        self.assertTrue(torch.all(all_values[~mask] == 0))
        for i, text in enumerate(texts):
            trajectory = self.predictor.predict_trajectory(text)
            expected = torch.tensor([[r[name] for name in PROPERTY_NAMES] for r in trajectory["predictions"]])
            torch.testing.assert_close(all_values[i, :len(sequences[i])], expected)

    def test_prefix_and_bos_only_ids(self):
        for text in ("CC", ""):
            ids = torch.tensor([self.predictor.encode(text, complete=False)])
            torch.testing.assert_close(self.predictor.predict_ids(ids),
                                       self.predictor.predict_tensor(text, complete=False))

    def test_invalid_id_batches_are_rejected(self):
        tok = self.predictor.tokenizer
        bad = [torch.tensor([[3, tok.eos_id]]), torch.tensor([[tok.bos_id, tok.bos_id]]),
               torch.tensor([[tok.bos_id, tok.eos_id, 3]]), torch.tensor([[tok.bos_id, tok.pad_id, 3]]),
               torch.tensor([[tok.pad_id]]), torch.tensor([[tok.bos_id, tok.vocab_size]]),
               torch.empty((0, 2), dtype=torch.long), torch.empty((1, 0), dtype=torch.long),
               torch.full((1, 17), tok.bos_id, dtype=torch.long)]
        for ids in bad:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.predictor.predict_ids(ids)
        with self.assertRaises(TypeError):
            self.predictor.predict_ids(torch.ones(1, 2))
        valid = torch.tensor([self.predictor.encode("CCO")])
        for mask in (torch.ones(1, 1), torch.full_like(valid, 2), torch.zeros_like(valid)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.predictor.predict_ids(valid, attention_mask=mask)

    def test_incremental_tensor_matches_full_ids(self):
        ids = torch.tensor(self.predictor.encode("CCO"))
        session = self.predictor.new_session()
        values = torch.cat([session.step_tensor(ids[:2]), session.step_tensor(ids[2:])])
        expected = self.predictor.predict_ids(ids[None], output_all_tokens=True)[0]
        torch.testing.assert_close(values, expected)
        with self.assertRaises(ValueError):
            session.step_tensor(self.predictor.tokenizer.eos_id)
        session.reset()
        standardized = session.step_tensor(ids, standardized=True)
        torch.testing.assert_close(self.predictor.scaler.inverse_transform(standardized), expected)

    def test_tensor_scores_can_weight_another_models_loss(self):
        scores = self.predictor.predict_tensor("CCO", standardized=True)
        self.assertFalse(scores.requires_grad)
        self.assertFalse(torch.is_inference(scores))
        parameter = torch.ones_like(scores, requires_grad=True)
        (parameter * scores).sum().backward()
        torch.testing.assert_close(parameter.grad, scores)
        self.assertTrue(all(p.grad is None for p in self.predictor.model.parameters()))

    def test_conflicting_shape_or_special_token_metadata_is_rejected(self):
        for change in ({"d_model": 16}, {"special_token_ids": {"[PAD]": 999}}):
            torch.save({"model_state_dict": self.state, "simpro_vn": {**self.metadata, **change}}, self.path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ValueNetworkPredictor(checkpoint_path=self.path, depth=2)

    def test_training_initializer_checks_description_dimensions_and_special_ids(self):
        from simpro_vn.training import TrainingConfig, _new_model
        config = TrainingConfig("unused.csv", "unused_output", depth=2, nhead=2)
        for change in ({"d_model": 16}, {"special_token_ids": {"[PAD]": 999}}):
            torch.save({"model_state_dict": self.state, "simpro_vn": {**self.metadata, **change}}, self.path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                _new_model(config, self.predictor.tokenizer, self.predictor.scaler,
                           self.predictor.bundle_fingerprint, self.path)

    def test_digest_identity_survives_renaming_and_rejects_wrong_depth(self):
        torch.save(self.state, self.path)
        sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        record = {**self.metadata, "model_id": "test_model", "checkpoint": {"sha256": sha}}
        renamed = self.path.parent / "renamed.pt"
        shutil.copyfile(self.path, renamed)
        with patch("simpro_vn.checkpoints._catalog", return_value={sha: record}):
            self.assertEqual(identify_checkpoint(renamed)["model_id"], "test_model")
            predictor = ValueNetworkPredictor(checkpoint_path=renamed, depth=2)
            self.assertEqual(predictor.describe()["verification"], "sha256")
            with self.assertRaises(ValueError):
                ValueNetworkPredictor(checkpoint_path=renamed, depth=4)

    def test_published_descriptions_cover_all_models(self):
        payload = json.loads((ASSETS / "pretrained_models.json").read_text())
        self.assertEqual(len(payload["models"]), 60)
        self.assertEqual(len({r["checkpoint"]["sha256"] for r in payload["models"]}), 60)
        for record in payload["models"]:
            self.assertEqual(record["property_names"], list(PROPERTY_NAMES))
            self.assertEqual((record["d_model"], record["nhead"], record["max_seq_len"]), (768, 12, 128))


if __name__ == "__main__":
    unittest.main()
