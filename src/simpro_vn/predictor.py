"""Public prediction API and per-sequence KV-cache sessions."""

import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .model import RecurrentValueNetwork
from .representations import get_splitter
from .scaling import PropertyScaler
from .tokenization import SurrogateChemicalTokenizer
from .settings import normalize_setting
from .bundles import resolve_bundle
from .checkpoints import checkpoint_metadata

class ValueNetworkPredictor:
    def __init__(self, *, checkpoint_path: str | Path, representation="smiles",
                 min_frequency=5000, depth=6, tokenizer_prefix=None, scaler_path=None,
                 device="auto", nhead=None):
        if depth not in (2, 4, 6):
            raise ValueError("Supported pretrained depths are 2, 4, and 6")
        self.representation = representation
        self.depth = depth
        self.tokenization_setting = normalize_setting(min_frequency)
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        tokenizer_prefix, scaler_path, self.bundle_fingerprint = resolve_bundle(
            representation, self.tokenization_setting, tokenizer_prefix, scaler_path,
        )
        self.tokenizer = SurrogateChemicalTokenizer(tokenizer_prefix, get_splitter(representation))
        self.scaler = PropertyScaler(scaler_path)
        # Accept training checkpoints and model-only state dictionaries.
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=True)
        metadata, verification = checkpoint_metadata(checkpoint_path, checkpoint)
        for key, expected in (("schema_version", 1), ("architecture", "recurrent_depth_transformer"),
                              ("representation", representation), ("min_frequency", self.tokenization_setting),
                              ("depth", depth), ("bundle_fingerprint", self.bundle_fingerprint)):
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"Checkpoint {key} does not match the selected model configuration")
        if "property_names" in metadata and metadata["property_names"] != list(self.scaler.property_names):
            raise ValueError("Checkpoint property order differs from the scaler")
        if nhead is not None and "nhead" in metadata and nhead != metadata["nhead"]:
            raise ValueError("Checkpoint nhead does not match the requested attention head count")
        nhead = metadata.get("nhead", 12) if nhead is None else nhead
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        vocab_size, d_model = state["embedding.weight"].shape
        self.max_seq_len = state["pos_encoder.pe"].shape[1]
        num_targets = state["value_head.3.weight"].shape[0]
        if vocab_size != self.tokenizer.vocab_size:
            raise ValueError("Checkpoint vocabulary size differs from tokenizer vocabulary size")
        if num_targets != len(self.scaler.property_names):
            raise ValueError("Checkpoint output size differs from scaler property count")
        dimensions = {"vocab_size": vocab_size, "d_model": d_model, "num_targets": num_targets,
                      "max_seq_len": self.max_seq_len,
                      "dim_feedforward": state["shared_transformer_block.linear1.weight"].shape[0]}
        for key, actual in dimensions.items():
            if key in metadata and metadata[key] != actual:
                raise ValueError(f"Checkpoint tensor shapes differ from its model description: {key}")
        special_ids = {"[PAD]": self.tokenizer.pad_id, "[BOS]": self.tokenizer.bos_id, "[EOS]": self.tokenizer.eos_id}
        if "special_token_ids" in metadata and metadata["special_token_ids"] != special_ids:
            raise ValueError("Checkpoint special token IDs differ from the selected tokenizer")
        self._description = {"schema_version": 1, "architecture": "recurrent_depth_transformer",
                             **metadata, **dimensions, "representation": representation,
                             "min_frequency": self.tokenization_setting, "depth": depth, "nhead": nhead,
                             "property_names": list(self.scaler.property_names), "special_token_ids": special_ids,
                             "bundle_fingerprint": self.bundle_fingerprint, "verification": verification}
        self.model = RecurrentValueNetwork(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead,
            dim_feedforward=state["shared_transformer_block.linear1.weight"].shape[0],
            num_targets=num_targets, max_seq_len=self.max_seq_len,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def describe(self):
        """Return model dimensions, preprocessing identity, and verification level."""
        return deepcopy(self._description)

    @classmethod
    def from_config(cls, path: str | Path):
        """Resolve all file paths relative to the JSON configuration directory."""
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        for key in ("checkpoint_path", "tokenizer_prefix", "scaler_path"):
            if config.get(key) is not None:
                value = Path(config[key]).expanduser()
                config[key] = value if value.is_absolute() else path.parent / value
        return cls(**config)

    def encode(self, text, *, complete=True):
        ids = self.tokenizer.encode(text, complete=complete)
        if len(ids) > self.max_seq_len:
            raise ValueError(f"Input has {len(ids)} tokens; model maximum is {self.max_seq_len}, including BOS/EOS")
        return ids

    def _records(self, values):
        return [dict(zip(self.scaler.property_names, row)) for row in values.float().cpu().tolist()]

    @torch.no_grad()
    def predict_ids(self, input_ids, *, attention_mask=None, standardized=False, output_all_tokens=False):
        """Score right-padded token IDs on this predictor's device, without gradients.

        Input: integer tensor [B,T], with BOS and optional final EOS in each row.
        attention_mask uses 1/True for valid tokens. Returns float32 [B,17], or
        [B,T,17] with zeroed padding when output_all_tokens=True. IDs must come
        from this model's tokenizer/mapping; token ranges alone cannot prove that.
        """
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be an int32/int64 tensor")
        if input_ids.ndim != 2 or input_ids.size(0) == 0 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have nonempty shape [batch, sequence]")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError("Token batch exceeds the model maximum sequence length")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        tok = self.tokenizer
        if torch.any((ids < 0) | (ids >= tok.vocab_size)):
            raise ValueError("input_ids contain an out-of-range token ID")
        if attention_mask is None:
            valid = ids.ne(tok.pad_id)
        else:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != ids.shape:
                raise ValueError("attention_mask must be a tensor with the same shape as input_ids")
            mask = attention_mask.to(self.device)
            if torch.any((mask != 0) & (mask != 1)):
                raise ValueError("attention_mask must contain only 0/1 or booleans")
            valid = mask.bool()
            if not torch.equal(valid, ids.ne(tok.pad_id)):
                raise ValueError("attention_mask must mark exactly the non-PAD positions")
        lengths = valid.sum(1)
        positions = torch.arange(ids.size(1), device=self.device)[None]
        if torch.any(lengths == 0) or not torch.equal(valid, positions < lengths[:, None]):
            raise ValueError("Each row must be nonempty and use contiguous right padding")
        if torch.any(ids[:, 0] != tok.bos_id) or torch.any((ids[:, 1:] == tok.bos_id)):
            raise ValueError("Each row must start with exactly one BOS")
        if torch.any((ids == tok.eos_id) & (positions != lengths[:, None] - 1)):
            raise ValueError("EOS may appear only at the final valid position")
        values = self.model(ids, padding_mask=~valid, inference_depth=self.depth).float()
        if not output_all_tokens:
            values = values[torch.arange(ids.size(0), device=self.device), lengths - 1]
        if not standardized:
            values = self.scaler.inverse_transform(values)
        return values.masked_fill(~valid[:, :, None], 0.0) if output_all_tokens else values

    @torch.no_grad()
    def predict_tensor(self, texts, *, complete=True, batch_size=32, standardized=False):
        """Return float32 [N,17] on self.device in input order, without CPU conversion."""
        if isinstance(texts, str):
            texts = [texts]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        results = []
        for start in range(0, len(texts), batch_size):
            sequences = [torch.tensor(self.encode(t, complete=complete), dtype=torch.long)
                         for t in texts[start:start + batch_size]]
            ids = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_id)
            results.append(self.predict_ids(ids, standardized=standardized))
        return torch.cat(results) if results else torch.empty((0, len(self.scaler.property_names)), device=self.device,
                                                              dtype=torch.float32)

    @torch.inference_mode()
    def predict(self, texts: str | list[str], *, complete=True, batch_size=32):
        """Return one property dictionary per input, in original property units.

        complete=False excludes EOS and accepts the empty prefix (BOS only).
        Inputs must already use this predictor's molecular representation.
        """
        if isinstance(texts, str):
            texts = [texts]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        results = []
        for start in range(0, len(texts), batch_size):
            sequences = [torch.tensor(self.encode(t, complete=complete), dtype=torch.long)
                         for t in texts[start:start + batch_size]]
            lengths = torch.tensor([len(s) for s in sequences], device=self.device)
            src = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_id).to(self.device)
            predictions = self.model(src, padding_mask=src.eq(self.tokenizer.pad_id), inference_depth=self.depth)
            final = predictions[torch.arange(len(sequences), device=self.device), lengths - 1]
            results.extend(self._records(self.scaler.inverse_transform(final)))
        return results

    @torch.inference_mode()
    def predict_trajectory(self, text: str, *, complete=True):
        """Predict after every encoded token, including BOS and optional EOS."""
        ids = self.encode(text, complete=complete)
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        values = self.scaler.inverse_transform(self.model(src, inference_depth=self.depth)[0])
        return {"token_ids": ids, "tokens": self.tokenizer.readable_tokens(ids),
                "predictions": self._records(values)}

    def new_session(self):
        """Create an empty cache for one already-tokenized molecular sequence."""
        return IncrementalSession(self)


class IncrementalSession:
    def __init__(self, predictor: ValueNetworkPredictor):
        self.predictor = predictor
        self.reset()

    def reset(self):
        self.caches = None
        self.length = 0
        self.finished = False

    @torch.inference_mode()
    def step(self, token_ids: int | list[int]):
        """Append token IDs, starting with BOS. Return predictions for new tokens.

        Use IDs from one fixed tokenizer stream; separately encoding text chunks
        can change BPE merges and is not equivalent to incremental token input.
        """
        return self.predictor._records(self.step_tensor(token_ids))

    @torch.no_grad()
    def step_tensor(self, token_ids, *, standardized=False):
        """Append IDs and return float32 [new_tokens,17] on the predictor device.

        Accept an integer, a list of integers, or an int32/int64 scalar/1D tensor.
        Cache semantics and EOS/reset rules are identical to step().
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dtype not in (torch.int32, torch.int64) or token_ids.ndim > 1:
                raise TypeError("Token IDs must be an int32/int64 scalar or 1D tensor")
            token_ids = token_ids.detach().cpu().tolist()
        ids = [token_ids] if isinstance(token_ids, int) else list(token_ids)
        tok = self.predictor.tokenizer
        if self.finished:
            raise ValueError("Session already consumed EOS; call reset() for a new sequence")
        if not ids or any(type(i) is not int or not 0 <= i < tok.vocab_size for i in ids):
            raise ValueError("Provide a nonempty sequence of valid integer token IDs")
        if tok.pad_id in ids:
            raise ValueError("A single-sequence cache session must not include PAD")
        if self.length == 0 and ids[0] != tok.bos_id:
            raise ValueError("The first token must be BOS")
        if tok.bos_id in (ids[1:] if self.length == 0 else ids):
            raise ValueError("BOS may appear only at the start of the sequence")
        if tok.eos_id in ids[:-1]:
            raise ValueError("EOS may appear only as the final token")
        if self.length + len(ids) > self.predictor.max_seq_len:
            raise ValueError("Session exceeds model maximum sequence length")
        src = torch.tensor([ids], dtype=torch.long, device=self.predictor.device)
        predictions, caches = self.predictor.model(
            src, inference_depth=self.predictor.depth, use_cache=True, past_caches=self.caches,
        )
        values = predictions[0].float() if standardized else self.predictor.scaler.inverse_transform(predictions[0])
        self.caches = caches
        self.length += len(ids)
        self.finished = ids[-1] == tok.eos_id
        return values
