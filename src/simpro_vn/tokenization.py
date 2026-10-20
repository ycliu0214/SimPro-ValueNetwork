"""Encode molecular strings using the selected mapping and BPE vocabulary."""

import json
from pathlib import Path

from tokenizers import Tokenizer


class SurrogateChemicalTokenizer:
    def __init__(self, prefix: str | Path, splitter):
        prefix = str(prefix)
        self.tokenizer = Tokenizer.from_file(prefix + "_tokenizer.json")
        # Padding belongs to the predictor; never silently truncate an input.
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        with open(prefix + "_mapping.json", encoding="utf-8") as handle:
            mapping = json.load(handle)
        self.encode_map = mapping["encode"]
        self.decode_map = mapping["decode"]
        config = mapping.get("config", {})
        self.special_tokens = config.get("special_tokens", ["[PAD]", "[BOS]", "[EOS]"])
        self.isolate_tokens = config.get("isolate_tokens", [])
        self.splitter = splitter
        self.pad_id = self._required_id("[PAD]")
        self.bos_id = self._required_id("[BOS]")
        self.eos_id = self._required_id("[EOS]")

    def _required_id(self, token):
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"Tokenizer does not contain {token}")
        return value

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, *, complete: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Input must be a string in the selected representation")
        tokens = self.splitter(text)
        if "".join(tokens) != text:
            raise ValueError("Splitter did not preserve the complete input string")
        forbidden = set(self.special_tokens) - set(self.isolate_tokens)
        if any(token in forbidden for token in tokens):
            raise ValueError("Input must not contain BOS, EOS, or PAD markers")
        unknown = sorted({token for token in tokens if token not in self.encode_map})
        if unknown:
            raise ValueError(f"Chemical tokens absent from this pretrained mapping: {unknown}")
        surrogate = "".join(self.encode_map[token] for token in tokens)
        ids = self.tokenizer.encode(surrogate, add_special_tokens=False).ids
        # An unknown BPE character must never be silently omitted.
        recovered = "".join(self.tokenizer.id_to_token(i) for i in ids)
        if recovered != surrogate:
            raise ValueError("BPE encoding did not preserve the mapped input")
        return [self.bos_id, *ids, *([self.eos_id] if complete else [])]

    def readable_tokens(self, ids: list[int]) -> list[str]:
        result = []
        for token_id in ids:
            token = self.tokenizer.id_to_token(token_id)
            if token is None:
                raise ValueError(f"Unknown token ID: {token_id}")
            if token in self.special_tokens:
                result.append(token)
            else:
                result.append("".join(self.decode_map.get(c, c) for c in token))
        return result
