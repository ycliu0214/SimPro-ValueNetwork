"""Resolve fixed pretrained preprocessing assets; paths may change, contents may not."""

import hashlib
import json
from pathlib import Path

from .settings import asset_stem

ASSETS = Path(__file__).parent / "assets"


def _payload(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_bundle(representation, setting, tokenizer_prefix=None, scaler_path=None):
    reference_prefix = ASSETS / "tokenizers" / asset_stem(representation, setting)
    prefix = Path(tokenizer_prefix).expanduser() if tokenizer_prefix is not None else reference_prefix
    scaler = Path(scaler_path).expanduser() if scaler_path is not None else ASSETS / "value_network_scaler.json"
    pairs = [(Path(str(prefix) + suffix), Path(str(reference_prefix) + suffix))
             for suffix in ("_tokenizer.json", "_mapping.json")]
    pairs.append((scaler, ASSETS / "value_network_scaler.json"))
    fingerprint = hashlib.sha256()
    for actual, reference in pairs:
        expected = _payload(reference)
        value = _payload(actual)
        if value != expected:
            raise ValueError(
                f"Preprocessing asset differs from the selected pretrained bundle: {actual}. "
                "Replacing or refitting tokenizer, mapping, or scaler is not supported; "
                "custom paths must refer to copies of the selected pretrained assets."
            )
        fingerprint.update(json.dumps(expected, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8"))
    return prefix, scaler, fingerprint.hexdigest()
