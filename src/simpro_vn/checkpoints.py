"""Identify published checkpoints by content and validate their model descriptions."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _catalog():
    path = Path(__file__).parent / "assets" / "pretrained_models.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = {record["checkpoint"]["sha256"]: record for record in payload["models"]}
    if len(records) != len(payload["models"]):
        raise ValueError("Published checkpoint descriptions contain ambiguous SHA256 identities")
    return records


def identify_checkpoint(path):
    """Return a published model description by SHA256, or None for other weights.

    Identification never depends on the filename and does not load model tensors.
    """
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    record = _catalog().get(digest.hexdigest())
    return deepcopy(record) if record is not None else None


def checkpoint_metadata(path, checkpoint):
    """Merge the packaged description with checkpoint metadata, rejecting conflicts."""
    known = identify_checkpoint(path)
    embedded = checkpoint.get("simpro_vn", {})
    if not isinstance(embedded, dict):
        raise ValueError("Checkpoint simpro_vn metadata must be a dictionary")
    metadata = {} if known is None else {key: value for key, value in known.items() if key != "checkpoint"}
    for key, value in embedded.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"Checkpoint metadata conflicts with the published model description: {key}")
        metadata[key] = value
    verification = "sha256" if known is not None else ("metadata" if embedded else "shape_only")
    return metadata, verification
