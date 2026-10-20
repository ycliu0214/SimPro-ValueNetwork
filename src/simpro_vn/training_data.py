"""Stream labeled CSV data into indexed, disk-backed training datasets."""

from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .scaling import PROPERTY_NAMES

TEXT_COLUMNS = {"smiles": "NEW_Isomeric_SMILES", "smarts": "SMARTS",
                "safe": "SAFE", "selfies": "SELFIES"}


class EncodedDataset(Dataset):
    """Keep byte offsets in RAM and read encoded examples from disk on demand."""

    def __init__(self, path):
        self.path = Path(path)
        self.offsets = array("Q")
        self._handle = None
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                if not handle.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        record = json.loads(self._handle.readline())
        return torch.tensor(record["ids"], dtype=torch.long), torch.tensor(record["z"], dtype=torch.float32)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def collate_examples(examples, pad_id):
    tokens, labels = zip(*examples)
    return (pad_sequence(tokens, batch_first=True, padding_value=pad_id),
            torch.stack(labels), torch.tensor([len(ids) for ids in tokens], dtype=torch.long))


def prepare_data(config, tokenizer, scaler, max_seq_len, output_dir):
    """Validate all rows before optimization; never fit preprocessing assets."""
    destination = Path(output_dir) / "data"
    destination.mkdir()
    text_column = config.text_column or TEXT_COLUMNS[config.representation]
    counts = {"Train": 0, "Val": 0, "Test": 0, "rejected": 0}
    fingerprint = hashlib.sha256()
    means, stds = scaler.mean.tolist(), scaler.std.tolist()
    database = sqlite3.connect(destination / "groups.sqlite")
    database.execute("CREATE TABLE groups (key TEXT PRIMARY KEY, split TEXT NOT NULL)")
    try:
        with Path(config.dataset_path).open(encoding="utf-8-sig", newline="") as source, \
             (destination / "train.jsonl").open("wb") as train, \
             (destination / "val.jsonl").open("wb") as val, \
             (destination / "rejections.jsonl").open("w", encoding="utf-8") as rejected:
            reader = csv.DictReader(source)
            columns = set(reader.fieldnames or [])
            if len(columns) != len(reader.fieldnames or []):
                raise ValueError("CSV column names must be unique")
            required = {text_column, config.split_column, *PROPERTY_NAMES}
            if config.group_column is not None:
                required.add(config.group_column)
            missing = required - columns
            if missing:
                raise ValueError(f"CSV is missing columns: {sorted(missing)}")
            group_column = config.group_column or ("Parent_Group" if "Parent_Group" in columns else None)
            for row_number, row in enumerate(reader, start=2):
                split = (row[config.split_column] or "").strip().lower()
                split = {"train": "Train", "val": "Val", "validation": "Val", "test": "Test"}.get(split)
                if split is None:
                    raise ValueError(f"CSV row {row_number}: split must be Train, Val, Validation, or Test")
                if split == "Test":
                    counts["Test"] += 1
                    continue
                try:
                    text = row[text_column]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("Empty molecular string")
                    ids = tokenizer.encode(text, complete=True)
                    if len(ids) > max_seq_len:
                        raise ValueError(f"Encoded length {len(ids)} exceeds {max_seq_len}")
                    if "".join(tokenizer.readable_tokens(ids)[1:-1]) != text:
                        raise ValueError("Tokenizer/mapping round trip did not preserve the molecular string")
                    labels = [float(row[name]) for name in PROPERTY_NAMES]
                    z = [(value - mean) / std for value, mean, std in zip(labels, means, stds)]
                    if not all(math.isfinite(v) and abs(v) <= torch.finfo(torch.float32).max for v in labels + z):
                        raise ValueError("Labels and standardized labels must be finite float32 values")
                    if group_column and not (row[group_column] or "").strip():
                        raise ValueError(f"Empty {group_column}")
                except (ValueError, TypeError, KeyError) as exc:
                    counts["rejected"] += 1
                    rejected.write(json.dumps({"csv_row": row_number, "reason": str(exc)}) + "\n")
                    if not config.skip_invalid:
                        raise ValueError(f"CSV row {row_number}: {exc}. See data/rejections.jsonl") from exc
                    continue
                # Check exact strings and, when available, supplied parent groups.
                keys = ["text:" + hashlib.sha256(text.encode()).hexdigest()]
                if group_column:
                    keys.append("parent:" + hashlib.sha256(row[group_column].strip().encode()).hexdigest())
                for key in keys:
                    found = database.execute("SELECT split FROM groups WHERE key=?", (key,)).fetchone()
                    if found and found[0] != split:
                        raise ValueError(f"Train/Val overlap at CSV row {row_number}; keep parent groups together")
                    database.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (key, split))
                payload = (json.dumps({"ids": ids, "z": z}, separators=(",", ":"), allow_nan=False) + "\n").encode()
                (train if split == "Train" else val).write(payload)
                fingerprint.update(split.encode() + b":" + payload)
                counts[split] += 1
                if row_number % 10000 == 0:
                    database.commit()
                    print(f"Prepared CSV rows: {row_number - 1}", flush=True)
        if not counts["Train"] or not counts["Val"]:
            raise ValueError("Both Train and Val must contain at least one accepted molecule")
        summary = {**counts, "fingerprint": fingerprint.hexdigest(), "text_column": text_column,
                   "group_column": group_column, "max_seq_len": max_seq_len}
        (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        database.close()
