"""Property order and training-set standardization."""

import json
import math
from pathlib import Path

import torch

PROPERTY_NAMES = (
    "MW", "LogP", "HBD", "HBA", "ROB", "TPSA", "AROM", "QED",
    "SAS", "ALERTS", "Asphericity", "NPR1", "NPR2", "SNDM",
    "3D_Polar_SA", "3D_NonPolar_SA", "Total_SASA",
)


class PropertyScaler:
    def __init__(self, path: str | Path, property_names=PROPERTY_NAMES):
        with open(path, encoding="utf-8") as handle:
            stats = json.load(handle)
        self.property_names = tuple(property_names)
        means = [float(stats[name]["mean"]) for name in self.property_names]
        stds = [float(stats[name]["std"]) for name in self.property_names]
        if not all(math.isfinite(v) for v in means + stds) or any(v <= 0 for v in stds):
            raise ValueError("Scaler requires finite means and positive finite standard deviations")
        self.mean = torch.tensor(means, dtype=torch.float32)
        self.std = torch.tensor(stds, dtype=torch.float32)

    def inverse_transform(self, predictions):
        return predictions.float() * self.std.to(predictions.device) + self.mean.to(predictions.device)
