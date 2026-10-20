"""Small-data SMILES preparation example. No training or import-time processing.

Uses CPU RDKit MMFF94 for conformer optimization.
Every accepted row has its own converged 3D optimization and all 17 labels.
"""

import hashlib
import math
import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, FilterCatalog
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Contrib.SA_Score import sascorer
import freesasa

from simpro_vn.scaling import PROPERTY_NAMES


def expand_smiles(data, smiles_column="SMILES", *, enumerate_stereo=True, max_isomers=32):
    if max_isomers < 1:
        raise ValueError("max_isomers must be positive")
    if smiles_column not in data:
        raise ValueError(f"Missing input column: {smiles_column}")
    rows, rejected, seen = [], [], {}
    for source_id, value in enumerate(data[smiles_column].tolist()):
        try:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Empty or nonstring SMILES")
            mol = Chem.MolFromSmiles(value)
            if mol is None or mol.GetNumHeavyAtoms() == 0:
                raise ValueError("SMILES could not be parsed as a nonempty molecule")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            parent = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            # Group stereoisomers together before any later split.
            group = hashlib.sha256(parent.encode()).hexdigest()
            if enumerate_stereo:
                options = StereoEnumerationOptions(
                    onlyUnassigned=True, unique=True, tryEmbedding=True, maxIsomers=max_isomers,
                )
                isomers = sorted({Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
                                  for m in EnumerateStereoisomers(mol, options=options)})
            else:
                isomers = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)]
            if not isomers:
                raise ValueError("No stereoisomer passed the embedding heuristic")
            for smiles in isomers:
                if smiles in seen:
                    rejected.append({"Source_Row": source_id, "Input_SMILES": value,
                                     "stage": "deduplication", "detail": f"Duplicate of source row {seen[smiles]}"})
                    continue
                seen[smiles] = source_id
                rows.append({"Source_Row": source_id, "Input_SMILES": value,
                             "NEW_Isomeric_SMILES": smiles, "Parent_Group": group,
                             "Enumeration_Enabled": enumerate_stereo,
                             "Enumeration_Possibly_Capped": enumerate_stereo and len(isomers) >= max_isomers})
        except Exception as exc:
            rejected.append({"Source_Row": source_id, "Input_SMILES": str(value),
                             "stage": "enumeration", "detail": str(exc)})
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def optimize_all(expanded, *, seed=42, max_iterations=1000):
    """Optimize every row, including rows with fully specified stereo or no stereo."""
    from tqdm.auto import tqdm
    optimized, rejected = [], []
    for record in tqdm(expanded.to_dict("records"), desc="3D optimization (CPU MMFF94)"):
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(record["NEW_Isomeric_SMILES"]))
            if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
                raise ValueError("3D embedding failed")
            if not AllChem.MMFFHasAllMoleculeParams(mol):
                raise ValueError("MMFF parameters unavailable")
            status = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=max_iterations)
            if status != 0:
                raise ValueError(f"MMFF94 did not converge; status={status}")
            forcefield = AllChem.MMFFGetMoleculeForceField(
                mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94"),
            )
            energy = float(forcefield.CalcEnergy())
            if not math.isfinite(energy):
                raise ValueError("Nonfinite force-field energy")
            if not all(math.isfinite(v) for position in mol.GetConformer().GetPositions() for v in position):
                raise ValueError("Nonfinite coordinates")
            heavy = Chem.RemoveHs(mol)
            Chem.SanitizeMol(heavy)
            restored = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
            if restored != record["NEW_Isomeric_SMILES"]:
                raise ValueError("Canonical isomeric SMILES changed after structure preparation")
            optimized.append({**record, "Mol_Binary": mol.ToBinary(), "MMFF_Energy": energy,
                              "Optimization_Backend": "rdkit_cpu_mmff94", "Optimization_Status": "Converged",
                              "Embedding_Seed": seed, "Max_Iterations": max_iterations})
        except Exception as exc:
            rejected.append({**record, "stage": "optimization", "detail": str(exc)})
    return pd.DataFrame(optimized), pd.DataFrame(rejected)


def calculate_all(optimized):
    """Calculate labels from optimized coordinates without regenerating conformers."""
    from tqdm.auto import tqdm
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog.FilterCatalog(params)
    labeled, rejected = [], []
    for record in tqdm(optimized.to_dict("records"), desc="17 properties"):
        metadata = {key: value for key, value in record.items() if key != "Mol_Binary"}
        try:
            if record["Optimization_Status"] != "Converged":
                raise ValueError("Optimization must converge before label calculation")
            mol3d = Chem.Mol(record["Mol_Binary"])
            mol = Chem.RemoveHs(mol3d)
            props = {
                "MW": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol),
                "HBD": Lipinski.NumHDonors(mol), "HBA": Lipinski.NumHAcceptors(mol),
                "ROB": Lipinski.NumRotatableBonds(mol), "TPSA": Descriptors.TPSA(mol),
                "AROM": rdMolDescriptors.CalcNumAromaticRings(mol), "QED": QED.qed(mol),
                "SAS": sascorer.calculateScore(mol), "ALERTS": len(catalog.GetMatches(mol)),
                "Asphericity": rdMolDescriptors.CalcAsphericity(mol3d),
                "NPR1": rdMolDescriptors.CalcNPR1(mol3d), "NPR2": rdMolDescriptors.CalcNPR2(mol3d),
            }
            rdPartialCharges.ComputeGasteigerCharges(mol3d)
            conf = mol3d.GetConformer()
            mu, coords, radii, polar = [0.0, 0.0, 0.0], [], [], set()
            replaced_charges = 0
            for index, atom in enumerate(mol3d.GetAtoms()):
                try:
                    charge = float(atom.GetProp("_GasteigerCharge"))
                except (KeyError, ValueError):
                    charge = float("nan")
                if not math.isfinite(charge):
                    charge = 0.0
                    replaced_charges += 1
                position = conf.GetAtomPosition(index)
                xyz = [position.x, position.y, position.z]
                mu = [a + charge * b for a, b in zip(mu, xyz)]
                coords.extend(xyz)
                radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
                if not math.isfinite(radius) or radius <= 0:
                    raise ValueError("Unsupported van der Waals radius")
                radii.append(radius)
                if atom.GetAtomicNum() in (7, 8):
                    polar.add(index)
            props["SNDM"] = math.sqrt(sum(v * v for v in mu)) * 4.803204 / mol.GetNumHeavyAtoms()
            surface = freesasa.calcCoord(coords, radii)
            props["3D_Polar_SA"] = sum(surface.atomArea(i) for i in polar)
            props["3D_NonPolar_SA"] = sum(surface.atomArea(i) for i in range(mol3d.GetNumAtoms()) if i not in polar)
            props["Total_SASA"] = surface.totalArea()
            if not all(math.isfinite(float(props[name])) for name in PROPERTY_NAMES):
                raise ValueError("At least one label is nonfinite")
            labeled.append({**metadata, **props, "Replaced_Gasteiger_Charges": replaced_charges})
        except Exception as exc:
            rejected.append({**metadata, "stage": "properties", "detail": str(exc)})
    return pd.DataFrame(labeled), pd.DataFrame(rejected)


def check_compatibility(labeled, predictor):
    """Check SMILES labels against the selected bundle and checkpoint context length."""
    if predictor.representation != "smiles":
        raise ValueError("This preparation example supports the SMILES model only")
    ready, rejected = [], []
    for record in labeled.to_dict("records"):
        try:
            ids = predictor.encode(record["NEW_Isomeric_SMILES"], complete=True)
            labels = [float(record[name]) for name in predictor.scaler.property_names]
            zscore = [(v - float(m)) / float(s) for v, m, s in
                      zip(labels, predictor.scaler.mean, predictor.scaler.std)]
            if not all(math.isfinite(v) for v in zscore):
                raise ValueError("Model scaler produced nonfinite standardized labels")
            reconstructed = "".join(predictor.tokenizer.readable_tokens(ids)[1:-1])
            if reconstructed != record["NEW_Isomeric_SMILES"]:
                raise ValueError("Tokenizer/mapping round trip failed")
            ready.append({**record, "Encoded_Length": len(ids), "Max_Absolute_ZScore": max(map(abs, zscore)),
                          "Bundle_Fingerprint": predictor.bundle_fingerprint})
        except Exception as exc:
            rejected.append({**record, "stage": "model_compatibility", "detail": str(exc)})
    return pd.DataFrame(ready), pd.DataFrame(rejected)


def split_by_parent(ready, *, val_fraction=0.2, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = sorted(ready["Parent_Group"].unique())
    if len(groups) < 2:
        raise ValueError("Need at least two independent parent groups for Train/Val split")
    random.Random(seed).shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    validation = set(groups[:count])
    result = ready.copy()
    result["Split_Set"] = result["Parent_Group"].map(lambda g: "Val" if g in validation else "Train")
    return result
