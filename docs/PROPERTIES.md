# Predicted Properties and Label Definitions

## Property calculation

The model predicts 17 complete-molecule properties. Graph descriptors use a view without explicit hydrogens; 3D descriptors use the hydrogen-containing conformer. Property order and scaler statistics are fixed.

## Graph-based properties

| Order | Field | Label calculation | Units or meaning |
| --- | --- | --- | --- |
| 1 | MW | `Descriptors.MolWt(mol)` | Molecular-weight value in g/mol |
| 2 | LogP | `Descriptors.MolLogP(mol)` | Dimensionless RDKit estimate |
| 3 | HBD | `Lipinski.NumHDonors(mol)` | Hydrogen-bond donor count |
| 4 | HBA | `Lipinski.NumHAcceptors(mol)` | Hydrogen-bond acceptor count |
| 5 | ROB | `Lipinski.NumRotatableBonds(mol)` | Rotatable bonds under function defaults |
| 6 | TPSA | `Descriptors.TPSA(mol)` | Topological polar surface area, Å²; not 3D SASA |
| 7 | AROM | `rdMolDescriptors.CalcNumAromaticRings(mol)` | Aromatic ring count |
| 8 | QED | `QED.qed(mol)` | Dimensionless drug-likeness score |
| 9 | SAS | RDKit Contrib `sascorer.calculateScore(mol)` | Synthetic accessibility score, not surface area |
| 10 | ALERTS | `len(catalog.GetMatches(mol))` | Matching entries in the combined PAINS/BRENK catalogs |

ALERTS is neither a count of unique atom sites nor a boolean; multiple entries can match related structures. SAS recalculation requires the corresponding RDKit Contrib scorer and its scoring data.

## Coordinate-dependent properties

| Order | Field | Label calculation | Units or meaning |
| --- | --- | --- | --- |
| 11 | Asphericity | `CalcAsphericity(mol_3d)` | Dimensionless shape descriptor |
| 12 | NPR1 | `CalcNPR1(mol_3d)` | I1/I3 |
| 13 | NPR2 | `CalcNPR2(mol_3d)` | I2/I3 |
| 14 | SNDM | Gasteiger-charge dipole magnitude divided by heavy-atom count | Debye per heavy atom |
| 15 | 3D_Polar_SA | FreeSASA areas of N and O atoms | Å² |
| 16 | 3D_NonPolar_SA | FreeSASA areas of every other atom | Å² |
| 17 | Total_SASA | `freesasa.calcCoord(...).totalArea()` | Å² |

The shape functions receive the explicit-hydrogen conformer without extra arguments. Recalculation must retain hydrogen handling and the installed function defaults. Principal moments are ordered I1 ≤ I2 ≤ I3. These labels depend on the actual conformer; a molecular string does not uniquely reconstruct the same coordinates.

## SNDM details

After `ComputeGasteigerCharges(mol_3d)`, the code uses all atoms, including explicit hydrogens:

```text
mu_vector = sum(q_i * r_i)
dipole_D = norm(mu_vector) * 4.803204
SNDM = dipole_D / number_of_heavy_atoms
```

Coordinates come from the RDKit conformer; the multiplier converts e·Å to Debye. Missing, NaN, or infinite atomic charges become zero. SNDM is not filled when there are no heavy atoms. It is a charge-model/conformer calculation, not a quantum-chemical or experimental dipole. The code neither recenters coordinates nor neutralizes the molecule here; for net-charged molecules the coordinate origin affects the sum.

## SASA details

All atoms, including hydrogens, are passed to FreeSASA. Radii come from `GetRvdw(atomic_number)`. `calcCoord(coords, radii)` leaves probe radius, algorithm, and other parameters to the installed version's defaults. Record the FreeSASA version, probe radius, and algorithm when recalculating labels; configuration and conformer differences affect results.

Polar means atomic numbers 7 and 8 only. Sulfur, phosphorus, halogens, and all hydrogens contribute to NonPolar. Replacing this definition with another polarity convention changes the labels. Polar and NonPolar partition the per-atom areas and should sum to Total within numerical error. The model independently regresses all three and does not enforce that constraint.

## Interpreting predictions

Outputs are continuous even for counts and bounded scores. Inference neither rounds counts nor clips QED, shape, or surface values to a range or nonnegative values. The model's scaler restores training z-scores to these units; no uncertainty intervals are provided.

Prefix predictions estimate the final molecule, not freshly calculated properties of the current fragment. Complete-string prediction reads EOS; prefix mode omits EOS. Even complete-string 3D predictions are learned estimates, not newly generated conformers.

Prediction uses strings, tokenizer, weights, and scaler. FreeSASA is required only when calculating surface-area labels. Comparisons to independently calculated labels must align conformers, hydrogen handling, charges, radii, and polarity rules. See [data preparation](DATA_PREPARATION.md).

Training CSVs supply labels in the units above; the model scaler converts them to z-scores. See [TRAINING.md](TRAINING.md) for temporal loss and fine-tuning.

## Tensor Standardization Option

predict(), CLI, and tensor methods default to property units. predict_tensor(), predict_ids(), and step_tensor() additionally accept standardized=True to return model z-scores in the same fixed 17-column order. Do not interpret z-scores as physical property units.
