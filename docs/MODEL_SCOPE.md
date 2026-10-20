# Model Scope and Interpretation

## Intended Use and Training Data

SimPro-ValueNetwork estimates 17 complete-molecule properties from molecular strings and their prefixes. Typical uses include scoring completed candidates, ranking partial sequences during generation, and obtaining property estimates before more expensive calculations. A generator is not included.

The released models were trained on processed ZINC250k and MOSES data. Preparation included stereoisomer expansion, conformer optimization, representation screening, and deduplication. MOSES is itself derived from ZINC Clean Leads, so these are related molecular sources, not independent coverage of all chemical space. GuacaMol was not part of the final training dataset. See [Research Results](RESEARCH_RESULTS.md) for the reported analysis and [Third-Party Notices](../THIRD_PARTY_NOTICES.md) for upstream references and acquisition limitations.

Training on these sources does not establish general accuracy for arbitrary chemistry, such as proteins, polymers, metal complexes, or molecules with substantially different sizes, charges, elements, or structural motifs. No validated applicability-domain detector or per-prediction confidence interval is supplied. Assess prediction errors on representative labeled examples before using estimates to guide a new application; successful tokenization alone is insufficient.

## Complete Molecules and Prefixes

| Mode | Python / CLI | Interpretation |
| --- | --- | --- |
| Complete string | `predict(text)` / default CLI | Append EOS and read the final prediction as an estimate for the supplied completed molecule |
| Prefix | `predict(text, complete=False)` / `--prefix` | Omit EOS and estimate completed-molecule properties from the available prefix |
| Trajectory | `predict_trajectory(text)` / `--trajectory` | Inspect estimates at successive encoded token positions; default complete mode also includes EOS |

For example, `predict("CC")` treats `CC` as a completed molecule, whereas `predict("CC", complete=False)` treats it as a sequence that may continue. These are different requests even though the text is identical. Prefix outputs are not the calculated properties of the visible fragment. Different completions can share a prefix and have different properties, and no particular completion or continuation policy is guaranteed by the estimate.

An empty prefix is allowed and contains BOS only. Very short prefixes provide little molecular information; early predictions should not be treated as precise descriptions of an eventual candidate. Relative prefix length in the research figures refers to encoded tokens, not raw characters. BPE settings can therefore produce different token positions for the same text.

## Outputs and Units

`predict()` and the CLI return 17 continuous values in the fixed order below. Tensor methods use the same column order and default to these units. Only an explicit `standardized=True` request to a supporting tensor method returns z-scores.

| Property | Units or interpretation |
| --- | --- |
| MW | g/mol |
| LogP | Dimensionless RDKit estimate |
| HBD | Hydrogen-bond donor count |
| HBA | Hydrogen-bond acceptor count |
| ROB | Rotatable bond count |
| TPSA | Topological polar surface area, square angstroms |
| AROM | Aromatic ring count |
| QED | Dimensionless drug-likeness score |
| SAS | Synthetic accessibility score; not surface area |
| ALERTS | Number of matching PAINS/BRENK catalog entries |
| Asphericity | Dimensionless shape descriptor |
| NPR1 | Principal moment ratio I1/I3 |
| NPR2 | Principal moment ratio I2/I3 |
| SNDM | Gasteiger-charge dipole magnitude in Debye per heavy atom |
| 3D_Polar_SA | Surface area of N/O atoms, square angstroms |
| 3D_NonPolar_SA | Surface area of all remaining atoms, square angstroms |
| Total_SASA | Total solvent-accessible surface area, square angstroms |

See [Predicted Properties and Label Definitions](PROPERTIES.md) for the exact descriptor functions, hydrogen handling, charge conventions, and surface-area settings. These definitions matter when comparing predictions to independently calculated labels.

Inference does not round predicted counts or constrain outputs to physical ranges. For example, a count can be fractional, a nominally bounded score can fall outside its usual range, and surface-area estimates need not be nonnegative. The three predicted surface areas are regressed independently and need not satisfy the sum identity exactly. Any downstream clipping or rounding is a caller decision and changes the raw predictions.

The coordinate-dependent labels were calculated from prepared conformers. A molecular string does not uniquely specify those coordinates; inference neither builds a conformer nor performs MMFF optimization, FreeSASA calculation, or a quantum-chemical calculation. SNDM is a charge-model estimate, not an experimental dipole measurement.

## Input and Model Compatibility

Supply strings in the selected representation: SMILES, SMARTS, SAFE, or SELFIES. The prediction API does not automatically convert a SMILES input into another representation. SMARTS support uses the project's molecular representation conventions; it does not establish support for arbitrary reaction or substructure-query SMARTS.

Use the checkpoint's matching representation, BPE setting, depth, tokenizer, mapping, and scaler. The released context limit is 128 encoded tokens including BOS and, in complete mode, EOS. Oversized inputs raise an error rather than being truncated. Prefix text must end at a recognizable token boundary. Consult `predictor.describe()` for the selected model's metadata.

Replacing the tokenizer or refitting the scaler is unsupported. Custom asset paths must refer to content-equivalent copies of the matching fixed bundle. Token recognition does not substitute for chemical parsing, valence checks, stereochemical checks, or checking that the target application resembles the training data.

## Use with a Molecular Generator

Complete-string estimates can be used to rank candidates; prefix estimates can inform an external search or reward procedure. Evaluate shortlisted completed molecules using the property calculations appropriate to your application. Repeatedly optimizing a learned score can select candidates where that score is inaccurate; a high predicted value alone does not validate the candidate.

Text-based scoring can use a generator with a different tokenizer if its decoded text is retokenized by this predictor and satisfies the predictor's input requirements. Direct token-ID input and incremental KV-cache sessions require exactly the Value Network's tokenizer, mapping, merges, and special IDs. Integer IDs with the same numeric range are not sufficient. Arbitrarily encoding text chunks separately can change BPE merges across chunk boundaries.

The public inference tensor interfaces run without Value Network gradients. They can supply fixed scores to an external algorithm, but do not by themselves implement differentiable generator optimization or a fine-tuning algorithm. See [Model Descriptions and Tensor Interfaces](MODEL_INTERFACE.md) for integration contracts.

## Evidence and Validation

[Research Results](RESEARCH_RESULTS.md) reports aggregate performance on the study's analysis data. Those figures do not establish the error of an individual prediction or accuracy on a new dataset. [Recorded Colab Validation](VALIDATION_RESULTS.md) establishes installation, interface consistency, saving/reloading, and training-workflow functionality; those checks are not a new scientific accuracy evaluation. Neither scientific performance nor the original figures transfer automatically to user-fine-tuned checkpoints.
