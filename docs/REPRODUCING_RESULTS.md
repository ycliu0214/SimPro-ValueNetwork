# Inputs for Analysis Figures

Prediction and figure analysis are separate. Viewing included figures requires no dataset; reanalysis needs the complete study data or evaluation cache.

| Analysis | Required inputs |
| --- | --- |
| Representation fidelity | Pre-filter conversion statuses and total count |
| Property distributions and shape | Source, molecule IDs, 17 labels, selection fields |
| Prefix RMSE、AUEC / Prefix RMSE and AUEC | Property-prefix trajectories and scaler |
| RMSE、PCC / RMSE and PCC | Paired targets, predictions, and scaler |
| MW-stratified PCC | Per-record true MW, property targets and predictions |
| BPE vocabulary and length | Full test strings and tokenizer/mapping |
| Depth comparison | Test predictions per setting, or full test inputs and weights |
| KV-cache speed | Timings, actual sample count, valid tokens, hardware settings |

Evaluation caches may contain trajectory and predictions entries. Retain representation, tokenization setting, depth, and property names/order for every configuration; targets and predictions must remain paired by record.

Fidelity denominators must use pre-filter data. Overall PCC cannot reconstruct MW-stratified PCC. Per-molecule timing must use actual sample counts rather than a fixed constant.

Recomputed inference requires matching weights and fixed preprocessing. Benchmarks must define padding, valid lengths, GPU synchronization, and timing scope. Plotting dependencies may include pandas, NumPy, Matplotlib, seaborn, SciPy, scikit-learn, and mpltern; redraw-only work does not require a GPU.

See [TRAINING.md](TRAINING.md) for training procedures. Weighted z-score losses in fine-tuning history differ from original-scale property errors and PCC. Compare custom models using consistent test inputs and evaluation definitions, retaining each model's configuration.
