# Training Smoke Data

This processed sample comes from the project's ZINC250k/MOSES data preparation workflow. Original download locations, dataset revisions, and per-record upstream provenance were not recorded. It is separate from the project's Apache-2.0 code/checkpoint grant and remains subject to applicable upstream data terms. See [Third-Party Notices](../../THIRD_PARTY_NOTICES.md) for source references, research citations, and the ZINC redistribution terms. The inclusion of this 300-record example does not authorize redistribution of the full upstream datasets.

`value_network_smoke.csv` contains 300 molecular records, four string representations, and 17 property labels. Split_Set has 100 Train, 100 Val, and 100 Test rows. Training uses Train and Val only and excludes Test.

Use these records to check training, fine-tuning, saving, and reload with the production-size network, rather than to establish generalization performance. No Parent_Group column is included. For production datasets, establish parent groups and splits using the [preparation workflow](../../docs/DATA_PREPARATION.md).

The fine-tuning and curriculum notebooks and configs/training/ configurations use this CSV by default. Set DATASET or dataset_path to your own data when needed, keeping property names and units consistent.
