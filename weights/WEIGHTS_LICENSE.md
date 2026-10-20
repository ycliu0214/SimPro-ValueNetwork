# Pretrained Checkpoint License

The pretrained SimPro-ValueNetwork checkpoints listed in [manifest.json](manifest.json) are licensed under the **Apache License, Version 2.0**. The complete license is supplied in [LICENSE](../LICENSE).

## Covered Files

This grant covers the 60 best pretrained checkpoint files identified by their paths and SHA256 digests in `weights/manifest.json`: SMILES, SMARTS, SAFE, and SELFIES; minimum frequencies 5000, 10000, 20000, and 50000, plus `no_bpe`; recurrent depths 2, 4, and 6.

Identical checkpoint bytes remain covered when renamed or relocated. The manifest identifies the released artifacts; it does not add terms to the Apache License. Local archive checkpoints outside this manifest are not included in this release's grant. New third-party fine-tuning outputs are not automatically represented as releases by this project.

## Use and Redistribution

Use, modification, and redistribution are governed by the Apache License, including its redistribution conditions, patent provisions, trademark limitations, and warranty/liability disclaimers. Include the license and applicable notices when redistributing covered material, and provide the required notices for modifications. No additional research-only or noncommercial restriction is imposed by this checkpoint declaration.

Each official checkpoint archive supplies LICENSE, NOTICE, THIRD_PARTY_NOTICES.md, this declaration, the model inventory/descriptions, and SHA256 checksums. The original checkpoint bytes are preserved during packaging.

## Scope of Other Materials

This grant does not relicense upstream datasets, the processed CSV sample, separately installed software, or third-party pretrained models. The SAFE converter code attribution and dependency licenses are documented in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). SAFE-GPT checkpoints are not included in these releases.

Model predictions and research results are described in the project documentation. The license supplies the applicable warranty disclaimer; this declaration does not promise prediction accuracy, fitness for a particular use, or upstream dataset rights.
