# Models and Tokenization Bundles

Four representations each have five tokenization settings at depths 2/4/6: 20 preprocessing bundles and 60 models.

| Representation | Setting | Vocabulary | BPE merges | Isolated symbols |
| --- | --- | --- | --- | --- |
| smiles | 5000 | 1087 | 1016 | None |
| smiles | 10000 | 655 | 584 | None |
| smiles | 20000 | 409 | 338 | None |
| smiles | 50000 | 239 | 168 | None |
| smiles | no_bpe | 71 | 0 | None |
| smarts | 5000 | 1164 | 1111 | None |
| smarts | 10000 | 725 | 672 | None |
| smarts | 20000 | 459 | 406 | None |
| smarts | 50000 | 261 | 208 | None |
| smarts | no_bpe | 53 | 0 | None |
| safe | 5000 | 1341 | 1263 | . |
| safe | 10000 | 821 | 743 | . |
| safe | 20000 | 529 | 451 | . |
| safe | 50000 | 314 | 236 | . |
| safe | no_bpe | 78 | 0 | . |
| selfies | 5000 | 1155 | 1040 | None |
| selfies | 10000 | 731 | 616 | None |
| selfies | 20000 | 470 | 355 | None |
| selfies | 50000 | 283 | 168 | None |
| selfies | no_bpe | 115 | 0 | None |

No BPE uses `<representation>_no_bpe`, has no BPE merges, and retains BOS/EOS. SAFE isolates periods. Each tokenizer has a matching mapping; models use the fixed property scaler.

Configuration JSONs select the corresponding weights. tokenizer_manifest.json and weights/manifest.json record file sizes and SHA256 checksums for integrity checks.

Full descriptions are in weights/model_descriptions.json and bundled with the package. Known checkpoint digests validate representation, tokenization, depth, attention heads, and preprocessing. Verify settings for custom weights without descriptions; matching shapes do not prove matching semantics. See [MODEL_INTERFACE.md](MODEL_INTERFACE.md).

This catalog lists pretrained models. User-trained and fine-tuned outputs live in new run directories and do not replace catalog entries. All four representations and five settings support [training and fine-tuning](TRAINING.md).
