# Model Descriptions and Tensor Interfaces

These interfaces let Python/PyTorch programs use the Value Network. Dictionary-based predict() and the CLI remain available. Hugging Face, a generator, and training data are not required.

## Model Descriptions

The package's assets/pretrained_models.json describes all 60 published checkpoints; weights/model_descriptions.json is a portable copy. Descriptions include representation, tokenization setting, recurrent depth, attention heads, dimensions, context limit, property order, special token IDs, preprocessing fingerprint, file SHA256, and size. Existing .pt files are unchanged.

```python
from simpro_vn import ValueNetworkPredictor, identify_checkpoint

description = identify_checkpoint('/path/to/model.pt')
# A known checkpoint returns its description; other weights return None.
predictor = ValueNetworkPredictor(
    checkpoint_path='/path/to/model.pt',
    representation='smiles', min_frequency=5000, depth=6,
)
print(predictor.describe())
```

Published checkpoints are identified by file SHA256 against the packaged catalog, not by filename. Renaming or relocating the same file preserves identification. Incorrect depth, representation, setting, head count, or preprocessing stops loading. Checkpoints are not downloaded automatically; supply a local path.

describe() reports one of three verification levels:

| Value | Meaning |
| --- | --- |
| sha256 | File content matches a published checkpoint |
| metadata | Other weights contain matching simpro_vn metadata; this does not identify published bytes |
| shape_only | No recognized digest or embedded metadata; only shape/preprocessing checks are available |

Matching tensor shapes cannot prove recurrent depth or token-ID semantics. Verify configuration yourself for shape_only weights. New training/fine-tuning outputs embed full descriptions; metadata checks do not certify arbitrary custom models.

## Text Input, Tensor Output

```python
values = predictor.predict_tensor(['CCO', 'CCN'], batch_size=32)
z_scores = predictor.predict_tensor(['CCO', 'CCN'], standardized=True)
prefix_values = predictor.predict_tensor('CC', complete=False)
```

Returns float32 [N,17] tensors on the predictor's selected device in input order. Defaults to property units; standardized=True returns model z-scores. Read column order from predictor.describe()['property_names']. Empty lists return [0,17].

## Token-ID Input

```python
import torch
from torch.nn.utils.rnn import pad_sequence

sequences = [torch.tensor(predictor.encode(text)) for text in ['CCO', 'C']]
ids = pad_sequence(sequences, batch_first=True,
                   padding_value=predictor.tokenizer.pad_id)
mask = ids.ne(predictor.tokenizer.pad_id)
final_values = predictor.predict_ids(ids, attention_mask=mask)
trajectory = predictor.predict_ids(ids, output_all_tokens=True)
```

input_ids must be a nonempty int32/int64 [B,T] tensor within the context limit. Each row starts with one BOS; optional EOS appears only at its last valid position. Only contiguous right padding is accepted. Mask 1/True must exactly identify non-PAD positions; omitted masks are inferred from PAD IDs.

Defaults to [B,17] from each row's final valid position. output_all_tokens=True returns [B,T,17], with padded outputs zeroed. standardized=True selects z-scores. CPU inputs are accepted; outputs use the predictor device. Caller tensors are not modified.

Direct IDs bypass text tokenization. Use exactly the same tokenizer, mapping, BPE merges, and special IDs; in-range integers do not prove matching semantics. Include EOS for complete strings and omit it for prefixes; an empty prefix contains BOS only.

## Incremental Tensor Output

```python
session = predictor.new_session()
values = session.step_tensor(predictor.encode('CCO'))
session.reset()
```

step_tensor() returns [new token count,17] and accepts integers, integer lists, or scalar/1D int32/int64 tensors, with optional standardized=True. EOS, length, and reset rules match step(). Cache is single-sequence, with no branch or batch-reorder interface, and cannot be shared with another model.

## Gradient Behavior

All three tensor interfaces use no_grad and return requires_grad=False without constructing a Value Network backward graph. When called in a normal PyTorch context, outputs are ordinary tensors usable as constants in another model's loss. They neither backpropagate through the Value Network nor update the generator automatically. An outer inference_mode still imposes its own restrictions.

See [PACKAGE_VALIDATION.md](PACKAGE_VALIDATION.md) for installation/interface validation. Existing dictionary API and CLI behavior remains available.

See [Comprehensive Colab Validation](../examples/colab_comprehensive_validation.ipynb) and its [coverage guide](COMPREHENSIVE_VALIDATION.md) for interface and installed-package checks.
