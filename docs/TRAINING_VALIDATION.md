# Training and Fine-Tuning Validation

These tests cover optional training examples. Start with [inference](../examples/colab_inference.ipynb); use [comprehensive validation](../examples/colab_comprehensive_validation.ipynb) to check installation, prediction, data preparation, and training together.

Use [colab_comprehensive_validation.ipynb](../examples/colab_comprehensive_validation.ipynb) to install the current project and run checks, or execute these training-only commands from the project root. Tests use Python's built-in unittest; pytest is not required.

```bash
pip install ".[chemistry]"
python -m unittest discover -s tests -p test_training.py -v
python -m unittest discover -s tests -p test_inference_interfaces.py -v
python -m unittest discover -s tests -p test_training_integration.py -v
python -m unittest discover -s tests -p test_training_cuda.py -v
```

| File | Coverage |
| --- | --- |
| [test_inference_interfaces.py](../tests/test_inference_interfaces.py) | Descriptions, SHA256 identity, tensors/IDs/masks/caches, fixed-score gradients, and training initializer metadata |
| [test_training.py](../tests/test_training.py) | Weight formulas, per-molecule normalization, padding, hand-computed loss, gradients, causality, depths, configuration |
| [test_training_integration.py](../tests/test_training_integration.py) | Scratch training, fine-tuning, curriculum branches, CLI, prediction reload, resume, source hashes, output protection, input rejection, split isolation, accumulation, representations |
| [test_training_cuda.py](../tests/test_training_cuda.py) | Small GPU training and reload with fp32, fp16, and supported bf16 |

CPU tests use tiny networks, synthetic labels, and temporary checkpoints. They do not read or modify released weights and do not measure accuracy. Integration tests compare uninterrupted and resumed CPU training; CUDA is not guaranteed to reproduce bitwise. Missing CUDA or bf16 support produces SKIP, which is not a pass for that capability.

Invalid-input checks include strings that fail token splitting: default handling must stop with the CSV row, while `skip_invalid=true` must skip and log the rejection. Decode errors for all four representations inherit from `ValueError` for shared validation handling.

Test CSVs are generated in temporary directories with four representations of five molecules, three Train rows, two Val rows, and 17 synthetic numeric labels. Negative tests deliberately replace a string, label, or group to verify rejection or logging. Correct handling of an expected error produces an OK test result; unexpected runtime errors must propagate.

The comprehensive notebook checks CLI help, isolated wheel installation, CPU formula/integration tests, and CUDA tests before the released-model matrix and real-label smoke workflow. Reports and logs are saved in a new directory; failures stop subsequent phases. Public notebooks contain no saved execution outputs. Rerun them to validate the current version in your own environment; see [recorded results](VALIDATION_RESULTS.md) for the completed Colab validation.

After functional checks, use the default [300-row dataset](../examples/data/README.md) in [colab_finetuning.ipynb](../examples/colab_finetuning.ipynb) to check production-network memory, compatibility, loss, saving, and inference. It defaults to batch size 2 and executes the first planned epoch; do not select checkpoints using the test set. The comprehensive notebook already includes the 60-model prediction matrix and a four-record production-model fine-tuning/resume check.

See [COMPREHENSIVE_VALIDATION.md](COMPREHENSIVE_VALIDATION.md) for combined installation, inference, and training checks. Rerun new description, tensor, and CUDA reload checks; saved notebook outputs do not establish that newly added checks passed.
