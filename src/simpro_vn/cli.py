"""Command-line prediction without a Colab dependency."""

import argparse
import json
from pathlib import Path

from .predictor import ValueNetworkPredictor
from .representations import REPRESENTATIONS
from .settings import normalize_setting


def main():
    parser = argparse.ArgumentParser(description="SimPro-ValueNetwork: predict molecular properties from complete strings or prefixes")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint or model state_dict")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="smiles")
    parser.add_argument("--min-frequency", type=normalize_setting, default=5000,
                        metavar="{5000,10000,20000,50000,no_bpe}",
                        help="Pretrained tokenization setting; use no_bpe for chemical tokens without BPE merges")
    parser.add_argument("--depth", type=int, choices=(2, 4, 6), default=6)
    parser.add_argument("--tokenizer-prefix", help="Path prefix without _tokenizer.json or _mapping.json")
    parser.add_argument("--scaler", help="Path to the scaler JSON used for this model")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", action="append", help="Input string; repeat for multiple inputs")
    source.add_argument("--input-file", type=Path, help="UTF-8 file with one molecular string per line")
    parser.add_argument("--prefix", action="store_true", help="Exclude EOS for incomplete prefixes")
    parser.add_argument("--trajectory", action="store_true", help="Return predictions at every token")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    texts = args.text if args.text is not None else args.input_file.read_text(encoding="utf-8-sig").splitlines()
    predictor = ValueNetworkPredictor(
        checkpoint_path=args.checkpoint, representation=args.representation,
        min_frequency=args.min_frequency, depth=args.depth,
        tokenizer_prefix=args.tokenizer_prefix, scaler_path=args.scaler, device=args.device,
    )
    if args.trajectory:
        result = [predictor.predict_trajectory(t, complete=not args.prefix) for t in texts]
    else:
        result = predictor.predict(texts, complete=not args.prefix, batch_size=args.batch_size)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
