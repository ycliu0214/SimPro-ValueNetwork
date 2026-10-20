"""Run after installing the project. Paths are supplied by the caller."""

import argparse

from simpro_vn import ValueNetworkPredictor

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()

predictor = ValueNetworkPredictor(checkpoint_path=args.checkpoint, depth=6)
print(predictor.predict(["CCO", "c1ccccc1"]))
print(predictor.predict("CC", complete=False))

# Start with BOS and consume one fixed encoded token stream.
session = predictor.new_session()
for token_id in predictor.encode("CCO"):
    print(session.step(token_id)[-1])
