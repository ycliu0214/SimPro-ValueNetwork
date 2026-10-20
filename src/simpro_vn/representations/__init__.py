"""Representation-specific splitters, imported only when requested."""

from importlib import import_module

REPRESENTATIONS = ("smiles", "smarts", "safe", "selfies")


def get_splitter(representation: str):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    try:
        module = import_module(f".{representation}_core", __name__)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Dependencies for {representation!r} are missing. "
            'Install with: pip install -e ".[chemistry]"'
        ) from exc
    return getattr(module, f"{representation}_split")
