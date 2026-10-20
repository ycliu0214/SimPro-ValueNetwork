"""Public names for the supported tokenization settings."""

TOKENIZATION_SETTINGS = (5000, 10000, 20000, 50000, "no_bpe")


def normalize_setting(value):
    if value == "no_bpe":
        return "no_bpe"
    try:
        frequency = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Use 5000, 10000, 20000, 50000, or 'no_bpe'") from exc
    if str(value) != str(frequency) or frequency not in TOKENIZATION_SETTINGS:
        raise ValueError("Use 5000, 10000, 20000, 50000, or 'no_bpe'")
    return frequency


def asset_stem(representation, setting):
    setting = normalize_setting(setting)
    if setting == "no_bpe":
        return f"{representation}_no_bpe"
    return f"{representation}_bpe_min_freq_{setting}"
