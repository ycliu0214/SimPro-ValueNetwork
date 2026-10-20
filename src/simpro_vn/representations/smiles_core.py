import re
from typing import List

class SMILESDecodeError(ValueError):
    pass

# Match bracket atoms, Br/Cl, extended ring labels %(123), two-digit labels %12, digits, bonds, and branches.
#SMILES_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%\([0-9]{1,5}\)|\%\d{2}|\d)"
SMILES_PATTERN = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\*|\%\([0-9]{1,5}\)|\%[0-9]{2}|[0-9]|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>)"
#SMILES_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#||\+|\\\\\/|:||@|\?|>|\*|\$|\%[0–9]{2}|[0–9])"
#SMILES_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
#SMILES_PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

class SMILESSplitter:
    def __init__(self):
        self.regex = re.compile(SMILES_PATTERN)

    def tokenize(self, line: str) -> List[str]:
        if not line:
            return []
        tokens = list(self.regex.findall(line))
        
        # Verify lossless reconstruction so the regular expression misses no characters.
        reconstruction = "".join(tokens)
        if line != reconstruction:
            raise ValueError(f"Token mismatch: some characters were missed.\nReconstruction: {reconstruction}\nOriginal: {line}")
        return tokens

_smiles_splitter = SMILESSplitter()

def smiles_split(smiles_str: str) -> List[str]:
    """Split a SMILES string into chemical tokens."""
    try:
        return _smiles_splitter.tokenize(smiles_str)
    except ValueError as e:
        raise SMILESDecodeError(f"Failed to split SMILES string: {smiles_str}") from e

