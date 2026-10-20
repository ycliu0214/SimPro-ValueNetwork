import re
from typing import List

class SMARTSDecodeError(ValueError):
    pass

SMARTS_PATTERN = re.compile(
    r"(\[[^\]]+\]|"          
    r"Br?|Cl?|N|O|S|P|F|I|"  
    r"b|c|n|o|s|p|"          
    r"a|A|\*|"               
    r"\%\([0-9]{1,5}\)|\%[0-9]{2}|[0-9]|"  
    r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\$|\!|\&|\;|\,)"
)

class SMARTSSplitter:
    def __init__(self):
        self.regex = SMARTS_PATTERN

    def tokenize(self, line: str) -> List[str]:
        if not line:
            return []
        tokens = self.regex.findall(line)
        reconstruction = "".join(tokens)
        if line != reconstruction:
            raise ValueError(f"Token mismatch: uncaptured characters detected.\nReconstruction: {reconstruction}\nOriginal: {line}")
        return tokens

_smarts_splitter = SMARTSSplitter()

def smarts_split(smarts_str: str) -> List[str]:
    """Split SMARTS into tokens for the Transformer model."""
    try:
        return _smarts_splitter.tokenize(smarts_str)
    except ValueError as e:
        raise SMARTSDecodeError(f"Failed to split SMARTS string: {smarts_str}") from e
