import re
from typing import List, Optional, Union

import selfies as sf
from rdkit import Chem

# ==========================================
# Exceptions
# ==========================================
class SELFIESDecodeError(ValueError):
    pass

class SELFIESEncodeError(Exception): 
    pass

# ==========================================
# Tokenizer / Splitter
# ==========================================
def selfies_split(selfies_str: str) -> List[str]:
    """Split a SELFIES string into chemical tokens."""
    if not selfies_str:
        return []
    try:
        # Use the token iterator provided by the SELFIES package.
        return list(sf.split_selfies(selfies_str))
    except ValueError as e:
        raise SELFIESDecodeError(f"Failed to split SELFIES string: {selfies_str}") from e

# ==========================================
# Helper Functions
# ==========================================
def _to_smiles(inp: Union[str, Chem.Mol]) -> str:
    """Return a SMILES representation of the input."""
    if isinstance(inp, str):
        # Assume string input is already SMILES and return it unchanged.
        return inp
    elif isinstance(inp, Chem.Mol):
        # Clear atom-map numbers before producing SMILES.
        for atom in inp.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(inp, isomericSmiles=True, canonical=True)
    return ""

def _clean_decoded_smiles(smiles: str, preserve: bool = False) -> str:
    """Clean decoded SMILES, optionally preserving its atom-map handling."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    if preserve:
        return Chem.MolToSmiles(mol)
        
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    cleaned_smiles = Chem.MolToSmiles(mol, canonical=True)
    return re.sub(r"\[(B|C|N|O|P|S|F|I|Cl|Br|c|o|n)\]", r"\1", cleaned_smiles)

# ==========================================
# Main Wrapper Endpoints
# ==========================================
def selfies_encode(inp: Union[str, Chem.Mol]) -> str:
    """
    Convert a SMILES string or Mol object to SELFIES.
    """
    smiles_str = _to_smiles(inp)
    if not smiles_str:
        raise SELFIESEncodeError("Invalid input: could not convert to SMILES.")
        
    try:
        selfies_str = sf.encoder(smiles_str)
        if selfies_str is None:
             raise ValueError("SELFIES encoder returned None.")
        return selfies_str
    except Exception as e:
        raise SELFIESEncodeError(f"Failed to encode SMILES to SELFIES: {smiles_str}") from e

def selfies_decode(
    selfies_str: str, 
    as_mol: bool = False, 
    canonical: bool = True,
    ignore_errors: bool = False
) -> Union[str, Chem.Mol, None]:
    """
    Decode SELFIES to a SMILES string or Mol object.
    """
    try:
        decoded_smiles = sf.decoder(selfies_str)
        if decoded_smiles is None:
            raise ValueError("SELFIES decoder returned None.")
            
        if canonical:
            decoded_smiles = _clean_decoded_smiles(decoded_smiles)
            
        if as_mol:
            mol = Chem.MolFromSmiles(decoded_smiles)
            if mol is None:
                raise ValueError("Could not parse decoded SMILES into RDKit Mol.")
            return mol
            
        return decoded_smiles
        
    except Exception as e:
        if ignore_errors:
            return None if as_mol else ""
        raise SELFIESDecodeError(f"Failed to decode SELFIES string: {selfies_str}") from e

