# SPDX-License-Identifier: Apache-2.0
# Portions adapted from the SAFE project: https://github.com/datamol-io/safe
# Modified for SimPro-ValueNetwork with local RDKit helpers and project wrappers.
# See the distribution's LICENSE, NOTICE, and THIRD_PARTY_NOTICES.md.
import re
import itertools
from collections import Counter
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

# ==========================================
# Exceptions
# ==========================================
class SAFEDecodeError(ValueError): pass
class SAFEEncodeError(Exception): pass
class SAFEFragmentationError(Exception): pass

# ==========================================
# Tokenizer / Splitter
# ==========================================
class SAFESplitter:
    REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%\([0-9]{1,5}\)|\%[0-9]{2}|[0-9])"""

    def __init__(self, pattern=None):
        self.regex = re.compile(pattern or self.REGEX_PATTERN)

    def tokenize(self, line):
        if isinstance(line, str):
            tokens = list(self.regex.findall(line))
            reconstruction = "".join(tokens)
            if line != reconstruction:
                raise ValueError(f"Token mismatch:\ntokens {reconstruction}\nsample {line}.")
        else:
            idxs = re.finditer(self.regex, str(line))
            tokens = [line[match.start(0) : match.end(0)] for match in idxs]
        return tokens

def safe_split(safe_str: str):
    """Split a SAFE string into chemical tokens."""
    return SAFESplitter().tokenize(safe_str)

# ==========================================
# Helper Functions (Replacing datamol)
# ==========================================
def _to_mol(inp, remove_hs=False):
    if isinstance(inp, Chem.Mol):
        mol = inp
    else:
        mol = Chem.MolFromSmiles(inp)
    if mol is None:
        return None
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    return mol

def _to_smiles(mol, canonical=True):
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=canonical)

# ==========================================
# Core SAFE Converter
# ==========================================
class SAFEConverter:
    SUPPORTED_SLICERS = ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]
    __SLICE_SMARTS = {
        "hr": ["[*]!@-[*]"],
        "recap": [
            "[$([C;!$(C([#7])[#7])](=!@[O]))]!@[$([#7;+0;!D1])]",
            "[$(C=!@O)]!@[$([O;+0])]",
            "[$([N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*]))]-!@[$([*])]",
            "[$(C(=!@O)([#7;+0;D2,D3])!@[#7;+0;D2,D3])]!@[$([#7;+0;D2,D3])]",
            "[$([O;+0](-!@[#6!$(C=O)])-!@[#6!$(C=O)])]-!@[$([#6!$(C=O)])]",
            "C=!@C",
            "[N;+1;D4]!@[#6]",
            "[$([n;+0])]-!@C",
            "[$([O]=[C]-@[N;+0])]-!@[$([C])]",
            "c-!@c",
            "[$([#7;+0;D2,D3])]-!@[$([S](=[O])=[O])]",
        ],
        "mmpa": ["[#6+0;!$(*=,#[!#6])]!@!=!#[*]"],
        "attach": ["[*]!@[*]"],
        "rotatable": ["[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"],
    }

    def __init__(self, slicer="brics", require_hs=None, use_original_opener_for_attach=True, ignore_stereo=False):
        self.slicer = slicer
        if isinstance(slicer, str) and slicer.lower() in self.SUPPORTED_SLICERS:
            self.slicer = self.__SLICE_SMARTS.get(slicer.lower(), slicer)
        if self.slicer != "brics" and isinstance(self.slicer, str):
            self.slicer = [self.slicer]
        if isinstance(self.slicer, (list, tuple)):
            self.slicer = [Chem.MolFromSmarts(x) for x in self.slicer]
            if any(x is None for x in self.slicer):
                raise ValueError(f"Slicer: {slicer} cannot be valid")
        self.require_hs = require_hs or (slicer == "attach")
        self.use_original_opener_for_attach = use_original_opener_for_attach
        self.ignore_stereo = ignore_stereo

    @staticmethod
    def randomize(mol, rng=None):
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()
        if mol.GetNumAtoms() == 0:
            return mol
        atom_indices = list(range(mol.GetNumAtoms()))
        atom_indices = rng.permutation(atom_indices).tolist()
        return Chem.RenumberAtoms(mol, atom_indices)

    @staticmethod
    def _format_ring_closure(num: int) -> str:
        if num < 10:
            return str(num)
        if num < 100:
            return f"%{num}"
        return f"%({num})"

    @classmethod
    def _find_branch_number_positions(cls, inp: str):
        matches = re.finditer(r"\[[^\]]+\]|%\((\d+)\)|%(\d{2})|(\d+)", inp)
        branch_numbers = []
        for match in matches:
            extended, double, singles = match.groups()
            if extended:
                branch_numbers.append((int(extended), match.start()))
            elif double:
                branch_numbers.append((int(double), match.start()))
            elif singles:
                branch_numbers.extend(
                    (int(digit), match.start(3) + offset) for offset, digit in enumerate(singles)
                )
        return branch_numbers

    @classmethod
    def _find_branch_number(cls, inp: str):
        return [label for label, _ in cls._find_branch_number_positions(inp)]

    def _ensure_valid(self, inp: str):
        missing_tokens = [inp]
        branch_numbers = self._find_branch_number(inp)
        branch_numbers = Counter(branch_numbers)
        for i, (bnum, bcount) in enumerate(branch_numbers.items()):
            if bcount % 2 != 0:
                bnum_str = self._format_ring_closure(bnum)
                _tk = f"[*:{i+1}]{bnum_str}"
                if self.use_original_opener_for_attach:
                    _tk = f"[*:{bnum}]{bnum_str}"
                missing_tokens.append(_tk)
        return ".".join(missing_tokens)

    @staticmethod
    def _canonical_isomeric_graph(inp):
        mol = _to_mol(inp, remove_hs=False)
        if mol is None:
            return None
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def decoder(self, inp: str, as_mol=False, canonical=False, fix=True, remove_dummies=True, remove_added_hs=True):
        if fix:
            inp = self._ensure_valid(inp)
        mol = _to_mol(inp)
        if mol is None:
            raise ValueError("SAFE string could not be parsed into a molecule")
        if remove_dummies:
            dummy_query = Chem.MolFromSmarts("[$([#0]!-!:*);$([#0;D1])]")
            if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
                replacements = Chem.ReplaceSubstructs(mol, dummy_query, Chem.MolFromSmiles("C"), True)
                mol = replacements[0]
                mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
        if as_mol:
            if remove_added_hs:
                mol = Chem.RemoveHs(mol, updateExplicitCount=True)
            return mol
            
        out = _to_smiles(mol, canonical=canonical)
        has_stereo = any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
        ) or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())
        
        mol_graph = self._canonical_isomeric_graph(mol)
        if has_stereo and self._canonical_isomeric_graph(out) != mol_graph:
            canonical_out = _to_smiles(mol, canonical=True)
            out = canonical_out if self._canonical_isomeric_graph(canonical_out) == mol_graph else inp
        return out

    def _fragment(self, mol, allow_empty=False):
        if self.slicer is None:
            matching_bonds = []
        elif callable(self.slicer):
            matching_bonds = list(self.slicer(mol))
        elif self.slicer == "brics":
            matching_bonds = BRICS.FindBRICSBonds(mol)
            matching_bonds = [brics_match[0] for brics_match in matching_bonds]
        else:
            matches = set()
            for smarts in self.slicer:
                matches |= {tuple(sorted(match)) for match in mol.GetSubstructMatches(smarts, uniquify=True)}
            matching_bonds = list(matches)

        if not matching_bonds and not allow_empty:
            raise SAFEFragmentationError("Slicing algorithms did not return any bonds that can be cut !")

        if not self.ignore_stereo:
            specified_stereos = [
                stereo for stereo in Chem.FindPotentialStereo(mol)
                if stereo.specified == Chem.StereoSpecified.Specified
            ]
            specified_double_bonds = {
                stereo.centeredOn for stereo in specified_stereos
                if stereo.type == Chem.StereoType.Bond_Double
            }
            specified_atom_centers = {
                stereo.centeredOn for stereo in specified_stereos
                if stereo.type in {
                    Chem.StereoType.Atom_Tetrahedral, Chem.StereoType.Atom_SquarePlanar,
                    Chem.StereoType.Atom_TrigonalBipyramidal, Chem.StereoType.Atom_Octahedral,
                }
            }

            atoms_near_stereocenters = set(specified_atom_centers)
            frontier = set(specified_atom_centers)
            if self.require_hs:
                for _ in range(2):
                    frontier = {
                        neighbor.GetIdx()
                        for atom_idx in frontier
                        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                        if neighbor.GetAtomicNum() != 1 and neighbor.GetIdx() not in atoms_near_stereocenters
                    }
                    atoms_near_stereocenters.update(frontier)

            stereo_safe_bonds = []
            for atom_pair in matching_bonds:
                bond = mol.GetBondBetweenAtoms(*atom_pair)
                if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                    continue
                atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in atom_pair]
                adjacent_stereo_bonds = {
                    adjacent_bond.GetIdx()
                    for atom_idx in atom_pair
                    for adjacent_bond in mol.GetAtomWithIdx(atom_idx).GetBonds()
                    if adjacent_bond.GetIdx() in specified_double_bonds
                }
                cuts_explicit_stereo_hydrogen = (self.require_hs and bool(adjacent_stereo_bonds) and any(atom.GetAtomicNum() == 1 for atom in atoms))
                cuts_noncarbon_stereocenter = any(atom_idx in specified_atom_centers and atom.GetAtomicNum() != 6 for atom_idx, atom in zip(atom_pair, atoms))
                cuts_hydrogen_near_stereocenter = self.require_hs and any(
                    atom.GetAtomicNum() == 1 and any(neighbor.GetIdx() in atoms_near_stereocenters for neighbor in atom.GetNeighbors())
                    for atom in atoms
                )
                cuts_multiple_bond_with_ez = (bool(specified_double_bonds) and bond.GetBondType() != Chem.BondType.SINGLE)
                
                if bond.GetBondType() == Chem.BondType.SINGLE and (len(adjacent_stereo_bonds) > 1 or cuts_explicit_stereo_hydrogen):
                    continue
                if cuts_noncarbon_stereocenter or cuts_hydrogen_near_stereocenter or cuts_multiple_bond_with_ez:
                    continue
                stereo_safe_bonds.append(atom_pair)
            matching_bonds = stereo_safe_bonds

        return matching_bonds or []

    def encoder(self, inp, canonical=True, randomize=False, seed=None, constraints=None, allow_empty=False, rdkit_safe=True):
        source_text = inp if isinstance(inp, str) else None
        source_mol = _to_mol(inp, remove_hs=False)
        if source_mol is None:
            raise ValueError("Input could not be parsed into a molecule")
        if not self.ignore_stereo and source_mol.GetStereoGroups():
            raise SAFEEncodeError("Enhanced CXSMILES stereo groups are not representable in SAFE 1.0.")

        rng = None
        should_randomize = bool(randomize and not canonical)
        if should_randomize:
            rng = np.random.default_rng(seed)
            inp = self.randomize(source_mol, rng)

        if isinstance(inp, Chem.Mol):
            inp = _to_smiles(inp, canonical=canonical)
        elif canonical:
            inp = _to_smiles(source_mol, canonical=True)

        branch_numbers = self._find_branch_number(inp)
        mol = _to_mol(inp, remove_hs=False)
        if mol is None:
            raise ValueError("Input could not be parsed into a molecule")
            
        has_specified_stereo = any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in source_mol.GetAtoms()
        ) or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in source_mol.GetBonds())
        
        if self.ignore_stereo:
            Chem.RemoveStereochemistry(mol)

        bond_map_id = 1
        open_attachment_ids = set()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetDegree() == 1:
                    open_attachment_ids.add(bond_map_id)
                atom.SetAtomMapNum(0)
                atom.SetIsotope(bond_map_id)
                bond_map_id += 1

        if self.require_hs:
            mol = Chem.AddHs(mol)
            
        matching_bonds = self._fragment(mol, allow_empty=allow_empty)
        substructed_ignored = []
        if constraints is not None:
            substructed_ignored = list(
                itertools.chain(*[mol.GetSubstructMatches(c, uniquify=True) for c in constraints])
            )

        bonds = []
        for i_a, i_b in matching_bonds:
            if any((i_a in ignore_x and i_b in ignore_x) for ignore_x in substructed_ignored):
                continue
            obond = mol.GetBondBetweenAtoms(i_a, i_b)
            bonds.append(obond.GetIdx())

        if len(bonds) > 0:
            mol = Chem.FragmentOnBonds(
                mol, bonds, dummyLabels=[(i + bond_map_id, i + bond_map_id) for i in range(len(bonds))]
            )

        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if should_randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(frags, key=lambda x: x.GetNumAtoms(), reverse=True)

        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0]
            frags_str.append(
                Chem.MolToSmiles(
                    frag, isomericSmiles=True, canonical=True,
                    rootedAtAtom=non_map_atom_idxs[0] if non_map_atom_idxs else -1,
                )
            )

        scaffold_str = ".".join(frags_str)
        scf_branch_num = self._find_branch_number(scaffold_str) + branch_numbers
        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str))
        attach_pos = sorted(attach_pos, reverse=not canonical)
        starting_num = 1 if len(scf_branch_num) == 0 else max(scf_branch_num) + 1
        
        for attach in attach_pos:
            val = self._format_ring_closure(starting_num)
            attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
            n_matches = len(attach_regexp.findall(scaffold_str))
            attachment_match = re.fullmatch(r"\[(\d+)\*\]", attach)
            is_explicit_attachment = (attachment_match is not None and int(attachment_match.group(1)) in open_attachment_ids)
            scaffold_str = (
                attach_regexp.sub(val, scaffold_str)
                if n_matches > 1 or is_explicit_attachment
                else scaffold_str.replace(attach, "*")
            )
            starting_num += 1

        wrong_attach = re.compile(r"(?<!%)\((%\(\d+\)|[\%\d]*)\)")
        scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
        
        if rdkit_safe:
            pattern = r"\(([=-@#\/\\]{0,2})(%\(\d+\)|%?\d{1,2})\)"
            replacement = r"\g<1>\g<2>"
            scaffold_str = re.sub(pattern, replacement, scaffold_str)
            
        if not self.ignore_stereo and has_specified_stereo:
            source_graph = self._canonical_isomeric_graph(source_mol)
            encoded_graph = self._canonical_isomeric_graph(
                self.decoder(scaffold_str, canonical=True, remove_dummies=False)
            )
            if source_graph is None or source_graph != encoded_graph:
                if source_text is not None:
                    return source_text
                return Chem.MolToSmiles(source_mol, canonical=True, isomericSmiles=True)
                
        return scaffold_str

# ==========================================
# Main Wrapper Endpoints
# ==========================================
def safe_encode(inp, canonical=True, randomize=False, seed=None, slicer=None, require_hs=None, constraints=None, ignore_stereo=False, allow_empty=False):
    """Convert a SMILES string or Mol object to SAFE."""
    if slicer is None:
        slicer = "brics"
    safe_obj = SAFEConverter(slicer=slicer, require_hs=require_hs, ignore_stereo=ignore_stereo)
    try:
        return safe_obj.encoder(
            inp, canonical=canonical, randomize=randomize, constraints=constraints, seed=seed, allow_empty=allow_empty
        )
    except Exception as e:
        raise SAFEEncodeError(f"Failed to encode {inp} with {slicer}") from e

def safe_decode(safe_str, as_mol=False, canonical=False, fix=True, remove_added_hs=True, remove_dummies=True, ignore_errors=False):
    """Decode SAFE to a SMILES string or Mol object."""
    safe_obj = SAFEConverter()
    try:
        return safe_obj.decoder(
            safe_str, as_mol=as_mol, canonical=canonical, fix=fix, remove_dummies=remove_dummies, remove_added_hs=remove_added_hs
        )
    except Exception as e:
        if ignore_errors:
            return None
        raise SAFEDecodeError(f"Failed to decode {safe_str}") from e

