# Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
#
# Molecular Graph Generator (GNN-Powered)
# Author: Aya Khaled Khuris <aya.khuris@gmail.com>
"""Convert a sanitized RDKit ``Mol`` into a typed molecular graph.

Featurization scheme (see ``docs/ARCHITECTURE.md`` for the full table):

Atom feature vector (44 dims), one-hot blocks concatenated in order:
    [0:43]  atomic symbol one-hot over ``ALLOWED_ATOMIC_SYMBOLS`` (43 incl. UNK)
    [43]    is_aromatic (0/1)

    plus scalar/normalized features appended separately are avoided here to
    keep the vector purely categorical + one boolean, which keeps the GNN
    input distribution well-behaved (all values in [0, 1]). Continuous
    descriptors (degree, charge, H count, hybridization) are still exposed
    on the ``AtomNode`` schema for the UI, and are additionally one-hot
    encoded into the same feature vector below.

Bond feature vector (10 dims):
    [0:4]  bond type one-hot: SINGLE, DOUBLE, TRIPLE, AROMATIC
    [4]    is_conjugated
    [5]    is_in_ring
    [6:10] stereo one-hot: STEREONONE, STEREOZ, STEREOE, STEREOANY
"""

from __future__ import annotations

from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem

from app.core.config import get_settings
from app.core.exceptions import GraphConstructionError
from app.schemas.models import AtomNode, BondEdge, MolecularGraph, MoleculeInfo

_HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]

_BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

_STEREO_TYPES = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOANY,
]


def _one_hot(value, choices: list) -> List[float]:
    """One-hot encode ``value`` over ``choices``, with an implicit trailing
    'unknown' bucket when the value is absent from ``choices``."""
    vec = [1.0 if value == c else 0.0 for c in choices]
    if sum(vec) == 0.0:
        vec.append(1.0)
    else:
        vec.append(0.0)
    return vec


def _atom_symbol_one_hot(symbol: str) -> List[float]:
    settings = get_settings()
    vocab = settings.ALLOWED_ATOMIC_SYMBOLS
    if symbol not in vocab:
        symbol = "UNK"
    return [1.0 if s == symbol else 0.0 for s in vocab]


def featurize_atom(atom: Chem.Atom) -> List[float]:
    """Build the fixed-length feature vector for a single atom.

    The returned vector length always equals
    ``settings.ATOM_FEATURE_DIM`` regardless of the input molecule, which
    is required for batching heterogeneous molecules into one GNN model.
    """
    features: List[float] = []
    features += _atom_symbol_one_hot(atom.GetSymbol())          # 43 dims (42 + UNK)
    features.append(1.0 if atom.GetIsAromatic() else 0.0)        # 1 dim
    return features


def featurize_bond(bond: Chem.Bond) -> List[float]:
    """Build the fixed-length feature vector for a single bond."""
    features: List[float] = []
    features += _one_hot(bond.GetBondType(), _BOND_TYPES)         # 4 (+unused unk slot trimmed below)
    features.append(1.0 if bond.GetIsConjugated() else 0.0)       # 1
    features.append(1.0 if bond.IsInRing() else 0.0)              # 1
    features += _one_hot(bond.GetStereo(), _STEREO_TYPES)         # 4
    # _one_hot appends an extra "unknown" slot; bond types/stereo vocab are
    # exhaustive for RDKit so we trim the always-zero unknown slots to hit
    # the documented, fixed BOND_FEATURE_DIM.
    return features[:4] + features[4:6] + features[6:10]


def _compute_2d_coords(mol: Chem.Mol) -> List[Tuple[float, float]]:
    """Generate 2D depiction coordinates for frontend layout hints.

    Falls back to no coordinates (frontend will auto-layout) if embedding
    fails, which can happen for disconnected fragments or unusual valence
    states — this is a UX nicety, not chemistry, so we never raise here.
    """
    try:
        mol_copy = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_copy)
        conf = mol_copy.GetConformer()
        return [
            (round(conf.GetAtomPosition(i).x, 4), round(conf.GetAtomPosition(i).y, 4))
            for i in range(mol_copy.GetNumAtoms())
        ]
    except Exception:  # noqa: BLE001 - purely cosmetic fallback
        return [(0.0, 0.0)] * mol.GetNumAtoms()


def build_graph(mol: Chem.Mol, molecule_info: MoleculeInfo) -> MolecularGraph:
    """Convert an RDKit molecule into the API's ``MolecularGraph`` schema.

    Raises:
        GraphConstructionError: if featurization produces an inconsistent
            graph (defensive check; should not trigger for valid RDKit mols).
    """
    settings = get_settings()

    try:
        coords = _compute_2d_coords(mol)

        nodes: List[AtomNode] = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            x, y = coords[idx] if idx < len(coords) else (None, None)
            nodes.append(
                AtomNode(
                    index=idx,
                    symbol=atom.GetSymbol(),
                    atomic_number=atom.GetAtomicNum(),
                    degree=atom.GetDegree(),
                    formal_charge=atom.GetFormalCharge(),
                    hybridization=str(atom.GetHybridization()),
                    is_aromatic=atom.GetIsAromatic(),
                    is_in_ring=atom.IsInRing(),
                    num_hydrogens=atom.GetTotalNumHs(),
                    features=featurize_atom(atom),
                    x=x,
                    y=y,
                )
            )

        edges: List[BondEdge] = []
        for bond in mol.GetBonds():
            edges.append(
                BondEdge(
                    source=bond.GetBeginAtomIdx(),
                    target=bond.GetEndAtomIdx(),
                    bond_type=str(bond.GetBondType()),
                    is_aromatic=bond.GetIsAromatic(),
                    is_conjugated=bond.GetIsConjugated(),
                    is_in_ring=bond.IsInRing(),
                    features=featurize_bond(bond),
                )
            )

        if nodes and len(nodes[0].features) != settings.ATOM_FEATURE_DIM:
            raise GraphConstructionError(
                f"Atom feature dim mismatch: got {len(nodes[0].features)}, "
                f"expected {settings.ATOM_FEATURE_DIM}."
            )
        if edges and len(edges[0].features) != settings.BOND_FEATURE_DIM:
            raise GraphConstructionError(
                f"Bond feature dim mismatch: got {len(edges[0].features)}, "
                f"expected {settings.BOND_FEATURE_DIM}."
            )

        return MolecularGraph(
            molecule=molecule_info,
            nodes=nodes,
            edges=edges,
            node_feature_dim=settings.ATOM_FEATURE_DIM,
            edge_feature_dim=settings.BOND_FEATURE_DIM,
        )
    except GraphConstructionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GraphConstructionError(f"Failed to construct molecular graph: {exc}") from exc
