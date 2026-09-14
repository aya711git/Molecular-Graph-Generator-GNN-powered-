"""Resolve arbitrary chemical input (SMILES / name / formula) into a
canonical, sanitized RDKit ``Mol`` object.

Design:
    * SMILES parsing is always attempted first because it is instantaneous
      and fully offline (no network dependency, no rate limits).
    * Names and formulas are resolved via PubChemPy, which wraps the
      PubChem PUG REST API. Network failures are caught and re-raised as
      typed domain exceptions so the API layer can return clean errors
      instead of leaking stack traces.
    * Every successful resolution is re-sanitized through RDKit before
      being handed to the rest of the pipeline, so downstream code can
      always assume a valid, sanitized ``Mol``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from app.core.config import get_settings
from app.core.exceptions import (
    InputResolutionError,
    InvalidMoleculeError,
    MoleculeTooLargeError,
)
from app.schemas.models import InputType, MoleculeInfo

logger = logging.getLogger(__name__)

# RDKit logs a lot of noisy parse warnings to stderr by default; we capture
# them ourselves via exceptions instead.
RDLogger.DisableLog("rdApp.*")


@dataclass
class ResolvedMolecule:
    mol: Chem.Mol
    resolved_from: InputType
    iupac_name: Optional[str] = None
    pubchem_cid: Optional[int] = None


def _try_parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Attempt to parse a raw SMILES string. Returns ``None`` (never
    raises) on failure so callers can cheaply try SMILES-first."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError):
        return None
    return mol


# def _lookup_pubchem(query: str, *, by: str) -> ResolvedMolecule:
#     try:
#         import pubchempy as pcp
#     except ImportError as exc:
#         raise InputResolutionError("PubChemPy is not installed on the server.") from exc

#     settings = get_settings()
#     try:
#         compounds = pcp.get_compounds(query, namespace=by)
#     except Exception as exc:
#         raise InputResolutionError(
#             f"Could not reach PubChem to resolve '{query}': {exc}"
#         ) from exc

#     if not compounds:
#         raise InputResolutionError(f"No compound found on PubChem for {by}='{query}'.")

#     compound = compounds[0]

#     # compound.canonical_smiles / .isomeric_smiles rely on the full compound
#     # record, which no longer reliably includes SMILES for every entry.
#     # Fall back to the dedicated properties endpoint, which is more stable.
#     smiles = compound.canonical_smiles or compound.isomeric_smiles
#     if not smiles and compound.cid:
#         try:
#             props = pcp.get_properties(
#                 ["CanonicalSMILES", "IsomericSMILES", "ConnectivitySMILES"],
#                 compound.cid,
#                 "cid",
#             )
#             if props:
#                 smiles = (
#                     props[0].get("CanonicalSMILES")
#                     or props[0].get("IsomericSMILES")
#                     or props[0].get("ConnectivitySMILES")
#                 )
#         except Exception:
#             pass  # fall through to the error below

#     if not smiles:
#         raise InputResolutionError(
#             f"PubChem entry for '{query}' has no associated structure (SMILES)."
#         )

#     mol = _try_parse_smiles(smiles)
#     if mol is None:
#         raise InvalidMoleculeError(
#             f"PubChem returned an unparsable structure for '{query}'."
#         )

#     resolved_type = InputType.NAME if by == "name" else InputType.FORMULA
#     return ResolvedMolecule(
#         mol=mol,
#         resolved_from=resolved_type,
#         iupac_name=getattr(compound, "iupac_name", None),
#         pubchem_cid=getattr(compound, "cid", None),
#     )

def _lookup_pubchem(query: str, *, by: str) -> ResolvedMolecule:
    try:
        import pubchempy as pcp
    except ImportError as exc:
        raise InputResolutionError("PubChemPy is not installed on the server.") from exc

    settings = get_settings()
    try:
        if by == "formula":
            # Formula search is an async, potentially large-result-set
            # lookup on PubChem's side (many structural isomers can share
            # the same formula). Capping listkey_count avoids waiting for
            # the full match set and drastically cuts timeout risk.
            compounds = pcp.get_compounds(query, namespace="formula", listkey_count=5)
        else:
            compounds = pcp.get_compounds(query, namespace=by)
    except Exception as exc:
        raise InputResolutionError(
            f"Could not reach PubChem to resolve '{query}': {exc}"
        ) from exc

    if not compounds:
        raise InputResolutionError(f"No compound found on PubChem for {by}='{query}'.")

    compound = compounds[0]

    # compound.canonical_smiles / .isomeric_smiles rely on the full compound
    # record, which no longer reliably includes SMILES for every entry.
    # Fall back to the dedicated properties endpoint, which is more stable.
    smiles = compound.canonical_smiles or compound.isomeric_smiles
    if not smiles and compound.cid:
        try:
            props = pcp.get_properties(
                ["CanonicalSMILES", "IsomericSMILES", "ConnectivitySMILES"],
                compound.cid,
                "cid",
            )
            if props:
                smiles = (
                    props[0].get("CanonicalSMILES")
                    or props[0].get("IsomericSMILES")
                    or props[0].get("ConnectivitySMILES")
                )
        except Exception:
            pass  # fall through to the error below

    if not smiles:
        raise InputResolutionError(
            f"PubChem entry for '{query}' has no associated structure (SMILES)."
        )

    mol = _try_parse_smiles(smiles)
    if mol is None:
        raise InvalidMoleculeError(
            f"PubChem returned an unparsable structure for '{query}'."
        )

    resolved_type = InputType.NAME if by == "name" else InputType.FORMULA
    return ResolvedMolecule(
        mol=mol,
        resolved_from=resolved_type,
        iupac_name=getattr(compound, "iupac_name", None),
        pubchem_cid=getattr(compound, "cid", None),
    )

def resolve_molecule(query: str, input_type: InputType = InputType.AUTO) -> ResolvedMolecule:
    """Resolve any supported input flavor into a sanitized RDKit molecule.

    Args:
        query: The raw user-provided string.
        input_type: Hint about how to interpret `query`. ``AUTO`` tries
            SMILES first, then falls back to a PubChem name lookup, then a
            PubChem formula lookup.

    Raises:
        InputResolutionError: input could not be resolved to any structure.
        InvalidMoleculeError: a structure was found but RDKit rejects it.
        MoleculeTooLargeError: structure exceeds configured atom-count limit.
    """
    settings = get_settings()

    if input_type == InputType.SMILES:
        mol = _try_parse_smiles(query)
        if mol is None:
            raise InvalidMoleculeError(f"'{query}' is not a valid/parsable SMILES string.")
        resolved = ResolvedMolecule(mol=mol, resolved_from=InputType.SMILES)

    elif input_type == InputType.NAME:
        resolved = _lookup_pubchem(query, by="name")

    elif input_type == InputType.FORMULA:
        resolved = _lookup_pubchem(query, by="formula")

    else:  # AUTO
        mol = _try_parse_smiles(query)
        if mol is not None:
            resolved = ResolvedMolecule(mol=mol, resolved_from=InputType.SMILES)
        else:
            try:
                resolved = _lookup_pubchem(query, by="name")
            except InputResolutionError:
                resolved = _lookup_pubchem(query, by="formula")

    num_heavy = resolved.mol.GetNumHeavyAtoms()
    if num_heavy > settings.MAX_HEAVY_ATOMS:
        raise MoleculeTooLargeError(
            f"Molecule has {num_heavy} heavy atoms; the configured limit is "
            f"{settings.MAX_HEAVY_ATOMS}."
        )
    if num_heavy == 0:
        raise InvalidMoleculeError("Resolved structure contains no heavy atoms.")

    return resolved


def build_molecule_info(resolved: ResolvedMolecule) -> MoleculeInfo:
    """Compute canonical descriptors for API responses."""
    mol = resolved.mol
    mol_with_h = Chem.AddHs(mol)

    return MoleculeInfo(
        canonical_smiles=Chem.MolToSmiles(mol),
        molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 3),
        num_atoms=mol_with_h.GetNumAtoms(),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_bonds=mol.GetNumBonds(),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        resolved_from=resolved.resolved_from,
        iupac_name=resolved.iupac_name,
        pubchem_cid=resolved.pubchem_cid,
    )
