# Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
#
# Molecular Graph Generator (GNN-Powered)
# Author: Aya Khaled Khuris <aya.khuris@gmail.com>
"""Domain-specific exception hierarchy.

Using typed exceptions (instead of bare `ValueError`/`Exception`) lets the
API layer map failures to precise HTTP status codes and error payloads,
and lets tests assert on *why* something failed, not just *that* it failed.
"""

from __future__ import annotations


class MolGNNError(Exception):
    """Base class for all application-specific errors."""


class InputResolutionError(MolGNNError):
    """Raised when a raw user input (name/formula/SMILES) cannot be resolved
    to a chemical structure at all (e.g. PubChem lookup returns no hits)."""


class InvalidMoleculeError(MolGNNError):
    """Raised when RDKit fails to parse or sanitize a candidate structure."""


class MoleculeTooLargeError(MolGNNError):
    """Raised when a valid molecule exceeds configured safety limits."""


class GraphConstructionError(MolGNNError):
    """Raised when a valid RDKit molecule cannot be converted into a graph
    tensor (e.g. unsupported atom features)."""


class ModelInferenceError(MolGNNError):
    """Raised when the GNN forward pass fails for a well-formed graph."""
