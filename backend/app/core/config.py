# Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
#
# Molecular Graph Generator (GNN-Powered)
# Author: Aya Khaled Khuris <aya.khuris@gmail.com>
"""Application-wide configuration.

All tunables live here so the rest of the codebase never hardcodes magic
numbers or environment-dependent values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, environment-overridable settings.

    Any field can be overridden via an environment variable of the same
    name (case-insensitive), or via a `.env` file in the backend root.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MOLGNN_")

    # --- App metadata ---
    APP_NAME: str = "Molecular Graph Generator"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --- Chemistry limits (defensive bounds against pathological input) ---
    MAX_HEAVY_ATOMS: int = 150
    PUBCHEM_TIMEOUT_SECONDS: int = 8

    # --- Atom/bond featurization vocab (see docs/ARCHITECTURE.md) ---
    ALLOWED_ATOMIC_SYMBOLS: List[str] = [
        "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
        "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
        "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
        "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr",
        "Pt", "Hg", "Pb", "UNK",
    ]
    ATOM_FEATURE_DIM: int = 45  # 44 (atomic symbol one-hot incl. UNK) + 1 (is_aromatic)  
    BOND_FEATURE_DIM: int = 10

    # --- GNN hyperparameters (defaults; overridable per-request) ---
    GNN_HIDDEN_DIM: int = 128
    GNN_NUM_LAYERS: int = 3
    GNN_LATENT_DIM: int = 64
    GNN_LAYER_TYPE: str = "mpnn"  # one of: "mpnn", "gcn"
    GNN_DROPOUT: float = 0.1


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
