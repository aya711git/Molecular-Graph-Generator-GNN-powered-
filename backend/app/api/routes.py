# Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for details.
#
# Molecular Graph Generator (GNN-Powered)
# Author: Aya Khaled Khuris <aya.khuris@gmail.com>

"""REST API routes.

Each route is a thin orchestration layer: it calls into
``app.services.*`` and translates domain exceptions into HTTP responses.
No chemistry or tensor logic lives here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.core.exceptions import (
    GraphConstructionError,
    InputResolutionError,
    InvalidMoleculeError,
    ModelInferenceError,
    MoleculeTooLargeError,
)
from app.schemas.models import (
    EmbedRequest,
    EmbeddingResult,
    HealthResponse,
    MolecularGraph,
    MoleculeInfo,
    MoleculeRequest,
)
from app.services import gnn_model, graph_builder, molecule_parser

logger = logging.getLogger(__name__)
router = APIRouter()


def _resolve_or_raise(query: str, input_type):
    """Shared resolution step for /parse, /graph, /embed — maps domain
    exceptions to appropriate HTTP status codes."""
    try:
        return molecule_parser.resolve_molecule(query, input_type)
    except InputResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidMoleculeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MoleculeTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    rdkit_ok, torch_ok = True, True
    try:
        import rdkit  # noqa: F401
    except ImportError:
        rdkit_ok = False
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_ok = False
    return HealthResponse(version=settings.APP_VERSION, rdkit_available=rdkit_ok, torch_available=torch_ok)


@router.post("/parse", response_model=MoleculeInfo)
def parse_molecule(request: MoleculeRequest) -> MoleculeInfo:
    """Validate and resolve raw input into canonical molecule metadata."""
    resolved = _resolve_or_raise(request.query, request.input_type)
    return molecule_parser.build_molecule_info(resolved)


@router.post("/graph", response_model=MolecularGraph)
def build_graph(request: MoleculeRequest) -> MolecularGraph:
    """Resolve input and construct the full node/edge molecular graph."""
    resolved = _resolve_or_raise(request.query, request.input_type)
    info = molecule_parser.build_molecule_info(resolved)
    try:
        return graph_builder.build_graph(resolved.mol, info)
    except GraphConstructionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/embed", response_model=EmbeddingResult)
def embed_molecule(request: EmbedRequest) -> EmbeddingResult:
    """Resolve input, build the graph, and run it through the GNN encoder."""
    resolved = _resolve_or_raise(request.query, request.input_type)
    info = molecule_parser.build_molecule_info(resolved)

    try:
        graph = graph_builder.build_graph(resolved.mol, info)
    except GraphConstructionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        node_embeddings, graph_embedding = gnn_model.run_inference(graph, request.gnn_config)
    except ModelInferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EmbeddingResult(
        graph=graph,
        node_embeddings=node_embeddings,
        graph_embedding=graph_embedding,
        embedding_dim=request.gnn_config.latent_dim,
        layer_type=request.gnn_config.layer_type,
        num_layers=request.gnn_config.num_layers,
    )
