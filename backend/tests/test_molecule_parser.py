"""Unit tests for the molecule resolution and graph construction services.

Run with:
    pytest backend/tests -v
"""

import pytest

from app.core.exceptions import InvalidMoleculeError, MoleculeTooLargeError
from app.schemas.models import InputType
from app.services import graph_builder, molecule_parser


class TestSmilesParsing:
    def test_valid_smiles_aspirin(self):
        resolved = molecule_parser.resolve_molecule(
            "CC(=O)OC1=CC=CC=C1C(=O)O", InputType.SMILES
        )
        info = molecule_parser.build_molecule_info(resolved)
        assert info.molecular_formula == "C9H8O4"
        assert info.num_heavy_atoms == 13
        assert info.resolved_from == InputType.SMILES

    def test_invalid_smiles_raises(self):
        with pytest.raises(InvalidMoleculeError):
            molecule_parser.resolve_molecule("not_a_smiles(((", InputType.SMILES)

    def test_auto_mode_detects_smiles(self):
        resolved = molecule_parser.resolve_molecule("CCO", InputType.AUTO)
        assert resolved.resolved_from == InputType.SMILES

    def test_single_atom_molecule(self):
        # Water without hydrogens explicit, and elemental sodium ion.
        resolved = molecule_parser.resolve_molecule("O", InputType.SMILES)
        info = molecule_parser.build_molecule_info(resolved)
        assert info.num_heavy_atoms == 1

    def test_molecule_too_large_raises(self, monkeypatch):
        from app.core import config

        settings = config.get_settings()
        monkeypatch.setattr(settings, "MAX_HEAVY_ATOMS", 2)
        with pytest.raises(MoleculeTooLargeError):
            molecule_parser.resolve_molecule("CC(=O)OC1=CC=CC=C1C(=O)O", InputType.SMILES)


class TestGraphConstruction:
    def _graph_for(self, smiles: str):
        resolved = molecule_parser.resolve_molecule(smiles, InputType.SMILES)
        info = molecule_parser.build_molecule_info(resolved)
        return graph_builder.build_graph(resolved.mol, info)

    def test_ethanol_graph_shape(self):
        graph = self._graph_for("CCO")
        assert len(graph.nodes) == 3  # C, C, O (heavy atoms only)
        assert len(graph.edges) == 2  # C-C, C-O
        for node in graph.nodes:
            assert len(node.features) == graph.node_feature_dim
        for edge in graph.edges:
            assert len(edge.features) == graph.edge_feature_dim

    def test_benzene_ring_detection(self):
        graph = self._graph_for("c1ccccc1")
        assert all(node.is_aromatic for node in graph.nodes)
        assert all(node.is_in_ring for node in graph.nodes)
        assert len(graph.nodes) == 6
        assert len(graph.edges) == 6

    def test_single_atom_no_bonds(self):
        graph = self._graph_for("O")
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0


class TestGNNInference:
    def test_embedding_shapes(self):
        from app.schemas.models import GNNConfig

        resolved = molecule_parser.resolve_molecule("CCO", InputType.SMILES)
        info = molecule_parser.build_molecule_info(resolved)
        graph = graph_builder.build_graph(resolved.mol, info)

        from app.services import gnn_model

        config = GNNConfig(latent_dim=32, num_layers=2, hidden_dim=64)
        node_emb, graph_emb = gnn_model.run_inference(graph, config)

        assert len(node_emb) == len(graph.nodes)
        assert all(len(v) == 32 for v in node_emb)
        assert len(graph_emb) == 32

    def test_single_atom_inference_does_not_crash(self):
        from app.schemas.models import GNNConfig
        from app.services import gnn_model

        resolved = molecule_parser.resolve_molecule("O", InputType.SMILES)
        info = molecule_parser.build_molecule_info(resolved)
        graph = graph_builder.build_graph(resolved.mol, info)

        node_emb, graph_emb = gnn_model.run_inference(graph, GNNConfig())
        assert len(node_emb) == 1
        assert len(graph_emb) == GNNConfig().latent_dim
