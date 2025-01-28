import warnings

import networkx as nx
import numpy as np

from .exception import InvalidGenerationSource, UnvalidatedGenerationSource
from .util import get_global_rng


class _PartialAtomGraph:
    def __init__(self, generating_graph, static_graph, source_node):
        self.generating_graph = generating_graph
        self.static_graph = static_graph

        self.atom_graph = nx.Graph()


class AtomGraph:
    def __init__(self, ml_graph):
        self._ml_graph = ml_graph.copy()

        self._static_graph = self._create_static_graph(self.ml_graph)

        self._starting_node_idx, self._starting_node_weight = self._create_init_weights(
            self.ml_graph
        )

    @staticmethod
    def _create_init_weights(graph):
        starting_node_idx = []
        starting_node_weight = []
        for node_idx, data in graph.nodes(data=True):
            if data["init_weight"] > 0:
                starting_node_idx.append(node_idx)
                starting_node_weight.append(data["init_weight"])

        starting_node_weight = np.asarray(starting_node_weight)
        starting_node_weight /= np.sum(starting_node_weight)

        return starting_node_idx, starting_node_weight

    @staticmethod
    def _create_static_graph(ml_graph):
        static_graph = ml_graph.copy()

        edges_to_delete = set()
        for u, v, k, d in static_graph.edges(keys=True, data=True):
            if not d["static"]:
                edges_to_delete.add((u, v, k))

        static_graph.remove_edges_from(edges_to_delete)
        return static_graph

    @property
    def ml_graph(self):
        return self._ml_graph.copy()

    def _get_random_start_node(self, rng):
        return rng.choice(self._starting_node_idx, p=self._starting_node_weight)

    def get_graph(self, source: str = None, rng=None):

        if rng is None:
            rng = get_global_rng()

        if source is None:
            source = self._get_random_start_node(rng)

        if source not in self.ml_graph.nodes():
            raise InvalidGenerationSource(source, self.ml_graph.nodes(), self.ml_graph)

        if source not in self._starting_node_idx:
            warnings.warn(
                UnvalidatedGenerationSource(source, self._starting_node_idx, self.ml_graph)
            )

        atom_graph = _PartialAtomGraph(self.ml_graph, self._static_graph)
