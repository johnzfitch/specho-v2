"""
Concept Topology Extractor (3D)

Treats nouns/concepts as nodes in a graph and analyzes network structure.
AI-generated text tends to have different graph properties than human text.

Features (estimated Cohen's d):
- small_world_coefficient (d~0.9-1.4): Clustering vs path length ratio
- graph_modularity (d~0.8-1.2): Community structure strength
- long_range_edge_density (d~0.7-1.0): Connections between distant concepts

Tier 1: Requires spaCy; networkx optional (graceful degradation without it)
"""

import re
import math
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict

from .base import BaseExtractor


class TopologyExtractor(BaseExtractor):
    """
    Extract concept topology features.

    These features build a co-occurrence graph of concepts (nouns) and
    analyze its network properties. Human writing tends to have more
    modular, clustered concept networks while AI has more uniform structure.
    """

    group = "topology"
    dependencies = ["spacy"]  # networkx is optional enhancement

    def __init__(self):
        self._nlp = None
        self._loaded = False
        self._nx = None
        self._nx_loaded = False

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if not self._loaded:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
                self._loaded = True
            except Exception:
                self._nlp = None
                self._loaded = True
        return self._nlp

    @property
    def nx(self):
        """Lazy-load networkx (optional)."""
        if not self._nx_loaded:
            try:
                import networkx as nx
                self._nx = nx
                self._nx_loaded = True
            except ImportError:
                self._nx = None
                self._nx_loaded = True
        return self._nx

    @property
    def feature_names(self) -> List[str]:
        return [
            "small_world_coefficient",  # d~0.9-1.4
            "graph_modularity",         # d~0.8-1.2
            "long_range_edge_density",  # d~0.7-1.0
        ]

    def _safe_extract(self, name: str, func, default: float = 0.0) -> float:
        """Fault-isolated feature extraction."""
        try:
            return func()
        except Exception:
            return default

    def _build_cooccurrence_graph(self, doc, window_size: int = 5) -> Tuple[Dict, Dict]:
        """
        Build a co-occurrence graph from nouns in the text.

        Returns:
            - nodes: dict mapping noun lemma to first occurrence position
            - edges: dict mapping (node1, node2) to weight (co-occurrence count)
        """
        # Extract nouns with their positions
        nouns = []
        for i, token in enumerate(doc):
            if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2:
                nouns.append((token.lemma_.lower(), i))

        if len(nouns) < 5:
            return {}, {}

        # Record first position of each noun
        nodes = {}
        for lemma, pos in nouns:
            if lemma not in nodes:
                nodes[lemma] = pos

        # Build edges based on co-occurrence within window
        edges = Counter()
        for i, (lemma1, pos1) in enumerate(nouns):
            for j in range(i + 1, len(nouns)):
                lemma2, pos2 = nouns[j]
                if pos2 - pos1 > window_size:
                    break
                if lemma1 != lemma2:
                    edge = tuple(sorted([lemma1, lemma2]))
                    edges[edge] += 1

        return nodes, dict(edges)

    def _calculate_clustering_coefficient(self, nodes: Dict, edges: Dict) -> float:
        """
        Calculate average clustering coefficient without networkx.

        Clustering coefficient = fraction of node's neighbors that are
        connected to each other.
        """
        if len(nodes) < 3:
            return 0.0

        # Build adjacency list
        adj = defaultdict(set)
        for (n1, n2), weight in edges.items():
            adj[n1].add(n2)
            adj[n2].add(n1)

        # Calculate clustering for each node
        clustering_sum = 0.0
        valid_nodes = 0

        for node in nodes:
            neighbors = adj[node]
            k = len(neighbors)
            if k < 2:
                continue

            # Count edges between neighbors
            neighbor_edges = 0
            neighbors_list = list(neighbors)
            for i, n1 in enumerate(neighbors_list):
                for n2 in neighbors_list[i + 1:]:
                    edge = tuple(sorted([n1, n2]))
                    if edge in edges:
                        neighbor_edges += 1

            # Clustering coefficient for this node
            max_edges = k * (k - 1) / 2
            if max_edges > 0:
                clustering_sum += neighbor_edges / max_edges
                valid_nodes += 1

        if valid_nodes == 0:
            return 0.0

        return clustering_sum / valid_nodes

    def _calculate_avg_path_length(self, nodes: Dict, edges: Dict) -> float:
        """
        Estimate average path length using BFS from sample nodes.
        """
        if len(nodes) < 3:
            return 0.0

        # Build adjacency list
        adj = defaultdict(set)
        for (n1, n2), weight in edges.items():
            adj[n1].add(n2)
            adj[n2].add(n1)

        node_list = list(nodes.keys())

        # Sample up to 10 nodes for efficiency
        sample_nodes = node_list[:min(10, len(node_list))]

        total_distance = 0
        pair_count = 0

        for start in sample_nodes:
            # BFS from this node
            distances = {start: 0}
            queue = [start]
            idx = 0

            while idx < len(queue):
                current = queue[idx]
                idx += 1

                for neighbor in adj[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)

            # Sum distances to all reachable nodes
            for node, dist in distances.items():
                if node != start and dist > 0:
                    total_distance += dist
                    pair_count += 1

        if pair_count == 0:
            return float('inf')

        return total_distance / pair_count

    def _calculate_modularity_simple(self, nodes: Dict, edges: Dict) -> float:
        """
        Simple modularity approximation using greedy community detection.
        """
        if len(nodes) < 5 or len(edges) < 3:
            return 0.0

        # Build adjacency with weights
        adj = defaultdict(dict)
        for (n1, n2), weight in edges.items():
            adj[n1][n2] = weight
            adj[n2][n1] = weight

        # Calculate node degrees
        degrees = {}
        total_weight = sum(edges.values())
        for node in nodes:
            degrees[node] = sum(adj[node].values())

        if total_weight == 0:
            return 0.0

        # Simple greedy community assignment
        # Each node starts in its own community, merge most connected pairs
        communities = {node: i for i, node in enumerate(nodes)}

        # Merge communities greedily (simplified)
        for _ in range(min(len(nodes) // 2, 10)):
            best_merge = None
            best_gain = 0

            for (n1, n2), weight in edges.items():
                if communities[n1] != communities[n2]:
                    # Calculate modularity gain of merging
                    gain = weight / total_weight - (degrees[n1] * degrees[n2]) / (2 * total_weight ** 2)
                    if gain > best_gain:
                        best_gain = gain
                        best_merge = (n1, n2)

            if best_merge:
                old_comm = communities[best_merge[1]]
                new_comm = communities[best_merge[0]]
                for node in nodes:
                    if communities[node] == old_comm:
                        communities[node] = new_comm
            else:
                break

        # Calculate final modularity
        modularity = 0.0
        for (n1, n2), weight in edges.items():
            if communities[n1] == communities[n2]:
                expected = degrees[n1] * degrees[n2] / (2 * total_weight)
                modularity += (weight - expected) / total_weight

        return max(0.0, min(1.0, modularity + 0.5))  # Shift to 0-1 range

    def _extract(self, text: str, **kwargs) -> Dict[str, float]:
        """Extract concept topology features."""
        features = {name: 0.0 for name in self.feature_names}

        # Early return for short text
        if len(text) < 200:
            return features

        # Check if spaCy is available
        if self.nlp is None:
            return features

        # Process text with spaCy
        doc = self.nlp(text)

        # Build co-occurrence graph
        nodes, edges = self._build_cooccurrence_graph(doc)

        if len(nodes) < 5 or len(edges) < 3:
            return features

        # =====================================================================
        # Feature 1: Small world coefficient - ISOLATED
        # =====================================================================
        def calc_small_world():
            # Small-world coefficient = clustering / (path_length * random_path_length)
            # High clustering + short paths = small world

            clustering = self._calculate_clustering_coefficient(nodes, edges)
            avg_path = self._calculate_avg_path_length(nodes, edges)

            if avg_path == 0 or avg_path == float('inf'):
                features["small_world_coefficient"] = 0.5
                return

            # Expected path length for random graph with same nodes/edges
            n = len(nodes)
            m = len(edges)
            random_path = math.log(n) / math.log(max(2 * m / n, 2)) if n > 1 and m > 0 else 1

            # Small-world coefficient
            if random_path > 0:
                sigma = (clustering / 0.5) / (avg_path / random_path) if clustering > 0 else 0
                features["small_world_coefficient"] = min(sigma / 2, 1.0)
            else:
                features["small_world_coefficient"] = clustering
        self._safe_extract("small_world", calc_small_world)

        # =====================================================================
        # Feature 2: Graph modularity - ISOLATED
        # =====================================================================
        def calc_modularity():
            modularity = self._calculate_modularity_simple(nodes, edges)
            features["graph_modularity"] = modularity
        self._safe_extract("modularity", calc_modularity)

        # =====================================================================
        # Feature 3: Long-range edge density - ISOLATED
        # =====================================================================
        def calc_long_range():
            # Count edges between concepts that are far apart in the text
            # (first occurrence positions differ by more than threshold)

            if len(edges) == 0:
                features["long_range_edge_density"] = 0.0
                return

            total_edges = len(edges)
            long_range_edges = 0

            # Threshold: concepts more than 50 tokens apart
            position_threshold = 50

            for (n1, n2), weight in edges.items():
                pos1 = nodes.get(n1, 0)
                pos2 = nodes.get(n2, 0)
                if abs(pos1 - pos2) > position_threshold:
                    long_range_edges += 1

            # Ratio of long-range edges
            ratio = long_range_edges / total_edges if total_edges > 0 else 0

            # Human text often has more long-range conceptual connections
            # AI text tends to have more local structure
            features["long_range_edge_density"] = ratio
        self._safe_extract("long_range", calc_long_range)

        return features


# Standalone function
def extract_topology_features(text: str) -> Dict[str, float]:
    """Extract concept topology features (standalone function)."""
    extractor = TopologyExtractor()
    result = extractor.extract(text)
    return result.features
