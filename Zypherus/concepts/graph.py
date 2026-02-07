"""Enhanced concept graph with semantic relationships and hierarchies.

Features:
- Semantic relationship types (synonym, parent-child, causal, etc.)
- Hierarchical organization (taxonomies)
- Co-occurrence weighting and decay
- Path finding for concept chains
- Concept clustering
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from logging import getLogger

logger = getLogger(__name__)


class RelationType(Enum):
    """Types of semantic relationships."""
    CO_OCCURS = "co_occurs"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    PARENT = "parent"
    CHILD = "child"
    PART_OF = "part_of"
    HAS_PART = "has_part"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SIMILAR = "similar"
    RELATED = "related"


@dataclass
class ConceptNode:
    """Represents a concept in the graph."""
    name: str
    created: float = field(default_factory=time.time)
    frequency: int = 0  # How often observed
    confidence: float = 0.5  # How confident we are about this concept
    domain: Optional[str] = None  # Domain category (e.g., "biology", "physics")
    meta: Dict[str, Any] = field(default_factory=dict)
    is_abstract: bool = False  # Whether this is a general/abstract concept
    synonyms: Set[str] = field(default_factory=set)
    
    def update_frequency(self) -> None:
        """Increment frequency counter."""
        self.frequency += 1


@dataclass
class ConceptEdge:
    """Represents a relationship between concepts."""
    source: str
    target: str
    relation_type: RelationType
    weight: float = 1.0  # Strength of relationship (0-1)
    created: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    confidence: float = 0.5  # How confident we are about this relationship
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def decay_weight(self, days: float = 1.0, decay_rate: float = 0.95) -> None:
        """Apply temporal decay to relationship weight."""
        age_days = (time.time() - self.created) / (86400)
        self.weight = max(0.1, self.weight * (decay_rate ** (age_days / days)))


class ConceptGraph:
    """Enhanced concept graph with semantic relationships and hierarchies."""
    
    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.edges: List[ConceptEdge] = []
        self.edge_index: Dict[Tuple[str, RelationType], List[ConceptEdge]] = defaultdict(list)
        self.reverse_edge_index: Dict[Tuple[str, RelationType], List[ConceptEdge]] = defaultdict(list)
        self.hierarchies: Dict[str, Set[str]] = defaultdict(set)  # Taxonomies
        self.clusters: List[Set[str]] = []  # Concept clusters
        
    def upsert_node(self, concept: str, domain: Optional[str] = None, is_abstract: bool = False) -> ConceptNode:
        """Create or update a concept node."""
        if concept not in self.nodes:
            self.nodes[concept] = ConceptNode(
                name=concept,
                domain=domain,
                is_abstract=is_abstract,
            )
            logger.debug(f"Created concept node: {concept}")
        
        node = self.nodes[concept]
        node.update_frequency()
        return node
    
    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: RelationType = RelationType.RELATED,
        weight: float = 1.0,
        confidence: float = 0.5,
        bidirectional: bool = False,
    ) -> ConceptEdge:
        """Add a relationship between concepts."""
        # Ensure nodes exist
        self.upsert_node(source)
        self.upsert_node(target)
        
        # Create edge
        edge = ConceptEdge(
            source=source,
            target=target,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
        )
        
        self.edges.append(edge)
        self.edge_index[(source, relation_type)].append(edge)
        self.reverse_edge_index[(target, relation_type)].append(edge)
        
        logger.debug(f"Added edge: {source} --[{relation_type.value}]--> {target}")
        
        # Add bidirectional edge if requested
        if bidirectional and source != target:
            reverse_edge = ConceptEdge(
                source=target,
                target=source,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence,
            )
            self.edges.append(reverse_edge)
            self.edge_index[(target, relation_type)].append(reverse_edge)
            self.reverse_edge_index[(source, relation_type)].append(reverse_edge)
        
        return edge
    
    def add_hierarchy(self, parent: str, child: str) -> None:
        """Add parent-child relationship (taxonomy)."""
        self.upsert_node(parent)
        self.upsert_node(child)
        self.hierarchies[parent].add(child)
        self.add_edge(parent, child, RelationType.PARENT)
        self.add_edge(child, parent, RelationType.CHILD)
        logger.debug(f"Added hierarchy: {parent} -> {child}")
    
    def add_synonym(self, concept1: str, concept2: str) -> None:
        """Mark two concepts as synonyms."""
        self.upsert_node(concept1)
        self.upsert_node(concept2)
        self.nodes[concept1].synonyms.add(concept2)
        self.nodes[concept2].synonyms.add(concept1)
        self.add_edge(concept1, concept2, RelationType.SYNONYM, weight=0.9, bidirectional=True)
    
    def observe_cooccurrence(
        self,
        concepts: List[str],
        relation: str = "co_occurs",
        weight_decay: bool = True,
    ) -> None:
        """Record concepts appearing together."""
        concepts = [c for c in (concepts or []) if isinstance(c, str) and c.strip()]
        if len(concepts) > 25:
            concepts = concepts[:25]
        
        # Limit pairs to avoid explosion
        max_pairs = min(50, len(concepts) * (len(concepts) - 1) // 2)
        pair_count = 0
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if pair_count >= max_pairs:
                    break
                
                source, target = concepts[i], concepts[j]
                weight = 1.0 / (abs(i - j) + 1)  # Closer concepts get higher weight
                
                # Find or create edge
                relation_type = RelationType.CO_OCCURS
                existing = None
                
                for edge in self.edge_index[(source, relation_type)]:
                    if edge.target == target:
                        existing = edge
                        break
                
                if existing:
                    # Strengthen existing edge
                    existing.weight = min(1.0, existing.weight + 0.1)
                    existing.last_updated = time.time()
                else:
                    # Create new edge
                    self.add_edge(source, target, relation_type, weight=weight)
                
                pair_count += 1
    
    def get_related_concepts(
        self,
        concept: str,
        relation_type: Optional[RelationType] = None,
        max_depth: int = 2,
    ) -> List[Tuple[str, float]]:
        """Find concepts related to a given concept, with confidence scores."""
        related = []
        visited = {concept}
        queue = deque([(concept, 0, 1.0)])  # (node, depth, confidence)
        
        while queue:
            current, depth, conf = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get outgoing edges
            for rel_type in [relation_type] if relation_type else RelationType:
                for edge in self.edge_index[(current, rel_type)]:
                    target = edge.target
                    if target not in visited:
                        visited.add(target)
                        edge_conf = edge.confidence * edge.weight
                        combined_conf = conf * edge_conf
                        related.append((target, combined_conf))
                        queue.append((target, depth + 1, combined_conf))
        
        # Sort by confidence
        related.sort(key=lambda x: x[1], reverse=True)
        return related
    
    def find_path(self, source: str, target: str, max_depth: int = 5) -> Optional[List[Tuple[str, str]]]:
        """Find a path between two concepts (BFS)."""
        if source not in self.nodes or target not in self.nodes:
            return None
        
        queue = deque([(source, [])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            if len(path) >= max_depth:
                continue
            
            # Explore all relationship types
            for rel_type in RelationType:
                for edge in self.edge_index[(current, rel_type)]:
                    target_node = edge.target
                    if target_node not in visited:
                        visited.add(target_node)
                        new_path = path + [(current, edge.relation_type.value)]
                        queue.append((target_node, new_path))
        
        return None
    
    def detect_clusters(self, min_cluster_size: int = 2, similarity_threshold: float = 0.6) -> List[Set[str]]:
        """Detect clusters of related concepts."""
        clusters = []
        visited = set()
        
        for concept in self.nodes:
            if concept in visited:
                continue
            
            # Start a new cluster with BFS
            cluster = {concept}
            queue = deque([concept])
            visited.add(concept)
            
            while queue:
                current = queue.popleft()
                
                # Add related concepts
                related = self.get_related_concepts(current, max_depth=1)
                for related_concept, confidence in related:
                    if related_concept not in visited and confidence >= similarity_threshold:
                        cluster.add(related_concept)
                        visited.add(related_concept)
                        queue.append(related_concept)
            
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
        
        self.clusters = clusters
        logger.debug(f"Detected {len(clusters)} concept clusters")
        return clusters
    
    def get_hierarchy(self, root: str) -> Dict[str, List[str]]:
        """Get hierarchical structure rooted at given concept."""
        hierarchy = {}
        queue = deque([root])
        visited = {root}
        
        while queue:
            current = queue.popleft()
            children = self.hierarchies.get(current, set())
            hierarchy[current] = list(children)
            
            for child in children:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        
        return hierarchy
    
    def propagate_confidence(self) -> Dict[str, float]:
        """Propagate confidence through relationships."""
        confidences = {}
        
        # Initialize with existing node confidences
        for concept, node in self.nodes.items():
            confidences[concept] = node.confidence
        
        # Propagate through edges for 3 iterations
        for _ in range(3):
            new_confidences = confidences.copy()
            
            for edge in self.edges:
                source_conf = confidences.get(edge.source, 0.5)
                target_conf = confidences.get(edge.target, 0.5)
                
                # Average with edge confidence and weight
                propagated = (source_conf + target_conf) / 2 * edge.weight
                new_confidences[edge.target] = max(new_confidences[edge.target], propagated)
            
            confidences = new_confidences
        
        # Update node confidences
        for concept, conf in confidences.items():
            if concept in self.nodes:
                self.nodes[concept].confidence = min(1.0, conf)
        
        return confidences
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_clusters": len(self.clusters),
            "avg_connections": len(self.edges) / max(1, len(self.nodes)),
            "num_hierarchies": len(self.hierarchies),
        }


__all__ = ["ConceptGraph", "RelationType", "ConceptNode", "ConceptEdge"]
