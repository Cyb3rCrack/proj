"""Phase 2: Ontology-Aware Memory

Semantic type hierarchies and category system:
- Physical > Matter > Element > Hydrogen
- Concept > Object > Living > Animal > Mammal > Dog
- Action > Process > Learning > Studying

Enables category queries: "What are types of animals?" "What physical properties does X have?"
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict
import json
import logging

logger = logging.getLogger("ACE")


class OntologyNode:
	"""Single node in the ontology hierarchy."""
	
	def __init__(self, name: str, label: str, parent: Optional[str] = None):
		self.name = name  # Internal identifier (lowercase, no spaces)
		self.label = label  # Display name
		self.parent = parent  # Parent type
		self.children: Set[str] = set()
		self.properties: Dict[str, str] = {}
	
	def to_dict(self):
		return {
			"name": self.name,
			"label": self.label,
			"parent": self.parent,
			"children": list(self.children),
			"properties": self.properties,
		}


class Ontology:
	"""Semantic type hierarchy and categorization system."""
	
	def __init__(self):
		self.nodes: Dict[str, OntologyNode] = {}
		self.instances: Dict[str, str] = {}  # instance -> type mapping
		
		# Bootstrap with core ontology
		self._bootstrap_core_ontology()
	
	def _bootstrap_core_ontology(self):
		"""Initialize with basic hierarchy."""
		
		# Physical hierarchy
		self.add_node("physical", "Physical", parent=None)
		self.add_node("matter", "Matter", parent="physical")
		self.add_node("element", "Element", parent="matter")
		self.add_node("compound", "Compound", parent="matter")
		
		# Abstract concepts
		self.add_node("concept", "Concept", parent=None)
		self.add_node("information", "Information", parent="concept")
		self.add_node("knowledge", "Knowledge", parent="information")
		
		# Actions/processes
		self.add_node("action", "Action", parent=None)
		self.add_node("process", "Process", parent="action")
		self.add_node("learning", "Learning", parent="process")
	
	def add_node(self, name: str, label: str, parent: Optional[str] = None) -> bool:
		"""Add node to ontology.
		
		Args:
			name: Internal identifier
			label: Display label
			parent: Parent type name
		
		Returns:
			True if added, False if already exists or parent missing
		"""
		name = str(name).strip().lower()
		
		if name in self.nodes:
			return False
		
		if parent and parent not in self.nodes:
			logger.warning(f"Parent {parent} not found for {name}")
			return False
		
		node = OntologyNode(name, label, parent)
		self.nodes[name] = node
		
		if parent:
			self.nodes[parent].children.add(name)
		
		return True
	
	def add_instance(self, instance_name: str, type_name: str) -> bool:
		"""Add instance of a type.
		
		Args:
			instance_name: Instance identifier
			type_name: Type it's an instance of
		
		Returns:
			True if added, False if type doesn't exist
		"""
		type_name = str(type_name).strip().lower()
		instance_name = str(instance_name).strip().lower()
		
		if type_name not in self.nodes:
			logger.warning(f"Type {type_name} not found")
			return False
		
		self.instances[instance_name] = type_name
		return True
	
	def get_type(self, instance_name: str) -> Optional[str]:
		"""Get type of an instance."""
		instance_name = str(instance_name).strip().lower()
		return self.instances.get(instance_name)
	
	def get_ancestors(self, type_name: str) -> List[str]:
		"""Get all ancestor types (path to root).
		
		Returns:
			List from specific to general
		"""
		type_name = str(type_name).strip().lower()
		ancestors = []
		
		current = type_name
		visited = set()
		
		while current:
			if current in visited:
				break
			visited.add(current)
			ancestors.append(current)
			
			if current in self.nodes:
				current = self.nodes[current].parent
			else:
				break
		
		return ancestors
	
	def get_descendants(self, type_name: str) -> Set[str]:
		"""Get all subtypes (entire subtree)."""
		type_name = str(type_name).strip().lower()
		
		if type_name not in self.nodes:
			return set()
		
		descendants = set()
		queue = [type_name]
		visited = set()
		
		while queue:
			current = queue.pop(0)
			if current in visited:
				continue
			visited.add(current)
			
			if current in self.nodes:
				for child in self.nodes[current].children:
					descendants.add(child)
					queue.append(child)
		
		return descendants
	
	def get_siblings(self, type_name: str) -> Set[str]:
		"""Get sibling types (same parent)."""
		type_name = str(type_name).strip().lower()
		
		if type_name not in self.nodes:
			return set()
		
		node = self.nodes[type_name]
		if not node.parent:
			return set()
		
		parent_node = self.nodes[node.parent]
		siblings = parent_node.children.copy()
		siblings.discard(type_name)
		
		return siblings
	
	def get_instances_of_type(self, type_name: str, include_subtypes: bool = True) -> Set[str]:
		"""Get all instances of a type.
		
		Args:
			type_name: Type to query
			include_subtypes: Include instances of subtypes
		
		Returns:
			Set of instance names
		"""
		type_name = str(type_name).strip().lower()
		
		instances = set()
		
		# Direct instances
		for instance, instance_type in self.instances.items():
			if instance_type == type_name:
				instances.add(instance)
		
		# Instances of subtypes
		if include_subtypes:
			descendants = self.get_descendants(type_name)
			for instance, instance_type in self.instances.items():
				if instance_type in descendants:
					instances.add(instance)
		
		return instances
	
	def is_subtype_of(self, type1: str, type2: str) -> bool:
		"""Check if type1 is a subtype of type2."""
		type1 = str(type1).strip().lower()
		type2 = str(type2).strip().lower()
		
		ancestors = self.get_ancestors(type1)
		return type2 in ancestors
	
	def most_specific_common_type(self, type1: str, type2: str) -> Optional[str]:
		"""Find lowest common ancestor."""
		type1 = str(type1).strip().lower()
		type2 = str(type2).strip().lower()
		
		ancestors1 = set(self.get_ancestors(type1))
		ancestors2 = self.get_ancestors(type2)
		
		for ancestor in ancestors2:
			if ancestor in ancestors1:
				return ancestor
		
		return None
	
	def save(self, path: str = "data/knowledge/ontology.json"):
		"""Save ontology to file."""
		data = {
			"nodes": {k: v.to_dict() for k, v in self.nodes.items()},
			"instances": self.instances,
		}
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	
	def load(self, path: str = "data/knowledge/ontology.json"):
		"""Load ontology from file."""
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			
			# Rebuild nodes
			self.nodes = {}
			for name, node_dict in data.get("nodes", {}).items():
				node = OntologyNode(node_dict["name"], node_dict["label"], node_dict["parent"])
				node.children = set(node_dict.get("children", []))
				node.properties = node_dict.get("properties", {})
				self.nodes[name] = node
			
			self.instances = data.get("instances", {})
		except Exception as e:
			logger.exception(f"Failed to load ontology: {e}")
	
	def describe_type(self, type_name: str) -> Dict:
		"""Get full description of a type."""
		type_name = str(type_name).strip().lower()
		
		if type_name not in self.nodes:
			return {}
		
		node = self.nodes[type_name]
		ancestors = self.get_ancestors(type_name)
		descendants = self.get_descendants(type_name)
		instances = self.get_instances_of_type(type_name, include_subtypes=False)
		
		return {
			"name": node.name,
			"label": node.label,
			"parent": node.parent,
			"ancestors": ancestors,
			"children": list(node.children),
			"all_descendants": descendants,
			"instances": instances,
			"instance_count": len(instances),
		}


__all__ = ["Ontology", "OntologyNode"]
