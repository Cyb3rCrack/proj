"""Phase 4: Versioned Definitions & Source Reliability

Track definition evolution:
- Store definition versions with timestamps
- Track contradictions between versions
- Enable rollback to previous definitions
- Track belief changes and their justifications

Source reliability tracking:
- Assign credibility scores to sources
- Track successful vs failed predictions
- Adjust weights based on accuracy
- Enable source-aware retrieval
"""

from typing import Dict, List, Optional, Any
import json
import time
import logging

logger = logging.getLogger("ACE")


class DefinitionVersion:
	"""Single version of a definition."""
	
	def __init__(self, concept: str, definition: str, source: str, timestamp: Optional[float] = None):
		self.concept = concept
		self.definition = definition
		self.source = source
		self.timestamp = timestamp or time.time()
		self.confidence = 0.5
		self.evidence_count = 0
		self.contradictions = []
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"concept": self.concept,
			"definition": self.definition,
			"source": self.source,
			"timestamp": self.timestamp,
			"confidence": self.confidence,
			"evidence_count": self.evidence_count,
			"contradictions": self.contradictions,
		}


class DefinitionHistory:
	"""Track versioned definitions for a concept."""
	
	def __init__(self, concept: str):
		self.concept = concept
		self.versions: List[DefinitionVersion] = []
		self.current_version = 0
	
	def add_version(self, definition: str, source: str, confidence: float = 0.5) -> int:
		"""Add new definition version."""
		version = DefinitionVersion(self.concept, definition, source)
		version.confidence = confidence
		self.versions.append(version)
		self.current_version = len(self.versions) - 1
		logger.info(f"New definition version for {self.concept}: v{self.current_version} from {source}")
		return self.current_version
	
	def get_current(self) -> Optional[DefinitionVersion]:
		"""Get current definition."""
		if 0 <= self.current_version < len(self.versions):
			return self.versions[self.current_version]
		return None
	
	def get_version(self, version_idx: int) -> Optional[DefinitionVersion]:
		"""Get specific version."""
		if 0 <= version_idx < len(self.versions):
			return self.versions[version_idx]
		return None
	
	def rollback(self, version_idx: int) -> bool:
		"""Rollback to previous version."""
		if 0 <= version_idx < len(self.versions):
			self.current_version = version_idx
			return True
		return False
	
	def mark_evidence(self, version_idx: int) -> None:
		"""Mark that version has supporting evidence."""
		if 0 <= version_idx < len(self.versions):
			self.versions[version_idx].evidence_count += 1
	
	def add_contradiction(self, version_idx: int, contradicting_source: str) -> None:
		"""Record contradiction for a version."""
		if 0 <= version_idx < len(self.versions):
			self.versions[version_idx].contradictions.append(contradicting_source)
	
	def get_all_versions(self) -> List[Dict[str, Any]]:
		"""Get all versions."""
		return [v.to_dict() for v in self.versions]
	
	def get_history_summary(self) -> Dict[str, Any]:
		"""Get summary of definition history."""
		return {
			"concept": self.concept,
			"version_count": len(self.versions),
			"current_version": self.current_version,
			"current_definition": self.get_current().definition if self.get_current() else None,
			"current_confidence": self.get_current().confidence if self.get_current() else 0.0,
			"versions": self.get_all_versions(),
		}


class VersionedDefinitionStore:
	"""Store versioned definitions for all concepts."""
	
	def __init__(self):
		self.definitions: Dict[str, DefinitionHistory] = {}
	
	def add_definition(self, concept: str, definition: str, source: str, confidence: float = 0.5) -> int:
		"""Add or update definition."""
		concept = str(concept).strip().lower()
		
		if concept not in self.definitions:
			self.definitions[concept] = DefinitionHistory(concept)
		
		return self.definitions[concept].add_version(definition, source, confidence)
	
	def get_definition(self, concept: str) -> Optional[str]:
		"""Get current definition."""
		concept = str(concept).strip().lower()
		if concept in self.definitions:
			version = self.definitions[concept].get_current()
			return version.definition if version else None
		return None
	
	def get_history(self, concept: str) -> Optional[Dict[str, Any]]:
		"""Get full history for concept."""
		concept = str(concept).strip().lower()
		if concept in self.definitions:
			return self.definitions[concept].get_history_summary()
		return None
	
	def rollback_definition(self, concept: str, version_idx: int) -> bool:
		"""Rollback concept to previous definition."""
		concept = str(concept).strip().lower()
		if concept in self.definitions:
			return self.definitions[concept].rollback(version_idx)
		return False
	
	def has_definition(self, concept: str) -> bool:
		"""Check if concept has definition."""
		concept = str(concept).strip().lower()
		return concept in self.definitions and self.definitions[concept].get_current() is not None
	
	def save(self, path: str = "data/knowledge/definitions.json"):
		"""Save to file."""
		data = {}
		for concept, history in self.definitions.items():
			data[concept] = history.get_history_summary()
		
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	
	def load(self, path: str = "data/knowledge/definitions.json"):
		"""Load from file."""
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			
			for concept, history_data in data.items():
				concept = str(concept).strip().lower()
				self.definitions[concept] = DefinitionHistory(concept)
				
				for version_data in history_data.get("versions", []):
					version = DefinitionVersion(
						concept=version_data["concept"],
						definition=version_data["definition"],
						source=version_data["source"],
						timestamp=version_data.get("timestamp"),
					)
					version.confidence = version_data.get("confidence", 0.5)
					version.evidence_count = version_data.get("evidence_count", 0)
					version.contradictions = version_data.get("contradictions", [])
					self.definitions[concept].versions.append(version)
				
				self.definitions[concept].current_version = history_data.get("current_version", 0)
		except FileNotFoundError:
			logger.info(f"No definitions file found at {path}")


class SourceReliability:
	"""Track source credibility and accuracy."""
	
	def __init__(self, source_name: str):
		self.source_name = source_name
		self.correct_claims = 0
		self.incorrect_claims = 0
		self.total_claims = 0
		self.reliability_score = 0.5  # Start neutral
		self.last_updated = time.time()
	
	def record_correct(self):
		"""Record correct claim from this source."""
		self.correct_claims += 1
		self.total_claims += 1
		self._update_score()
	
	def record_incorrect(self):
		"""Record incorrect claim from this source."""
		self.incorrect_claims += 1
		self.total_claims += 1
		self._update_score()
	
	def _update_score(self):
		"""Update reliability score."""
		if self.total_claims > 0:
			self.reliability_score = self.correct_claims / self.total_claims
		self.last_updated = time.time()
	
	def get_score(self) -> float:
		"""Get current reliability score (0.0-1.0)."""
		return self.reliability_score
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			"source": self.source_name,
			"correct": self.correct_claims,
			"incorrect": self.incorrect_claims,
			"total": self.total_claims,
			"reliability_score": self.reliability_score,
			"last_updated": self.last_updated,
		}


class SourceReliabilityRegistry:
	"""Track reliability of all sources."""
	
	def __init__(self):
		self.sources: Dict[str, SourceReliability] = {}
	
	def record_claim(self, source: str, correct: bool):
		"""Record claim verification result."""
		source = str(source).strip().lower()
		
		if source not in self.sources:
			self.sources[source] = SourceReliability(source)
		
		if correct:
			self.sources[source].record_correct()
		else:
			self.sources[source].record_incorrect()
	
	def get_reliability(self, source: str) -> float:
		"""Get source reliability score."""
		source = str(source).strip().lower()
		if source in self.sources:
			return self.sources[source].get_score()
		return 0.5  # Default neutral if no history
	
	def get_top_reliable_sources(self, limit: int = 10) -> List[Dict[str, Any]]:
		"""Get most reliable sources."""
		sorted_sources = sorted(
			self.sources.values(),
			key=lambda s: s.reliability_score,
			reverse=True
		)
		return [s.to_dict() for s in sorted_sources[:limit]]
	
	def get_unreliable_sources(self, threshold: float = 0.3, limit: int = 10) -> List[Dict[str, Any]]:
		"""Get unreliable sources (below threshold)."""
		unreliable = [s for s in self.sources.values() if s.reliability_score < threshold and s.total_claims > 0]
		sorted_sources = sorted(unreliable, key=lambda s: s.reliability_score)
		return [s.to_dict() for s in sorted_sources[:limit]]
	
	def get_summary(self) -> Dict[str, Any]:
		"""Get summary of all sources."""
		return {
			"total_sources": len(self.sources),
			"top_reliable": self.get_top_reliable_sources(5),
			"unreliable": self.get_unreliable_sources(0.3, 5),
		}
	
	def save(self, path: str = "data/knowledge/source_reliability.json"):
		"""Save to file."""
		data = {
			source_name: source.to_dict()
			for source_name, source in self.sources.items()
		}
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
	
	def load(self, path: str = "data/knowledge/source_reliability.json"):
		"""Load from file."""
		try:
			with open(path, "r", encoding="utf-8") as f:
				data = json.load(f)
			
			for source_name, source_data in data.items():
				source = SourceReliability(source_name)
				source.correct_claims = source_data.get("correct", 0)
				source.incorrect_claims = source_data.get("incorrect", 0)
				source.total_claims = source_data.get("total", 0)
				source.reliability_score = source_data.get("reliability_score", 0.5)
				source.last_updated = source_data.get("last_updated", time.time())
				self.sources[source_name] = source
		except FileNotFoundError:
			logger.info(f"No source reliability file found at {path}")
