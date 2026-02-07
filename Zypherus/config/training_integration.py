"""
ACE Training Data Integration Module

Integrates the collected training dataset with the ACE core system.
Provides utilities for:
  - Loading training data into memory
  - Querying by category/type
  - Integration with reasoning engines
  - Metrics tracking and evaluation
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class TrainingItem:
    """Single training item with metadata."""
    narrative_id: str
    source: str
    collection_phase: str
    item_type: str
    
    problem: str
    proposed_solution: str
    proposed_rationale: str
    revised_solution: str
    final_explanation: str
    
    quality_score: float
    quality_tier: str
    
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative_id": self.narrative_id,
            "source": self.source,
            "collection_phase": self.collection_phase,
            "type": self.item_type,
            "problem": self.problem,
            "proposed_solution": self.proposed_solution,
            "proposed_rationale": self.proposed_rationale,
            "revised_solution": self.revised_solution,
            "final_explanation": self.final_explanation,
            "quality_score": self.quality_score,
            "quality_tier": self.quality_tier,
            "metadata": self.metadata,
        }
    
    def hash(self) -> str:
        """Generate hash for deduplication tracking."""
        content = f"{self.problem}{self.proposed_solution}{self.final_explanation}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class TrainingDataset:
    """In-memory training dataset with query capabilities."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.items: List[TrainingItem] = []
        self.by_phase: Dict[str, List[TrainingItem]] = {}
        self.by_source: Dict[str, List[TrainingItem]] = {}
        self.by_category: Dict[str, List[TrainingItem]] = {}
        self.by_quality_tier: Dict[str, List[TrainingItem]] = {}
        self.loaded_at = None
        
        if dataset_path:
            self.load(dataset_path)
    
    def load(self, dataset_path: Path) -> int:
        """Load training data from JSONL file."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        count = 0
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        item = self._parse_item(data)
                        self.items.append(item)
                        self._index_item(item)
                        count += 1
                    except Exception as e:
                        print(f"Warning: Failed to parse item: {e}")
        
        self.loaded_at = datetime.now()
        return count
    
    def _parse_item(self, data: Dict[str, Any]) -> TrainingItem:
        """Convert JSON to TrainingItem."""
        return TrainingItem(
            narrative_id=data.get("narrative_id", ""),
            source=data.get("source", "unknown"),
            collection_phase=data.get("collection_phase", "unknown"),
            item_type=data.get("type", "unknown"),
            
            problem=data.get("problem", ""),
            proposed_solution=data.get("proposed_solution", ""),
            proposed_rationale=data.get("proposed_rationale", ""),
            revised_solution=data.get("revised_solution", ""),
            final_explanation=data.get("final_explanation", ""),
            
            quality_score=float(data.get("quality_score", 0.0)),
            quality_tier=data.get("quality_tier", "bronze"),
            
            metadata={k: v for k, v in data.items() 
                     if k not in [
                        "narrative_id", "source", "collection_phase", "type",
                        "problem", "proposed_solution", "proposed_rationale",
                        "revised_solution", "final_explanation",
                        "quality_score", "quality_tier"
                     ]},
        )
    
    def _index_item(self, item: TrainingItem):
        """Add item to all index structures."""
        # By phase
        if item.collection_phase not in self.by_phase:
            self.by_phase[item.collection_phase] = []
        self.by_phase[item.collection_phase].append(item)
        
        # By source
        if item.source not in self.by_source:
            self.by_source[item.source] = []
        self.by_source[item.source].append(item)
        
        # By category
        category = item.metadata.get("category", "unknown")
        if category not in self.by_category:
            self.by_category[category] = []
        self.by_category[category].append(item)
        
        # By quality tier
        if item.quality_tier not in self.by_quality_tier:
            self.by_quality_tier[item.quality_tier] = []
        self.by_quality_tier[item.quality_tier].append(item)
    
    def query(self, 
              phase: Optional[str] = None,
              source: Optional[str] = None,
              category: Optional[str] = None,
              quality_tier: Optional[str] = None,
              min_quality_score: Optional[float] = None) -> List[TrainingItem]:
        """Query dataset with multiple filters."""
        results = self.items
        
        if phase:
            results = [i for i in results if i.collection_phase == phase]
        
        if source:
            results = [i for i in results if i.source == source]
        
        if category:
            results = [i for i in results if i.metadata.get("category") == category]
        
        if quality_tier:
            results = [i for i in results if i.quality_tier == quality_tier]
        
        if min_quality_score is not None:
            results = [i for i in results if i.quality_score >= min_quality_score]
        
        return results
    
    def statistics(self) -> Dict[str, Any]:
        """Generate dataset statistics."""
        return {
            "total_items": len(self.items),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "by_phase": {k: len(v) for k, v in self.by_phase.items()},
            "by_source": {k: len(v) for k, v in self.by_source.items()},
            "by_category": {k: len(v) for k, v in self.by_category.items()},
            "by_quality_tier": {k: len(v) for k, v in self.by_quality_tier.items()},
            "average_quality_score": sum(i.quality_score for i in self.items) / len(self.items) if self.items else 0,
        }


class TrainingDataIntegration:
    """Integration layer for Zypherus system."""
    
    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset = TrainingDataset(dataset_path)
        self.reasoning_examples: Dict[str, List[TrainingItem]] = {}
    
    def initialize_from_combined(self):
        """Load combined Phase 1+2+3 dataset."""
        combined_path = Path("training_data/combined/collected.jsonl")
        if not combined_path.exists():
            combined_path = Path("training_data/final_combined/collected.jsonl")
        
        if combined_path.exists():
            count = self.dataset.load(combined_path)
            print(f"Loaded {count} training items from combined dataset")
            return count
        else:
            raise FileNotFoundError("Combined dataset not found")
    
    def get_similar_examples(self, 
                           problem_keywords: List[str],
                           quality_threshold: float = 0.5) -> List[TrainingItem]:
        """Get training examples similar to a given problem."""
        # Simple keyword matching for demonstration
        matching_items = []
        
        for item in self.dataset.items:
            if item.quality_score >= quality_threshold:
                problem_text = (item.problem + " " + item.proposed_rationale).lower()
                if any(kw.lower() in problem_text for kw in problem_keywords):
                    matching_items.append(item)
        
        # Sort by quality score (descending)
        matching_items.sort(key=lambda x: x.quality_score, reverse=True)
        
        return matching_items[:5]
    
    def get_decision_examples(self, decision_type: str) -> List[TrainingItem]:
        """Get examples for specific decision type."""
        category_map = {
            "architecture": "architecture",
            "performance": "optimization",
            "scaling": "scaling",
            "security": "security",
            "reliability": "reliability",
            "refactoring": "refactoring",
        }
        
        category = category_map.get(decision_type.lower(), decision_type)
        return self.dataset.query(
            category=category,
            min_quality_score=0.5
        )
    
    def get_failure_lessons(self, domain: str) -> List[TrainingItem]:
        """Get failure patterns from Phase 2."""
        items = self.dataset.query(
            phase="phase2_postmortems_cves",
            min_quality_score=0.4
        )
        
        if domain:
            items = [i for i in items 
                    if domain.lower() in i.metadata.get("project", "").lower()]
        
        return items[:10]
    
    def export_for_training(self, output_path: Path) -> int:
        """Export dataset in format suitable for model training."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for item in self.dataset.items:
                # Export as training pair: (context, response)
                training_pair = {
                    "context": f"{item.problem}\n\nProposed: {item.proposed_solution}",
                    "response": item.final_explanation,
                    "metadata": item.metadata,
                }
                f.write(json.dumps(training_pair, ensure_ascii=False) + "\n")
                count += 1
        
        return count


def initialize_zypherus_training_integration():
    """Initialize and return integration object."""
    integration = TrainingDataIntegration()
    integration.initialize_from_combined()
    return integration


if __name__ == "__main__":
    print("ACE Training Data Integration Module")
    print("=" * 60)
    
    # Demo usage
    integration = initialize_zypherus_training_integration()
    
    stats = integration.dataset.statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  By phase: {stats['by_phase']}")
    print(f"  Average quality: {stats['average_quality_score']:.2f}")
    
    # Get examples
    examples = integration.get_similar_examples(["performance", "optimization"])
    print(f"\nFound {len(examples)} performance-related examples")
    
    # Get security lessons
    security = integration.get_decision_examples("security")
    print(f"Found {len(security)} security decision examples")
    
    # Get failure patterns
    failures = integration.get_failure_lessons("database")
    print(f"Found {len(failures)} database failure lessons")
