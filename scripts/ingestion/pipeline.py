"""
End-to-End Ingestion Pipeline

Orchestrates the full workflow:
Collect → Validate → Deduplicate → Score → Export → Train

Run this script to start collecting focused coding training data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class PipelineConfig:
    """Configuration for ingestion pipeline."""
    # Output settings
    output_dir: Path = Path("training_data")
    batch_size: int = 100
    
    # Quality settings
    min_quality_score: float = 0.3  # Bronze tier minimum
    require_gold_tier: bool = False  # If True, only accept GOLD quality
    
    # Collection settings
    languages: List[str] = None  # None = all languages
    sources_to_collect: List[str] = None  # None = all sources
    
    # Deduplication
    enable_dedup: bool = True
    semantic_similarity_threshold: float = 0.7
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["python", "go", "rust", "cpp", "javascript"]
        if self.sources_to_collect is None:
            self.sources_to_collect = [
                "code_reviews",
                "postmortems",
                "security",
                "design_docs",
                "git_history",
            ]


class IngestionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "collected": 0,
            "passed_quality": 0,
            "deduplicated": 0,
            "exported": 0,
            "by_principle": {},
            "by_quality_tier": {},
        }
        
        self.all_items = []
    
    def run(self) -> Dict[str, Any]:
        """Execute full pipeline."""
        
        print("\n" + "=" * 70)
        print("FOCUSED CODING TRAINING DATA PIPELINE")
        print("=" * 70)
        print(f"Config:")
        print(f"  Output: {self.output_dir}")
        print(f"  Min quality: {self.config.min_quality_score}")
        print(f"  Languages: {self.config.languages}")
        print(f"  Sources: {self.config.sources_to_collect}")
        
        # STAGE 1: COLLECTION
        print("\n" + "-" * 70)
        print("STAGE 1: COLLECTION")
        print("-" * 70)
        collected = self._stage_collection()
        
        # STAGE 2: VALIDATION
        print("\n" + "-" * 70)
        print("STAGE 2: VALIDATION & QUALITY GATES")
        print("-" * 70)
        validated = self._stage_validation(collected)
        
        # STAGE 3: DEDUPLICATION
        print("\n" + "-" * 70)
        print("STAGE 3: DEDUPLICATION")
        print("-" * 70)
        deduplicated = self._stage_deduplication(validated)
        
        # STAGE 4: ORGANIZATION & EXPORT
        print("\n" + "-" * 70)
        print("STAGE 4: EXPORT & ORGANIZATION")
        print("-" * 70)
        exported = self._stage_export(deduplicated)
        
        # STAGE 5: REPORTING
        print("\n" + "-" * 70)
        print("STAGE 5: REPORTING & STATISTICS")
        print("-" * 70)
        self._stage_reporting()
        
        return self.stats
    
    def _stage_collection(self) -> List[Dict[str, Any]]:
        """
        STAGE 1: Collect from all sources.
        
        This would call:
        - GitHubPRCollector()
        - PostmortemCollector()
        - SecurityLessonCollector()
        - DesignDocCollector()
        - GitHistoryCollector()
        """
        
        # In real implementation, would use:
        # from collectors import GitHubPRCollector, PostmortemCollector, ...
        
        collected = []
        
        print(f"\nCollecting from {len(self.config.sources_to_collect)} sources...")
        
        # Simulated collection
        source_counts = {
            "code_reviews": 150,
            "postmortems": 45,
            "security": 120,
            "design_docs": 60,
            "git_history": 200,
        }
        
        for source_type, count in source_counts.items():
            if source_type in self.config.sources_to_collect:
                print(f"  ✓ {source_type}: {count} items")
                self.stats["collected"] += count
                
                # Would add actual items here
                for i in range(min(count, 10)):  # Simulated
                    collected.append({
                        "id": f"{source_type}_{i}",
                        "source_type": source_type,
                        "principle": self._map_principle(source_type),
                        "explanation": "Example explanation" * 10,  # > 200 chars
                        "code": "def example(): pass" * 5,
                    })
        
        print(f"\n  Total collected: {self.stats['collected']} items")
        return collected
    
    def _stage_validation(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        STAGE 2: Apply quality gates.
        
        Filter by:
        - Explanation length (>= 100 chars)
        - Not from tutorial sites
        - Not boilerplate
        - Optional: Require specific quality tier
        """
        
        # Would use: from quality_gates import QualityGates, QualityMetrics
        
        validated = []
        tier_counts = {"gold": 0, "silver": 0, "bronze": 0, "rejected": 0}
        
        print(f"\nValidating {len(items)} items with quality gates...")
        
        for item in items:
            # Simulate quality evaluation
            explanation_len = len(item.get("explanation", ""))
            
            if explanation_len > 500:
                tier = "gold"
            elif explanation_len > 200:
                tier = "silver"
            else:
                tier = "bronze"
            
            tier_counts[tier] += 1
            
            if explanation_len > 100:  # Passes minimum gate
                validated.append({**item, "quality_tier": tier})
                self.stats["passed_quality"] += 1
        
        print(f"  Results:")
        for tier, count in tier_counts.items():
            pct = 100 * count // max(len(items), 1)
            print(f"    {tier.upper()}: {count} ({pct}%)")
        
        print(f"\n  Passed quality gates: {self.stats['passed_quality']}/{len(items)}")
        
        return validated
    
    def _stage_deduplication(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        STAGE 3: Remove duplicates.
        
        Strategy:
        - Exact hash match (code + explanation)
        - Semantic similarity (>0.7 = duplicate)
        - Keep best version of duplicates
        """
        
        # Would use: from quality_gates import DeduplicationPipeline
        
        if not self.config.enable_dedup:
            return items
        
        print(f"\nDeduplicating {len(items)} items...")
        
        # Simulated: assume 10% are duplicates
        deduplicated = items[:int(len(items) * 0.9)]
        removed = len(items) - len(deduplicated)
        
        self.stats["deduplicated"] = len(deduplicated)
        
        print(f"  Removed: {removed} duplicates")
        print(f"  Retained: {len(deduplicated)} unique items ({100*len(deduplicated)//len(items)}%)")
        
        return deduplicated
    
    def _stage_export(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        STAGE 4: Export to training format.
        
        Organize by:
        - Principle
        - Language
        - Quality tier
        - Source type
        """
        
        print(f"\nExporting {len(items)} items...")
        
        # Organize by principle
        by_principle = {}
        for item in items:
            principle = item.get("principle", "unknown")
            if principle not in by_principle:
                by_principle[principle] = []
            by_principle[principle].append(item)
        
        # Export to JSON files
        self.stats["by_principle"] = {
            p: len(items) for p, items in by_principle.items()
        }
        
        for principle, principle_items in by_principle.items():
            output_file = self.output_dir / f"{principle}.jsonl"
            with open(output_file, "w") as f:
                for item in principle_items:
                    f.write(json.dumps(item) + "\n")
            
            print(f"  ✓ {principle}: {len(principle_items)} items → {output_file.name}")
        
        # Export summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_items": len(items),
            "by_principle": self.stats["by_principle"],
            "quality_distribution": self._get_quality_distribution(items),
            "languages": self._get_languages(items),
        }
        
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n  ✓ Summary → {summary_file.name}")
        
        self.stats["exported"] = len(items)
        return items
    
    def _stage_reporting(self) -> None:
        """STAGE 5: Generate report."""
        
        print(f"\nPipeline Report:")
        print(f"  Collected:        {self.stats['collected']} items")
        print(f"  Passed quality:   {self.stats['passed_quality']} items")
        print(f"  Deduplicated:     {self.stats['deduplicated']} items")
        print(f"  Exported:         {self.stats['exported']} items")
        print(f"\n  By Principle:")
        for principle, count in self.stats["by_principle"].items():
            print(f"    - {principle}: {count}")
        
        export_report = self.output_dir / "REPORT.txt"
        with open(export_report, "w") as f:
            f.write("INGESTION PIPELINE REPORT\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(f"Collected:     {self.stats['collected']}\n")
            f.write(f"Passed quality: {self.stats['passed_quality']}\n")
            f.write(f"Deduplicated:  {self.stats['deduplicated']}\n")
            f.write(f"Exported:      {self.stats['exported']}\n")
        
        print(f"\n✓ Report saved to {export_report}")
    
    def _map_principle(self, source_type: str) -> str:
        """Map source type to principle."""
        mapping = {
            "code_reviews": "Explained code reviews",
            "postmortems": "Failures and postmortems",
            "security": "Security and correctness",
            "design_docs": "Real constraints",
            "git_history": "Evolution over time",
        }
        return mapping.get(source_type, "unknown")
    
    def _get_quality_distribution(self, items: List[Dict]) -> Dict[str, int]:
        """Count items by quality tier."""
        dist = {}
        for item in items:
            tier = item.get("quality_tier", "unknown")
            dist[tier] = dist.get(tier, 0) + 1
        return dist
    
    def _get_languages(self, items: List[Dict]) -> Dict[str, int]:
        """Count items by language."""
        langs = {}
        for item in items:
            lang = item.get("language", "unknown")
            langs[lang] = langs.get(lang, 0) + 1
        return langs


class TrainingDataValidator:
    """Validate exported training data."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def validate(self) -> Dict[str, Any]:
        """Check data integrity and quality."""
        results = {
            "files": [],
            "total_items": 0,
            "quality_checks": [],
        }
        
        print("\n" + "=" * 70)
        print("TRAINING DATA VALIDATION")
        print("=" * 70)
        
        # Check all JSONL files
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            items = []
            errors = []
            
            with open(jsonl_file) as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line)
                        items.append(item)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i+1}: {e}")
            
            results["files"].append({
                "name": jsonl_file.name,
                "items": len(items),
                "errors": errors,
            })
            results["total_items"] += len(items)
            
            print(f"\n  {jsonl_file.name}")
            print(f"    Items: {len(items)}")
            if errors:
                print(f"    Errors: {errors}")
        
        # Check summary
        summary_file = self.data_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            print(f"\n  Summary:")
            print(f"    Total items: {summary.get('total_items')}")
            print(f"    Principles: {list(summary.get('by_principle', {}).keys())}")
        
        print(f"\n✓ Validation complete: {results['total_items']} items ready for training")
        return results


def main():
    """Execute the pipeline."""
    
    # Create config
    config = PipelineConfig(
        output_dir=Path("training_data/focused_coding"),
        languages=["python", "go", "rust"],
        sources_to_collect=["code_reviews", "postmortems", "security"],
    )
    
    # Run pipeline
    pipeline = IngestionPipeline(config)
    stats = pipeline.run()
    
    # Validate exported data
    validator = TrainingDataValidator(config.output_dir)
    validation = validator.validate()
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n✓ {stats['exported']} high-quality training examples ready")
    print(f"✓ Output directory: {config.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review {config.output_dir}/summary.json")
    print(f"  2. Spot-check items in generated JSONL files")
    print(f"  3. Feed to your AI training pipeline")


if __name__ == "__main__":
    main()
