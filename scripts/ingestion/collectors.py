"""
Concrete Implementation: Ingestion Collectors

Maps the 10 principles to actual data collection scripts.
Each collector focuses on ONE principle.
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
import requests
from pathlib import Path
import json


class AbstractCollector(ABC):
    """Base class for all collectors."""
    
    principle: str
    sources_found: int = 0
    quality_score: float = 0.0
    
    @abstractmethod
    def collect(self) -> List[Dict[str, Any]]:
        """Collect from source."""
        pass
    
    @abstractmethod
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Check quality gates."""
        pass


# PRINCIPLE 1-2: CODE REVIEWS
class GitHubPRCollector(AbstractCollector):
    """Collect code reviews from GitHub PRs with explanation discussions."""
    
    principle = "Code reviews + human explanation"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Collect detailed code review discussions.
        Focus: Explanation length, reviewer rationale, author response
        """
        reviews = []
        
        repos_to_scan = [
            "torvalds/linux",
            "llvm/llvm-project",
            "postgres/postgres",
            "kubernetes/kubernetes",
            "golang/go",
            "rust-lang/rust",
        ]
        
        for repo in repos_to_scan:
            # Would use GitHub API here
            # GET /repos/{repo}/pulls?state=closed&per_page=100
            # For each PR:
            #   - Check review comments
            #   - Filter for substantive feedback (length > 100)
            #   - Extract code blocks
            #   - Parse author response
            pass
        
        return reviews
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Review must have substantial explanation."""
        total_review_length = sum(
            len(c.get("body", "")) for c in item.get("review_comments", [])
        )
        return total_review_length > 200  # Meaningful discussion


# PRINCIPLE 3: FAILURES & POSTMORTEMS
class PostmortemCollector(AbstractCollector):
    """Collect postmortems with failure analysis."""
    
    principle = "Failures, bugs, postmortems with explanations"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Collect from postmortem databases and incident reports.
        Focus: Root cause analysis, debugging process, prevention
        """
        postmortems = []
        
        sources = [
            "https://github.com/danluu/post-mortems",  # Aggregated
            "https://kubernetes.io/blog/",  # Incident reports
            "https://stripe.com/blog/engineering",  # Engineering
            "https://opensource.googleblog.com/",  # Google
        ]
        
        # Would scrape and parse each source
        # Extract: What happened → Root cause → Fix → Prevention
        
        return postmortems
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Postmortem must have all sections + explanation."""
        required = ["what_happened", "root_cause", "fix", "prevention"]
        has_all = all(item.get(k, "") for k in required)
        explanation_length = len(item.get("explanation", ""))
        return has_all and explanation_length > 200


# PRINCIPLE 4: REAL CONSTRAINTS
class DesignDocCollector(AbstractCollector):
    """Collect design documents showing constraints and tradeoffs."""
    
    principle = "Real constraints force intelligent code"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Parse design documents, architecture decision records, RFCs.
        Focus: Constraints, performance requirements, scalability limits
        """
        design_docs = []
        
        # Specific repos with great design docs
        sources = {
            "redis/redis": "Extreme performance focus",
            "sqlite/sqlite": "Zero-allocation culture",
            "torvalds/linux": "Constraint-driven design",
            "google/abseil-cpp": "Performance requirements",
        }
        
        # Would parse:
        # - /doc/ folders for design docs
        # - RFC files with rationale
        # - Commit messages with "constraint" keywords
        # - Performance analysis documentation
        
        return design_docs
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Design doc must mention constraints explicitly."""
        constraints = item.get("constraints", [])
        return len(constraints) > 0


# PRINCIPLE 8: EVOLUTION OVER TIME
class GitHistoryCollector(AbstractCollector):
    """Analyze git history to show evolution and refactoring."""
    
    principle = "Evolution over time teaches abstraction lifecycle"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Parse git log, refactoring history, deprecation patterns.
        Focus: How code evolved, why abstractions rotted
        """
        histories = []
        
        # Example repos with good commit history
        repos = [
            "postgres/postgres",  # 25 years of evolution
            "kubernetes/kubernetes",  # Rapid evolution with care
            "numpy/numpy",  # Python evolution patterns
        ]
        
        # Would use: git log --all --decorate --oneline --graph
        # Parse commit messages for "Refactor", "Fix", "Deprecate"
        # Track API changes over time
        
        return histories
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """History must show multiple refactors and evolution."""
        commits = item.get("commits", [])
        refactor_pattern_count = sum(
            1 for c in commits 
            if any(k in c.lower() for k in ["refactor", "deprecat", "evolv"])
        )
        return refactor_pattern_count >= 3


# PRINCIPLE 9: SECURITY & CORRECTNESS
class SecurityLessonCollector(AbstractCollector):
    """Collect security vulnerabilities with fixes and explanations."""
    
    principle = "Security and correctness boundaries"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Parse CVE databases, OWASP, CWE, security blog posts.
        Focus: Vulnerability → example → exploit → fix
        """
        lessons = []
        
        # Sources
        sources = [
            "https://nvd.nist.gov/vuln",  # CVE database
            "https://owasp.org/Top10/",  # Common weaknesses
            "https://cwe.mitre.org/",  # CWE entries
            "https://msrc.microsoft.com/",  # Microsoft advisories
        ]
        
        # Would parse:
        # - CVE entries with technical description
        # - OWASP examples with code snippets
        # - Security blog posts with: vulnerable code + fixed code
        
        return lessons
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Must have vulnerable + fixed code and explanation."""
        has_codes = bool(item.get("vulnerable_code") and item.get("fixed_code"))
        has_explain = len(item.get("explanation", "")) > 100
        return bool(has_codes and has_explain)


# PRINCIPLE 7: LANGUAGE-SPECIFIC IDIOMS
class IdiomsCollector(AbstractCollector):
    """Collect language-specific idioms and patterns."""
    
    principle = "Language-specific taste and idioms"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Language design docs, style guides, "Effective X" content.
        Focus: What feels right in this language
        """
        idioms = []
        
        language_sources = {
            "c++": ["isocpp.org", "github.com/isocpp/CppCoreGuidelines"],
            "python": ["pep8.org", "google.github.io/styleguide/pyguide.html"],
            "rust": ["rust-lang.org/api-guidelines"],
            "go": ["golang.org/effective_go.html"],
        }
        
        # Would parse each language's style guide
        # Extract: "Do this", "Don't do this", and WHY
        # Show idiomatic vs non-idiomatic examples
        
        return idioms
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Idiom must show correct vs incorrect pattern with reason."""
        return bool(
            item.get("idiomatic_code") and 
            item.get("non_idiomatic_code") and
            len(item.get("why", "")) > 100
        )


# PRINCIPLE 10: META-SKILLS
class MetaSkillCollector(AbstractCollector):
    """Collect meta-skills: asking questions, saying "it depends"."""
    
    principle = "Meta-skills - knowing what we don't know"
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Blog posts about decisions, ambiguity, tradeoffs.
        Focus: Understanding when code should ask questions
        """
        skills = []
        
        sources = [
            "martinfowler.com",
            "highscalability.com",
            "stripe.com/blog/engineering",
            "aws.amazon.com/blogs/architecture/",
        ]
        
        # Would parse blog posts for:
        # - "We chose X over Y because..."
        # - "This depends on..."
        # - "We didn't have enough information"
        # - Decision matrices and tradeoffs
        
        return skills
    
    def validate_quality(self, item: Dict[str, Any]) -> bool:
        """Meta-skill must show ambiguity and decision process."""
        examples = item.get("examples", [])
        has_decision = "depends" in item.get("content", "").lower()
        return len(examples) > 0 and has_decision


# MASTER ORCHESTRATOR
class FocusedIngestController:
    """Coordinates collection across all principles."""
    
    def __init__(self):
        self.collectors: Dict[str, AbstractCollector] = {
            "code_reviews": GitHubPRCollector(),
            "postmortems": PostmortemCollector(),
            "constraints": DesignDocCollector(),
            "history": GitHistoryCollector(),
            "security": SecurityLessonCollector(),
            "idioms": IdiomsCollector(),
            "meta_skills": MetaSkillCollector(),
        }
        self.all_items = []
    
    def collect_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all collectors."""
        results = {}
        
        for name, collector in self.collectors.items():
            print(f"Collecting: {collector.principle}...")
            items = collector.collect()
            validated = [i for i in items if collector.validate_quality(i)]
            results[name] = validated
            print(f"  ✓ {len(validated)} items passed quality gates")
        
        return results
    
    def export(self, output_path: Path) -> None:
        """Export to training format."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "principles_covered": 7,  # Out of 10
            "note": "Should also collect: principle_5, principle_6, principle_2_rfc",
            "collectors": {
                name: {
                    "principle": c.principle,
                    "items_collected": len(items)
                }
                for name, (c, items) in zip(
                    self.collectors.keys(),
                    [(c, []) for c in self.collectors.values()]
                )
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


# NEXT STEPS FOR REAL IMPLEMENTATION
"""
Each collector needs actual implementation of:
1. Authentication (GitHub token, etc.)
2. HTTP requests to sources
3. HTML/JSON parsing
4. Text extraction (code blocks, explanations)
5. Quality scoring
6. Deduplication

Suggested order of implementation:
1. Start with GitHub PR collector (many high-quality reviews)
2. Add postmortem collector (early warning signals)
3. Add security collector (CVE databases are well-structured)
4. Add git history (already have local data)
5. Add design docs (parse markdown from repos)

Then evaluate coverage vs 10 principles.
If missing:
- Principle 5 (idiomatic patterns): Need language-specific code analysis
- Principle 6 (multi-layer): Need architecture visualization tools
- Principle 2 (RFCs): Need RFC database scraper
"""


if __name__ == "__main__":
    controller = FocusedIngestController()
    print("Available collectors:")
    for name, collector in controller.collectors.items():
        print(f"  - {name}: {collector.principle}")
