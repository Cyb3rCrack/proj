"""
C++ AI Code Quality Curator & Phased Training Manager

Implements:
- Quality > Quantity > Diversity principle
- Phase-gated curriculum
- Content tagging and filtering
- Poison detection
- Strategic prioritization
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

class Phase(Enum):
    PHASE_1_FOUNDATION = 1
    PHASE_2_EXPLAINED = 2
    PHASE_3_COMPLEXITY = 3
    PHASE_4_FAILURES = 4

class Quality(Enum):
    GOLD = "gold"       # Authoritative, well-maintained
    SILVER = "silver"   # Good, reliable, tested
    BRONZE = "bronze"   # Acceptable, verified works
    UNVETTED = "unvetted"

class ContentType(Enum):
    REFERENCE = "reference"
    TUTORIAL = "tutorial"
    STYLE_GUIDE = "style_guide"
    CONCEPT_WALKTHROUGH = "concept_walkthrough"
    DESIGN_PATTERN = "design_pattern"
    PRODUCTION = "production"
    COMPARISON = "comparison"
    BUGFIX = "bugfix"
    POSTMORTEM = "postmortem"
    CODE_REVIEW = "code_review"

class KnowledgeValue(Enum):
    EXEMPLAR = "exemplar"           # Model this code
    ANTIPATTERN = "antipattern"     # Don't do this
    CAUTIONARY = "cautionary"       # Interesting but dangerous
    REFERENCE = "reference"         # Look it up

class ContentMetadata:
    """Metadata for a training resource."""
    
    def __init__(
        self,
        title: str,
        source: str,
        phase: Phase,
        difficulty: str,
        domain: str,
        content_type: ContentType,
        quality: Quality,
        knowledge_value: KnowledgeValue,
        teaches: List[str],
        explanation: Optional[str] = None,
    ):
        self.title = title
        self.source = source
        self.phase = phase
        self.difficulty = difficulty
        self.domain = domain
        self.content_type = content_type
        self.quality = quality
        self.knowledge_value = knowledge_value
        self.teaches = teaches
        self.explanation = explanation
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "phase": self.phase.value,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "content_type": self.content_type.value,
            "quality": self.quality.value,
            "knowledge_value": self.knowledge_value.value,
            "teaches": self.teaches,
            "explanation": self.explanation,
        }


class QualityCurator:
    """Manages content quality and phased training."""
    
    # Map ingested sources to metadata (PHASE 1 priority)
    PHASE_1_CATALOG = {
        "cpp_learning:cppreference_containers": ContentMetadata(
            title="std::vector, std::map, std::set - Core Containers",
            source="https://en.cppreference.com/w/cpp/container",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="beginner",
            domain="stdlib",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["container_basics", "naming_conventions", "API_design"],
            explanation="Official reference for standard containers - study the design patterns",
        ),
        
        "cpp_learning:cppreference_memory": ContentMetadata(
            title="Smart Pointers & Memory Management",
            source="https://en.cppreference.com/w/cpp/memory",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="memory",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["ownership", "lifetime", "smart_pointers"],
            explanation="Authoritative reference on memory semantics - understand unique_ptr/shared_ptr ownership",
        ),
        
        "cpp_practical:cpp_raii": ContentMetadata(
            title="RAII Pattern - Resource Acquisition Is Initialization",
            source="https://en.cppreference.com/w/cpp/language/raii",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="patterns",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["resource_management", "constructor_destructor", "exception_safety"],
            explanation="The FUNDAMENTAL C++ pattern - this is not optional teaching",
        ),
        
        "cpp_practical:cpp_unique_ptr": ContentMetadata(
            title="unique_ptr - Exclusive Ownership Semantics",
            source="https://en.cppreference.com/w/cpp/memory/unique_ptr",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="memory",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["ownership_semantics", "move_semantics", "zero_cost_abstraction"],
            explanation="Zero-overhead abstraction over raw pointers - study this deeply",
        ),
        
        "cpp_practical:cpp_shared_ptr": ContentMetadata(
            title="shared_ptr - Shared Ownership & Lifecycle",
            source="https://en.cppreference.com/w/cpp/memory/shared_ptr",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="memory",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["shared_ownership", "reference_counting", "circular_references"],
            explanation="How shared ownership works - tradeoffs vs unique_ptr",
        ),
        
        "cpp_learning:cpp_classes": ContentMetadata(
            title="Classes - Design & Semantics",
            source="https://en.cppreference.com/w/cpp/language/class",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="principles",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["class_design", "encapsulation", "member_functions"],
            explanation="Foundation of object-oriented C++ - naming and organization",
        ),
        
        "cpp_learning:cpp_pointers": ContentMetadata(
            title="Pointers & References - Low-Level Mechanics",
            source="https://en.cppreference.com/w/cpp/language/pointer",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="beginner",
            domain="memory",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["pointer_semantics", "memory_addressing", "aliasing"],
            explanation="Foundational understanding of what pointers are",
        ),
        
        "cpp_learning:cpp_strings": ContentMetadata(
            title="std::string - Correct String Handling",
            source="https://en.cppreference.com/w/cpp/string",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="beginner",
            domain="stdlib",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["string_safety", "ownership", "performance"],
            explanation="How to handle strings correctly in modern C++",
        ),
        
        "cpp_practical:cpp_string_view": ContentMetadata(
            title="std::string_view - Non-Owning String References",
            source="https://en.cppreference.com/w/cpp/string/basic_string_view",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="stdlib",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["non_owning_references", "performance", "lifetime"],
            explanation="Modern pattern: use string_view for parameters, string for ownership",
        ),
        
        "cpp_learning:cpp_core_guidelines": ContentMetadata(
            title="C++ Core Guidelines - Best Practices",
            source="https://isocpp.org/guidelines/",
            phase=Phase.PHASE_1_FOUNDATION,
            difficulty="intermediate",
            domain="principles",
            content_type=ContentType.STYLE_GUIDE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["best_practices", "idioms", "design_principles"],
            explanation="The authoritative guidelines - WHY things should be done",
        ),
    }
    
    # PHASE 2: Code + Explanations (requires good source + explanation)
    PHASE_2_CATALOG = {
        "cpp_practical:cpp_lambdas": ContentMetadata(
            title="Lambda Expressions - Understanding Closures & Capture",
            source="https://en.cppreference.com/w/cpp/language/lambda",
            phase=Phase.PHASE_2_EXPLAINED,
            difficulty="intermediate",
            domain="patterns",
            content_type=ContentType.CONCEPT_WALKTHROUGH,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["lambda_syntax", "capture_semantics", "functional_programming"],
            explanation="Lambda = function + closure. Capture rules are precise; learn them deeply.",
        ),
        
        "cpp_practical:cpp_constexpr": ContentMetadata(
            title="constexpr - Compile-Time Computation",
            source="https://en.cppreference.com/w/cpp/language/constexpr",
            phase=Phase.PHASE_2_EXPLAINED,
            difficulty="intermediate",
            domain="principles",
            content_type=ContentType.CONCEPT_WALKTHROUGH,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["compile_time_computation", "zero_cost_abstraction", "optimization"],
            explanation="constexpr forces compile-time evaluation when possible - powerful for zero-cost",
        ),
        
        "cpp_practical:cpp_optional": ContentMetadata(
            title="std::optional - Type-Safe Nullable Values",
            source="https://en.cppreference.com/w/cpp/utility/optional",
            phase=Phase.PHASE_2_EXPLAINED,
            difficulty="intermediate",
            domain="patterns",
            content_type=ContentType.DESIGN_PATTERN,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["nullable_semantics", "type_safety", "absence_handling"],
            explanation="Replaces unsafe null pointers with compiler-enforced checking. Mandatory learning.",
        ),
        
        "cpp_practical:cpp_variant": ContentMetadata(
            title="std::variant - Type-Safe Unions",
            source="https://en.cppreference.com/w/cpp/utility/variant",
            phase=Phase.PHASE_2_EXPLAINED,
            difficulty="advanced",
            domain="patterns",
            content_type=ContentType.DESIGN_PATTERN,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["discriminated_unions", "type_safety", "pattern_matching"],
            explanation="Safe variant handling - understand visitor pattern and alternatives",
        ),
        
        "cpp_practical:cpp_copy_semantics": ContentMetadata(
            title="Copy Semantics - Constructors & Assignment",
            source="https://en.cppreference.com/w/cpp/language/copy_constructor",
            phase=Phase.PHASE_2_EXPLAINED,
            difficulty="intermediate",
            domain="principles",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["copy_constructor", "copy_assignment", "rule_of_five"],
            explanation="Rule of five: understand when/why to define copy/move operations",
        ),
    }
    
    # PHASE 3: Production-Grade Complexity
    PHASE_3_CATALOG = {
        "cpp_learning:cpp_threading": ContentMetadata(
            title="std::thread - Concurrency Basics",
            source="https://en.cppreference.com/w/cpp/thread",
            phase=Phase.PHASE_3_COMPLEXITY,
            difficulty="advanced",
            domain="concurrency",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["threading", "synchronization", "race_conditions"],
            explanation="Threads are hard. This is the reference - understand safety thoroughly",
        ),
        
        "cpp_learning:cpp_atomic": ContentMetadata(
            title="std::atomic - Lock-Free Synchronization",
            source="https://en.cppreference.com/w/cpp/atomic",
            phase=Phase.PHASE_3_COMPLEXITY,
            difficulty="advanced",
            domain="concurrency",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["memory_ordering", "lock_free", "atomic_operations"],
            explanation="Memory ordering is subtle. Learn before writing concurrent code.",
        ),
        
        "cpp_practical:cpp_memory_ordering": ContentMetadata(
            title="Memory Ordering - Concurrency Guarantees",
            source="https://en.cppreference.com/w/cpp/atomic/memory_order",
            phase=Phase.PHASE_3_COMPLEXITY,
            difficulty="expert",
            domain="concurrency",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["memory_model", "happens_before", "optimizations"],
            explanation="This is graduate-level C++. Essential for production code.",
        ),
        
        "cpp_learning:cpp_algorithms": ContentMetadata(
            title="Standard Algorithms - Complete Reference",
            source="https://en.cppreference.com/w/cpp/algorithm",
            phase=Phase.PHASE_3_COMPLEXITY,
            difficulty="intermediate",
            domain="stdlib",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.EXEMPLAR,
            teaches=["algorithm_library", "ranges", "generic_programming"],
            explanation="Don't write loops - use algorithms. Know what's available.",
        ),
    }
    
    # PHASE 4: Learning from Failures
    PHASE_4_CATALOG = {
        "cpp_learning:cpp_errors": ContentMetadata(
            title="Error Handling - Exceptions & Safety",
            source="https://en.cppreference.com/w/cpp/error",
            phase=Phase.PHASE_4_FAILURES,
            difficulty="intermediate",
            domain="principles",
            content_type=ContentType.REFERENCE,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.CAUTIONARY,
            teaches=["exception_safety", "defensive_coding", "error_propagation"],
            explanation="Errors will happen. Code defensively. Understand exception safety levels.",
        ),
        
        "cpp_practical:cpp_sfinae": ContentMetadata(
            title="SFINAE - Substitution Failure Is Not An Error",
            source="https://en.cppreference.com/w/cpp/language/sfinae",
            phase=Phase.PHASE_4_FAILURES,
            difficulty="expert",
            domain="patterns",
            content_type=ContentType.CONCEPT_WALKTHROUGH,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.CAUTIONARY,
            teaches=["template_constraints", "metaprogramming", "compiler_behavior"],
            explanation="Advanced: how templates behave under constraints. Tricky, but powerful.",
        ),
        
        "cpp_practical:cpp_template_specialization": ContentMetadata(
            title="Template Specialization - Focused Implementations",
            source="https://en.cppreference.com/w/cpp/language/template_specialization",
            phase=Phase.PHASE_4_FAILURES,
            difficulty="advanced",
            domain="patterns",
            content_type=ContentType.DESIGN_PATTERN,
            quality=Quality.GOLD,
            knowledge_value=KnowledgeValue.CAUTIONARY,
            teaches=["specialization", "template_metaprogramming", "compile_time"],
            explanation="Specialization can be abused. Learn discipline and clear intent.",
        ),
    }
    
    def __init__(self):
        self.catalog = {
            Phase.PHASE_1_FOUNDATION: self.PHASE_1_CATALOG,
            Phase.PHASE_2_EXPLAINED: self.PHASE_2_CATALOG,
            Phase.PHASE_3_COMPLEXITY: self.PHASE_3_CATALOG,
            Phase.PHASE_4_FAILURES: self.PHASE_4_CATALOG,
        }
        
        # Track what's been cleared for ingestion
        self.approved = set()
        self.skipped = set()
    
    def get_phase_materials(self, phase: Phase) -> Dict[str, ContentMetadata]:
        """Get all materials for a specific phase."""
        return self.catalog.get(phase, {})
    
    def is_poison(self, source: str) -> Tuple[bool, Optional[str]]:
        """Check if a source is poison content."""
        
        poison_keywords = [
            "ai-generated",
            "chatgpt",
            "copilot",
            "competit",  # competitive programming
            "exploit",
            "hack",
            "decompile",
            "junk",
            "abandoned",
            "deprecated",
        ]
        
        source_lower = source.lower()
        for keyword in poison_keywords:
            if keyword in source_lower:
                return True, f"Contains poison keyword: '{keyword}'"
        
        # Check for telltale signs
        if "stackoverflow" in source_lower and "raw" not in source_lower:
            return True, "Generic StackOverflow (not curated reference)"
        
        if "random github" in source_lower or "dump" in source_lower:
            return True, "Untrusted content dump"
        
        return False, None
    
    def print_phase_summary(self, phase: Phase):
        """Print summary of a training phase."""
        
        materials = self.get_phase_materials(phase)
        
        phase_names = {
            Phase.PHASE_1_FOUNDATION: "Foundation (Clean Fundamentals)",
            Phase.PHASE_2_EXPLAINED: "Explained (Code + Reasoning)",
            Phase.PHASE_3_COMPLEXITY: "Complexity (Production Reality)",
            Phase.PHASE_4_FAILURES: "Failures (Defensive Learning)",
        }
        
        print("\n" + "=" * 80)
        print(f"📚 PHASE {phase.value}: {phase_names[phase]}")
        print("=" * 80)
        
        for source_id, metadata in materials.items():
            quality_emoji = {
                Quality.GOLD: "⭐",
                Quality.SILVER: "✨",
                Quality.BRONZE: "✓",
                Quality.UNVETTED: "?",
            }[metadata.quality]
            
            value_emoji = {
                KnowledgeValue.EXEMPLAR: "📖",
                KnowledgeValue.ANTIPATTERN: "⚠️",
                KnowledgeValue.CAUTIONARY: "🔶",
                KnowledgeValue.REFERENCE: "📋",
            }[metadata.knowledge_value]
            
            print(f"\n{quality_emoji} {value_emoji} {metadata.title}")
            print(f"   Difficulty: {metadata.difficulty:15} | Domain: {metadata.domain}")
            print(f"   Teaches: {', '.join(metadata.teaches)}")
            if metadata.explanation:
                print(f"   → {metadata.explanation}")
    
    def suggest_next_steps(self):
        """Suggest what to ingest next based on curriculum."""
        
        print("\n" + "=" * 80)
        print("🎯 CURRICULUM - SUGGESTED PRIORITY ORDER")
        print("=" * 80)
        
        print("""
BURN IN THIS ORDER:

Week 1-2: PHASE 1 FOUNDATION
═════════════════════════════════════════════════════════════
Why: Teach idiomatic style and correct fundamentals
What: ONLY gold-quality references
How:
  1. Start with containers (vector, map, set)
  2. Then memory management (pointers, smart pointers)
  3. Then classes and RAII
  4. End with strings and Core Guidelines

Success: AI can explain WHY code is idiomatic


Week 3-4: PHASE 2 EXPLANATIONS  
═════════════════════════════════════════════════════════════
Why: Teach reasoning (not just syntax)
What: Code paired with deep explanations
How:
  1. Lambdas + their capture semantics
  2. constexpr and compile-time reasoning
  3. std::optional (why not nullptrs?)
  4. Copy vs move semantics

Success: AI explains tradeoffs and design decisions


Week 5-6: PHASE 3 COMPLEXITY
═════════════════════════════════════════════════════════════
Why: Handle real systems
What: Production-grade code only
How:
  1. Threading basics
  2. Atomics and memory ordering
  3. Large algorithm reference
  4. Pattern studies from real codebases

Success: AI handles multi-threaded concerns


Week 7-8: PHASE 4 FAILURES
═════════════════════════════════════════════════════════════
Why: Develop defensive instincts
What: Real bugs + fixes + lessons
How:
  1. Exception safety levels
  2. SFINAE and template pitfalls
  3. Specialization discipline
  4. Bug analysis case studies

Success: AI stops hallucinating confidently wrong answers
        """)


def main():
    """Main curator interface."""
    
    curator = QualityCurator()
    
    print("=" * 80)
    print("🎓 C++ AI TRAINING CURATOR")
    print("Golden Rule: Quality > Quantity > Diversity")
    print("=" * 80)
    
    print("\n📖 LOADING CURRICULUM...")
    
    # Show each phase
    for phase in Phase:
        curator.print_phase_summary(phase)
    
    # Show curriculum structure
    curator.suggest_next_steps()
    
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHT")
    print("=" * 80)
    print("""
What you've already ingested (72 resources) is mostly PHASE 1 & 2 material.
That's actually perfect for foundation building.

Now the strategy:
  1. CURATE: Filter those 72 for only highest quality
  2. TAG: Apply metadata so AI learns structured
  3. PHASE: Serve to AI in curriculum order
  4. VERIFY: Test AI understanding after each phase
  5. FAIL: Study failures to prevent hallucination

You will end up with ~30-40 CURATED resources, not 72 random ones.
The smaller, cleaner set will outperform the larger junk pile EVERY TIME.

This is the difference between:
  • AI that can recite facts (bad)
  • AI that understands reasoning (good)
    """)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
