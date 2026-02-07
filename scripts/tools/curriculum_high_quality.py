"""
C++ AI Training Curriculum

Implements the Golden Rule:
Code Quality > Code Quantity > Code Diversity

Phased approach to building AI reasoning, not just syntax mimicry.
"""

# ==============================================================================
# PHASE 1: FOUNDATION (Clean, Boring, Correct Code)
# ==============================================================================
# Goal: Teach taste and fundamentals
# Duration: Weeks 1-2
# Quality filter: STRICT

PHASE_1_FOUNDATION = {
    "name": "Clean Fundamentals & Idiomatic Code",
    "goal": "Establish correct mental models and coding style",
    "duration_weeks": 2,
    "priority": "CRITICAL",
    
    "topics": [
        {
            "name": "C++ Standard Library - Core",
            "source": "https://en.cppreference.com/w/cpp/container/vector",
            "type": "reference",
            "quality": "gold",  # Authoritative, well-maintained
            "teaches": ["naming_conventions", "API_design", "error_handling"],
            "difficulty": "beginner",
            "domain": "stdlib",
        },
        {
            "name": "RAII Pattern - Canonical Form",
            "source": "https://en.cppreference.com/w/cpp/language/raii",
            "type": "reference",
            "quality": "gold",
            "teaches": ["resource_management", "exception_safety", "constructor_destructor"],
            "difficulty": "intermediate",
            "domain": "patterns",
        },
        {
            "name": "Smart Pointers - Design",
            "source": "https://en.cppreference.com/w/cpp/memory/unique_ptr",
            "type": "reference",
            "quality": "gold",
            "teaches": ["ownership_semantics", "lifetime_management", "move_semantics"],
            "difficulty": "intermediate",
            "domain": "memory",
        },
        {
            "name": "C++ Core Guidelines (Initiation)",
            "source": "https://isocpp.org/guidelines/",
            "type": "style_guide",
            "quality": "gold",
            "teaches": ["best_practices", "design_principles", "idioms"],
            "difficulty": "intermediate",
            "domain": "principles",
        },
    ],
    
    "anti_patterns": [
        "❌ No StackOverflow code snippets (unreliable)",
        "❌ No random GitHub scrapes (quality unknown)",
        "❌ No obfuscated code (teaches bad habits)",
        "❌ No competitive programming golf (not idiomatic)",
    ],
    
    "success_metrics": [
        "AI correctly explains RAII lifecycle",
        "AI recognizes idiomatic vs non-idiomatic naming",
        "AI explains constructor/destructor responsibilities",
        "AI understands ownership vs observation",
    ],
}

# ==============================================================================
# PHASE 2: PAIRED EXPLANATIONS (Code + Reasoning)
# ==============================================================================
# Goal: Teach WHY, not just HOW
# Duration: Weeks 3-4
# Quality filter: STRICT (must have explanations)

PHASE_2_EXPLAINED = {
    "name": "Code + Explanations (Reasoning Training)",
    "goal": "Build architectural judgment and problem-solving",
    "duration_weeks": 2,
    "priority": "CRITICAL",
    
    "ideal_units": [
        {
            "title": "Building a Type-Safe Optional (explained)",
            "problem": "How do you handle 'value might not exist' safely?",
            "design_reasoning": "Why not just use pointers or exceptions?",
            "code_source": "std::optional implementation and usage",
            "explanation": "Optional replaces unsafe pointer nullability with compiler-enforced checking",
            "type": "tutorial",
            "quality": "gold",
            "teaches": ["type_safety", "design_intent", "error_handling"],
        },
        {
            "title": "Move Semantics - Why It Matters",
            "problem": "Returning large objects efficiently from functions",
            "design_reasoning": "Copy is expensive; move is cheap; compiler should choose",
            "code_source": "Move constructor/assignment operator articles",
            "explanation": "Move semantics turn O(n) operations into O(1) through ownership transfer",
            "type": "concept_walkthrough",
            "quality": "gold",
            "teaches": ["performance", "semantics", "efficiency"],
        },
        {
            "title": "Exception Safety Guarantees",
            "problem": "How do you write code that doesn't leak in error cases?",
            "design_reasoning": "Four levels: no-throw, strong, basic, none",
            "code_source": "Examples of each guarantee level",
            "explanation": "Choose guarantee based on context; document it",
            "type": "design_pattern",
            "quality": "gold",
            "teaches": ["error_handling", "design_contracts", "robustness"],
        },
        {
            "title": "When to Use std::string vs std::string_view",
            "problem": "Ownership vs observation - when do you need each?",
            "design_reasoning": "string owns; string_view observes - use the right one",
            "code_source": "Comparison with real usage examples",
            "explanation": "Choosing wrong causes unnecessary allocations or dangling references",
            "type": "comparison",
            "quality": "gold",
            "teaches": ["lifetime", "ownership", "performance"],
        },
    ],
    
    "sources": [
        "Bjarne Stroustrup's C++ design articles",
        "Scott Meyers (Effective C++) blog/articles",
        "CppCon talks + articles (high quality, explained)",
        "Official standard library documentation + rationale",
        "'Build X from scratch' articles (with explanations)",
    ],
    
    "success_metrics": [
        "AI explains tradeoffs, not just features",
        "AI can explain WHY one design beats another",
        "AI predicts consequences of design choices",
        "AI suggests alternatives with reasoning",
    ],
}

# ==============================================================================
# PHASE 3: CONTROLLED COMPLEXITY
# ==============================================================================
# Goal: Handle real systems while maintaining quality
# Duration: Weeks 5-6
# Quality filter: STRICT (production-grade only)

PHASE_3_COMPLEXITY = {
    "name": "Scaled, Production-Grade Complexity",
    "goal": "Build judgment on large, real systems",
    "duration_weeks": 2,
    "priority": "HIGH",
    
    "introduce": [
        "Multi-file projects (header/implementation separation)",
        "Larger classes with complex state management",
        "Async and callback patterns",
        "Thread-safe code (mutexes, atomics)",
        "Configuration and dependency management",
        "Memory-pooled allocations",
    ],
    
    "quality_filters": [
        "✅ Active projects (< 1 year since last commit)",
        "✅ Well-tested (tests exist, not ornamental)",
        "✅ Production code (real applications, not toys)",
        "✅ Clear documentation (architecture explained)",
        "✅ Code review culture (PRs with feedback)",
    ],
    
    "anti_patterns": [
        "❌ Abandoned projects",
        "❌ Copy-paste code",
        "❌ StackOverflow Frankenstein code",
        "❌ 'Clever' one-liners without context",
        "❌ Hobby projects without testing",
    ],
    
    "examples": [
        {
            "name": "ThreadPool Implementation",
            "source": "Well-maintained library (e.g., Asio, Boost)",
            "difficulty": "advanced",
            "domain": "concurrency",
            "teaches": ["thread_management", "synchronization", "state_machines"],
            "type": "production",
        },
        {
            "name": "Connection Pooling",
            "source": "Database client library code",
            "difficulty": "advanced",
            "domain": "systems",
            "teaches": ["resource_lifecycle", "reuse_patterns", "fairness"],
            "type": "production",
        },
        {
            "name": "Configuration System",
            "source": "Well-designed config lib",
            "difficulty": "intermediate",
            "domain": "architecture",
            "teaches": ["composability", "validation", "error_reporting"],
            "type": "production",
        },
    ],
    
    "success_metrics": [
        "AI handles multi-file concerns",
        "AI suggests appropriate synchronization",
        "AI spots resource management bugs",
        "AI understands architectural patterns",
    ],
}

# ==============================================================================
# PHASE 4: FAILURE CASES (The Sharp Edge)
# ==============================================================================
# Goal: Learn defensive thinking and bug detection
# Duration: Weeks 7-8
# Quality filter: STRICT (real failures with lessons)

PHASE_4_FAILURES = {
    "name": "Learning from Failures & Missteps",
    "goal": "Develop defensive instincts and bug detection",
    "duration_weeks": 2,
    "priority": "HIGH",
    
    "training_materials": [
        {
            "type": "production_bug_fix",
            "description": "Real bug with before/after code",
            "example": "Memory leak from exception in constructor",
            "teaches": ["defensive_coding", "exception_safety", "edge_cases"],
        },
        {
            "type": "bad_pr_review",
            "description": "PR with issues + reviewer feedback",
            "example": "Race condition in double-checked lock",
            "teaches": ["concurrency_issues", "memory_ordering", "code_review"],
        },
        {
            "type": "outage_postmortem",
            "description": "'This caused production outage' writeups",
            "example": "Infinite loop in error handler triggered at scale",
            "teaches": ["system_thinking", "failure_modes", "testing"],
        },
        {
            "type": "refactor_analysis",
            "description": "Before/after comparisons of refactors",
            "example": "Unsafe cast → type-safe wrapper",
            "teaches": ["code_smell", "maintainability", "safety"],
        },
    ],
    
    "sources": [
        "CppCoreGuidelines rationale (WHY not HOW)",
        "Github issues marked 'bug' + PR fixes",
        "Engineering blog post postmortems",
        "Talk recordings (bugs + lessons)",
        "Safety-focused code reviews",
    ],
    
    "critical_lessons": [
        "Defensive coding saves debugging time",
        "Edge cases are where bugs hide",
        "Concurrency is harder than it looks",
        "Resource cleanup is not optional",
        "Type safety catches many bugs early",
    ],
    
    "success_metrics": [
        "AI spots potential bugs before running code",
        "AI suggests defensive patterns",
        "AI identifies concurrency issues",
        "AI stops confidently hallucinating wrong answers",
        "AI explains multiple failure modes",
    ],
}

# ==============================================================================
# CURRICULUM STRUCTURE
# ==============================================================================

CURRICULUM = {
    "title": "High-Quality C++ AI Training Curriculum",
    "principle": "Quality > Quantity > Diversity",
    "total_duration": "8 weeks (structured)",
    
    "phases": [
        ("Phase 1", PHASE_1_FOUNDATION, "Weeks 1-2", "CRITICAL"),
        ("Phase 2", PHASE_2_EXPLAINED, "Weeks 3-4", "CRITICAL"),
        ("Phase 3", PHASE_3_COMPLEXITY, "Weeks 5-6", "HIGH"),
        ("Phase 4", PHASE_4_FAILURES, "Weeks 7-8", "HIGH"),
    ],
    
    "flow": """
    Phase 1: Foundation (Taste, Basics)
           ↓
    Phase 2: Reasoning (Why, Not How)
           ↓
    Phase 3: Scale (Production Reality)
           ↓
    Phase 4: Defense (Mistakes & Growth)
    """,
    
    "key_principles": [
        "Start with ONE language: C++ (systems thinking foundation)",
        "All code is tagged: language, difficulty, domain, quality, type",
        "Reference material FIRST (authoritative baseline)",
        "Explanations paired with code (not code alone)",
        "Production code trumps toy code always",
        "Real failures teach better than theory",
        "NO junk: abandoned repos, AI-generated code, exploits",
    ],
}

# ==============================================================================
# DATASET TAGGING STRUCTURE (for tracking quality)
# ==============================================================================

TAGGING_SCHEMA = {
    "language": ["c++", "c", "python", "rust", "java"],
    "difficulty": ["beginner", "intermediate", "advanced", "expert"],
    "domain": [
        "stdlib",        # Standard library
        "memory",        # Memory management
        "concurrency",   # Threading, async
        "patterns",      # Design patterns
        "systems",       # Systems programming
        "architecture",  # Large-scale design
        "principles",    # Fundamentals
        "tools",         # Compilers, build systems
    ],
    "content_type": [
        "reference",           # Official documentation
        "tutorial",            # Guided walkthrough
        "style_guide",         # Conventions + rules
        "concept_walkthrough", # Explanation + code
        "design_pattern",      # Pattern with context
        "production",          # Real application code
        "comparison",          # This vs that analysis
        "bugfix",              # Problem + solution
        "postmortem",          # Failure analysis
        "code_review",         # PR feedback
    ],
    "quality": [
        "gold",        # Authoritative, well-maintained
        "silver",      # Good, reliable, tested
        "bronze",      # Acceptable, verified works
        "unvetted",    # Not yet verified
    ],
    "knowledge_value": [
        "exemplar",     # Model this code
        "antipattern",  # Don't do this (with explanation)
        "cautionary",   # Interesting but dangerous
        "reference",    # Look it up, don't copy
    ],
}

# ==============================================================================
# WHAT NOT TO TRAIN ON (POISON)
# ==============================================================================

POISON_FILTERS = {
    "critical_avoid": [
        "AI-generated code (poisons the well)",
        "Competitive programming obfuscation (not idiomatic)",
        "Exploit code (teaches attack patterns, not defense)",
        "Decompiled binaries (corrupted structure)",
        "Random GitHub dumps (unknown quality)",
    ],
    
    "avoid_early": [
        "Undocumented legacy code (teaches bad practices)",
        "Hack-filled 'production' code (often is terrible)",
        "StackOverflow snippets without context (misleading)",
        "Toy implementations (don't reflect real concerns)",
        "Abandoned projects (unmaintained advice)",
    ],
    
    "avoid_without_context": [
        "Platform-specific code (C++/Windows only)",
        "Highly optimized code (premature optimization)",
        "Bleeding-edge experimental features (unstable)",
    ],
}

# ==============================================================================
# PHASED INGESTION CHECKLIST
# ==============================================================================

INGESTION_CHECKLIST = {
    "before_ingestion": [
        "[ ] Is source authoritative or well-curated?",
        "[ ] Is it recent (< 2 years old)?",
        "[ ] Is it actively maintained?",
        "[ ] Are there tests validating claims?",
        "[ ] Does code follow idiomatic C++ style?",
        "[ ] Is it documented (why, not just what)?",
        "[ ] Is it PRODUCTION code or explained example?",
    ],
    
    "tagging_phase": [
        "[ ] Language: ?",
        "[ ] Difficulty: ?",
        "[ ] Domain: ?",
        "[ ] Content type: ?",
        "[ ] Quality level: ?",
        "[ ] Knowledge value: exemplar/antipattern/reference?",
        "[ ] What specific concepts does it teach?",
    ],
    
    "ingestion_phase": [
        "[ ] Extracted without corruption?",
        "[ ] Split into sensible chunks?",
        "[ ] Paired with explanation (if code)?",
        "[ ] Tagged with full metadata?",
        "[ ] Verified ingestion success?",
    ],
    
    "verification_phase": [
        "[ ] AI can explain key concepts?",
        "[ ] AI knows when/why to apply patterns?",
        "[ ] AI spots violations of what it learned?",
        "[ ] No hallucinations introduced?",
    ],
}

# ==============================================================================
# SUMMARY
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("C++ AI TRAINING CURRICULUM")
    print("Golden Rule: Quality > Quantity > Diversity")
    print("=" * 80)
    
    print("\n📚 PHASES:")
    for phase, data, weeks, priority in CURRICULUM["phases"]:
        print(f"\n  {phase} ({weeks}) - PRIORITY: {priority}")
        print(f"    → {data['goal']}")
    
    print("\n\n🎯 KEY PRINCIPLES:")
    for principle in CURRICULUM["key_principles"]:
        print(f"  • {principle}")
    
    print("\n\n⚠️  POISON FILTERS (NEVER):")
    for item in POISON_FILTERS["critical_avoid"]:
        print(f"  ✘ {item}")
    
    print("\n\n" + "=" * 80)
    print("Start Phase 1. Complete it fully before moving to Phase 2.")
    print("=" * 80)
