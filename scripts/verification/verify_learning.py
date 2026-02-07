"""
Learning Verification System

Tests whether AI actually learned what it should have at each phase.
Detects hallucination, gaps, and misconceptions.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from Zypherus.core.ace import ACE


@dataclass
class TestQuestion:
    phase: int
    category: str
    question: str
    checks_for: List[str]  # What the answer should demonstrate
    red_flags: List[str]   # Signs of hallucination
    level: str  # "basic", "intermediate", "advanced"


# ==============================================================================
# PHASE 1: FOUNDATION VERIFICATION
# ==============================================================================

PHASE_1_TESTS = [
    # RAII - foundational
    TestQuestion(
        phase=1,
        category="RAII",
        question="What is RAII in C++ and why is it important?",
        checks_for=[
            "mentions Resource Acquisition Is Initialization",
            "ties resource lifecycle to object lifetime",
            "mentions exceptions and guarantees",
            "gives concrete example (file, memory, lock)",
        ],
        red_flags=[
            "only mentions destructors",
            "talks about garbage collection",
            "doesn't mention exceptions",
            "gives no concrete example",
        ],
        level="basic",
    ),
    
    TestQuestion(
        phase=1,
        category="Pointers",
        question="Explain the difference between raw pointers and smart pointers. When would you use each?",
        checks_for=[
            "raw pointer: zero overhead, manual management",
            "smart pointer: automatic cleanup via RAII",
            "mentions unique_ptr for exclusive ownership",
            "mentions shared_ptr for shared ownership",
            "explains when each is appropriate",
        ],
        red_flags=[
            "treats them as equivalent",
            "doesn't mention ownership semantics",
            "no discussion of lifetime",
            "recommends raw pointers for new code",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=1,
        category="Smart Pointers",
        question="When should you use unique_ptr vs shared_ptr?",
        checks_for=[
            "unique_ptr: single owner (most cases)",
            "shared_ptr: multiple owners (rare)",
            "unique_ptr is zero-overhead",
            "shared_ptr has reference counting overhead",
            "warning: shared_ptr + cyclic references = leak",
        ],
        red_flags=[
            "recommends shared_ptr as default",
            "doesn't mention reference counting",
            "doesn't mention cyclic references",
            "treats them as interchangeable",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=1,
        category="Containers",
        question="Compare std::vector, std::list, and std::deque. What are the performance tradeoffs?",
        checks_for=[
            "vector: contiguous, fast random access, expensive insert/delete at front",
            "list: double-linked, slow random access, fast insert/delete anywhere",
            "deque: indexed access, fast insert/delete at both ends",
            "mentions cache locality for vector",
            "mentions iterator invalidation implications",
        ],
        red_flags=[
            "wrong performance characteristics",
            "doesn't mention cache effects",
            "recommends list without reason",
            "ignores real usage patterns",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=1,
        category="String Handling",
        question="Should you use std::string or std::string_view for function parameters? Why?",
        checks_for=[
            "string_view for read-only parameters (default)",
            "string_view: no allocation, works with literals",
            "string: when you need to store the string",
            "mentions lifetime/dangling reference risks with string_view",
            "explains performance advantage of string_view",
        ],
        red_flags=[
            "recommends string by default",
            "doesn't mention allocation costs",
            "ignores dangling reference risks",
            "doesn't distinguish ownership patterns",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=1,
        category="RAII - Concrete",
        question="Design a class that safely manages a file handle, ensuring it's closed even if exceptions happen.",
        checks_for=[
            "create-time: open file in constructor",
            "destroy-time: close file in destructor",
            "no manual cleanup required",
            "exception-safe",
            "mentions rule of five if needed",
        ],
        red_flags=[
            "suggests explicit close() calls",
            "no exception safety consideration",
            "manual resource management",
            "resource leaks in error case",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=1,
        category="Idiomatic Style",
        question="What makes C++ code idiomatic vs non-idiomatic? Give examples.",
        checks_for=[
            "uses STL algorithms instead of manual loops",
            "prefers const and references",
            "avoids raw pointers for ownership",
            "uses smart pointers correctly",
            "follows C++ Core Guidelines patterns",
        ],
        red_flags=[
            "writes C code in C++ syntax",
            "recommends Java-like patterns",
            "ignores STL algorithms",
            "uses C-style resource management",
        ],
        level="basic",
    ),
]

# ==============================================================================
# PHASE 2: EXPLANATIONS VERIFICATION
# ==============================================================================

PHASE_2_TESTS = [
    TestQuestion(
        phase=2,
        category="Why Questions",
        question="Why does std::optional exist in C++? Why not just use a pointer or return error codes?",
        checks_for=[
            "type-safe, compiler-enforced checking",
            "no null pointer issues",
            "clearer intent than pointers",
            "compared to alternatives with reasoning",
            "mentions performance (no allocation unlike unique_ptr)",
        ],
        red_flags=[
            "just lists features",
            "no comparison to alternatives",
            "doesn't explain safety advantages",
            "treats as equivalent to pointers",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=2,
        category="Lambda Capture",
        question="Explain the difference between capturing a variable by value vs by reference in a lambda.",
        checks_for=[
            "by value: copy at lambda creation, safe",
            "by reference: uses reference, lifetime dependent",
            "by-ref: dangerous if referent destroyed",
            "explains [=] vs [&] vs explicit captures",
            "shows concrete example with issue",
        ],
        red_flags=[
            "gets semantics wrong",
            "doesn't mention lifetime issues",
            "by-reference treated as always safe",
            "no concrete example",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=2,
        category="constexpr Reasoning",
        question="When should you use constexpr? What are the benefits and constraints?",
        checks_for=[
            "constexpr: forces compile-time evaluation when possible",
            "enables zero-cost abstractions",
            "requires pure functions (no I/O, global state access)",
            "can fallback to runtime if needed",
            "better than macros for same purpose",
        ],
        red_flags=[
            "treats as optimization hint (it's more)",
            "no mention of compile-time constraints",
            "doesn't explain use cases",
            "recommends for everything",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=2,
        category="Design Tradeoffs",
        question="Design a function to process a large vector of strings. Discuss parameter choices: std::vector<std::string>, std::vector<std::string_view>, or std::span?",
        checks_for=[
            "explains each choice's implications",
            "span: view over existing data (modern)",
            "string_view: non-owning, efficient",
            "vector: takes ownership",
            "performance and lifetime considerations",
        ],
        red_flags=[
            "picks one without justifying",
            "doesn't discuss ownership",
            "ignores performance implications",
            "treats all as equivalent",
        ],
        level="advanced",
    ),
    
    TestQuestion(
        phase=2,
        category="Copy Constructors",
        question="What is the Rule of Five in C++? Why does it matter?",
        checks_for=[
            "destructor, copy constructor, copy assignment",
            "move constructor, move assignment",
            "if you define one, consider all five",
            "examples: custom allocator, resource management",
            "explains why default versions are wrong",
        ],
        red_flags=[
            "outdated Rule of Three only",
            "doesn't explain move semantics",
            "doesn't give concrete cases",
            "treats defaults as always OK",
        ],
        level="advanced",
    ),
]

# ==============================================================================
# PHASE 3: PRODUCTION VERIFICATION
# ==============================================================================

PHASE_3_TESTS = [
    TestQuestion(
        phase=3,
        category="Concurrency",
        question="Write thread-safe code to increment a counter from multiple threads. What synchronization primitive should you use and why?",
        checks_for=[
            "atomic<int> for lock-free (preferred for simple counters)",
            "mutex + lock_guard for more complex operations",
            "explains why busy-spin is bad",
            "memory ordering not forgotten",
            "justifies choice",
        ],
        red_flags=[
            "uses volatile (doesn't provide thread safety)",
            "no memory ordering consideration",
            "race condition in proposed solution",
            "doesn't compare alternatives",
        ],
        level="advanced",
    ),
    
    TestQuestion(
        phase=3,
        category="Memory Ordering",
        question="What is memory_order_acquire vs memory_order_release? Why does it matter?",
        checks_for=[
            "acquire: prevents subsequent ops from moving before",
            "release: prevents prior ops from moving after",
            "explains happens-before relationships",
            "gives concurrency example",
            "mentions performance implications",
        ],
        red_flags=[
            "treats as too advanced to explain",
            "swaps acquire/release meanings",
            "no concrete example",
            "performance impact unexplained",
        ],
        level="expert",
    ),
]

# ==============================================================================
# PHASE 4: DEFENSIVE THINKING VERIFICATION
# ==============================================================================

PHASE_4_TESTS = [
    TestQuestion(
        phase=4,
        category="Bug Spotting",
        question="""
        What's wrong with this code?
        
        std::vector<int> v = getVector();
        auto it = v.begin();
        v.push_back(99);
        std::cout << *it;  // Safe?
        """,
        checks_for=[
            "identifies iterator invalidation",
            "push_back may reallocate",
            "it becomes invalid after reallocation",
            "dereferencing is undefined behavior",
            "suggests fix: store value, not iterator",
        ],
        red_flags=[
            "says it's safe",
            "doesn't mention reallocation",
            "misses undefined behavior",
            "suggests wrong fix",
        ],
        level="intermediate",
    ),
    
    TestQuestion(
        phase=4,
        category="Exception Safety",
        question="What are the exception safety levels? Give an example of each.",
        checks_for=[
            "strong: all-or-nothing",
            "basic: may leave something modified, but valid",
            "weak: exceptions, no guarantees",
            "no-throw: exception-safe operations",
            "concrete examples for each level",
        ],
        red_flags=[
            "missing exception safety concept",
            "wrong definitions",
            "no concrete examples",
            "doesn't explain when to use each",
        ],
        level="advanced",
    ),
    
    TestQuestion(
        phase=4,
        category="Defensive Coding",
        question="Spot potential bugs in this concurrent code.",
        checks_for=[
            "identifies race conditions",
            "spots atomicity issues",
            "sees memory ordering problems",
            "detects deadlock potential",
            "suggests defensive fixes",
        ],
        red_flags=[
            "misses obvious race condition",
            "doesn't consider memory ordering",
            "ignores deadlock possibilities",
            "suggests unsafe 'fix'",
        ],
        level="advanced",
    ),
]


class LearningVerifier:
    """Verifies AI learning at each phase."""
    
    def __init__(self):
        try:
            self.ace = ACE()
        except Exception as e:
            print(f"Warning: Could not initialize ACE: {e}")
            self.ace = None
    
    def run_test_suite(self, phase: int, interactive: bool = True) -> Dict:
        """Run all tests for a phase."""
        
        if not self.ace:
            print("Cannot run tests without Zypherus")
            return {}
        
        test_suites = {
            1: PHASE_1_TESTS,
            2: PHASE_2_TESTS,
            3: PHASE_3_TESTS,
            4: PHASE_4_TESTS,
        }
        
        tests = test_suites.get(phase, [])
        if not tests:
            print(f"No tests for phase {phase}")
            return {}
        
        print(f"\n{'=' * 80}")
        print(f"🧪 PHASE {phase} VERIFICATION TEST SUITE")
        print(f"{'=' * 80}")
        print(f"Running {len(tests)} tests to verify AI learning...\n")
        
        results = {
            "phase": phase,
            "total": len(tests),
            "passed": 0,
            "failed": 0,
            "details": [],
        }
        
        for i, test in enumerate(tests, 1):
            print(f"\n[{i}/{len(tests)}] {test.category}: {test.level.upper()}")
            print(f"Q: {test.question}")
            
            if interactive:
                # Get response from user running in REPL
                print(f"\n>>> Ask your AI:")
                print(f"    ask: {test.question}")
                response = input("\n✎ Summarize AI's response: ").strip()
            else:
                # Try to query ACE directly (limited)
                response = "[See test in REPL]"
            
            print(f"\n✓ What to look for (AI should mention these):")
            for check in test.checks_for:
                print(f"   {check}")
            
            print(f"\n✘ Red flags (hallucination signs):")
            for flag in test.red_flags:
                print(f"   ✘ {flag}")
            
            passed = input("\nDid AI pass this test? (yes/no): ").strip().lower() in ("yes", "y", "1")
            
            if passed:
                results["passed"] += 1
                results["details"].append({"test": test.category, "result": "PASS"})
                print("PASS")
            else:
                results["failed"] += 1
                results["details"].append({"test": test.category, "result": "FAIL"})
                print("FAIL - Note this. AI may be hallucinating or hasn't learned this.")
        
        # Summary
        success_rate = results["passed"] / results["total"] * 100
        
        print(f"\n{'=' * 80}")
        print(f"RESULTS: Phase {phase}")
        print(f"{'=' * 80}")
        print(f"Passed:  {results['passed']}/{results['total']} ({success_rate:.0f}%)")
        print(f"Failed:  {results['failed']}/{results['total']}")
        
        if results["passed"] == results["total"]:
            print(f"\nPHASE {phase} COMPLETE - AI has learned well!")
        elif results["passed"] >= results["total"] * 0.8:
            print(f"\n⚠️  PHASE {phase} MOSTLY OK - Minor gaps remain")
        else:
            print(f"\nPHASE {phase} NEEDS WORK - Significant gaps detected")
        
        return results


def main():
    """Main entry point."""
    
    print("=" * 80)
    print("🧪 AI LEARNING VERIFICATION SYSTEM")
    print("=" * 80)
    print("\nTests whether your AI actually learned what it should have.")
    print("Detects hallucination and gaps in knowledge.\n")
    
    print("Which phase to verify?")
    print("  1. Phase 1 (Foundation)")
    print("  2. Phase 2 (Explained)")
    print("  3. Phase 3 (Production)")
    print("  4. Phase 4 (Failures)")
    print("  0. Exit")
    
    choice = input("\nSelect phase (0-4): ").strip()
    
    if choice in ("1", "2", "3", "4"):
        phase = int(choice)
        verifier = LearningVerifier()
        results = verifier.run_test_suite(phase, interactive=True)
        
        if results:
            print("\n" + "=" * 80)
            print("💡 INTERPRETATION")
            print("=" * 80)
            
            if results["passed"] == results["total"]:
                print(f"AI mastered Phase {phase}. Ready for next phase.")
            elif results["passed"] >= results["total"] * 0.8:
                print(f"⚠️  AI mostly understands Phase {phase}.")
                print(f"   Recommendation: Minor review, then move forward")
            elif results["passed"] >= results["total"] * 0.6:
                print(f"AI has significant gaps in Phase {phase}.")
                print(f"   Recommendation: Reingest materials, retry verification")
            else:
                print(f"AI did not learn Phase {phase} well.")
                print(f"   Recommendation: Start over with Phase 1,")
                print(f"                  check for poison content,")
                print(f"                  verify ingestion worked")
            
            # Save results
            import json
            results_file = Path("data/ingestion") / f"phase_{phase}_verification.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n📁 Results saved: {results_file}")


if __name__ == "__main__":
    main()
