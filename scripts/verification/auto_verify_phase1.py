#!/usr/bin/env python3
"""
AUTO-VERIFICATION TEST SUITE - Phase 1 Learning Validation

Automatically tests whether AI learned C++ fundamentals correctly.
Detects hallucination and gaps in knowledge without requiring interaction.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from Zypherus.core.ace import ACE

# ============================================================================
# PHASE 1 FOUNDATION TESTS - Auto-verify
# ============================================================================

PHASE1_TESTS = [
    {
        "id": "test_raii_concept",
        "question": "What is RAII (Resource Acquisition Is Initialization) and why is it important?",
        "checks_for": [
            "destructor",
            "resource",
            "automatic",
            "acquired",
            "released"
        ],
        "red_flags": [
            "RAII is just about initializing variables",
            "RAII is only for memory",
            "destructors are optional",
            "manual cleanup is better",
        ],
    },
    {
        "id": "test_smart_pointers",
        "question": "Explain the difference between unique_ptr and shared_ptr",
        "checks_for": [
            "unique_ptr",
            "shared_ptr",
            "ownership",
            "exclusive",
            "reference count",
            "move",
        ],
        "red_flags": [
            "they are the same",
            "unique_ptr doesn't transfer ownership",
            "shared_ptr can be copied directly",
            "no performance difference",
        ],
    },
    {
        "id": "test_containers_selection",
        "question": "When should you use std::vector instead of std::list?",
        "checks_for": [
            "vector",
            "contiguous",
            "cache",
            "performance",
            "random access",
            "sequential",
        ],
        "red_flags": [
            "they perform the same",
            "list is always faster",
            "cache locality doesn't matter",
            "choose randomly",
        ],
    },
    {
        "id": "test_const_correctness",
        "question": "What does 'const' correctness mean and why does it matter?",
        "checks_for": [
            "const",
            "intention",
            "interface",
            "compile-time",
            "guarantee",
            "modification",
        ],
        "red_flags": [
            "const doesn't prevent anything",
            "const is just documentation",
            "const and mutable do the same thing",
            "const correctness is optional",
        ],
    },
    {
        "id": "test_move_semantics",
        "question": "Explain move semantics and when std::move should be used",
        "checks_for": [
            "rvalue",
            "lvalue",
            "efficient",
            "resources",
            "temporary",
            "copy",
        ],
        "red_flags": [
            "move and copy are the same",
            "move always makes code faster",
            "use move everywhere",
            "move doesn't transfer ownership",
        ],
    },
    {
        "id": "test_templates_why",
        "question": "Why do templates exist? What problem do they solve?",
        "checks_for": [
            "generic",
            "type",
            "compile-time",
            "reuse",
            "specialization",
        ],
        "red_flags": [
            "templates are just for std library",
            "templates always create bloat",
            "templates are too complicated",
            "runtime polymorphism is better",
        ],
    },
    {
        "id": "test_exception_safety",
        "question": "What is the strong exception guarantee?",
        "checks_for": [
            "exception",
            "committed",
            "rollback",
            "state",
            "atomic",
            "all or nothing",
        ],
        "red_flags": [
            "exceptions don't matter",
            "strong guarantee means no exceptions",
            "weak and strong are the same",
            "exception safety is optional",
        ],
    },
]

def score_response(response, test, debug=False):
    """Score AI response against test criteria"""
    
    # Handle dict response (from answer())
    if isinstance(response, dict):
        answer_text = response.get("answer", "")
        confidence = response.get("confidence", 0.0)
    else:
        answer_text = str(response)
        confidence = 0.0
    
    response_lower = answer_text.lower()
    
    if debug:
        print(f"    [DEBUG] Answer length: {len(answer_text)}")
        print(f"    [DEBUG] Confidence: {confidence}")
        print(f"    [DEBUG] Answer preview: {answer_text[:200]}...")
        print(f"    [DEBUG] Checking for: {test['checks_for']}")
    
    # Check for red flags (immediate fail)
    red_flag_hits = 0
    for flag in test["red_flags"]:
        if flag.lower() in response_lower:
            red_flag_hits += 1
    
    # Check for expected knowledge
    check_hits = 0
    found_keywords = []
    for check in test["checks_for"]:
        check_lower = check.lower()
        # Lenient matching: exact substring match (handles common variations)
        # e.g., "reference count" in response matches "reference count" check
        # e.g., "acquiring" contains stem "acquir" which is from "acquired"
        
        # Strategy: look for exact match first, then try common stems
        if check_lower in response_lower:
            check_hits += 1
            found_keywords.append(check)
        else:
            # Try stems for common word families
            stems_to_try = []
            if check_lower.endswith('ed'):
                stems_to_try.append(check_lower[:-2])  # "acquired" -> "acquire"
            if check_lower.endswith('s'):
                stems_to_try.append(check_lower[:-1])  # "pointers" -> "pointer"
            if check_lower.endswith('ing'):
                stems_to_try.append(check_lower[:-3])  # "acquiring" -> "acquire"
            
            # Check if any stem is in the response
            if any(stem in response_lower for stem in stems_to_try):
                check_hits += 1
                found_keywords.append(check)
    
    if debug and found_keywords:
        print(f"    [DEBUG] Found keywords: {found_keywords}")
    
    # Scoring
    if red_flag_hits > 0:
        # Found hallucination
        return {
            "passed": False,
            "hallucination": True,
            "score": 0,
            "reason": f"Found {red_flag_hits} red flag(s) indicating hallucination",
            "confidence": confidence,
        }
    
    if check_hits < len(test["checks_for"]) * 0.6:
        # Missing key concepts
        return {
            "passed": False,
            "hallucination": False,
            "score": check_hits / len(test["checks_for"]),
            "reason": f"Missing key concepts ({check_hits}/{len(test['checks_for'])})",
            "confidence": confidence,
        }
    
    return {
        "passed": True,
        "hallucination": False,
        "score": 1.0,
        "reason": f"Correct understanding ({check_hits}/{len(test['checks_for'])} concepts)",
        "confidence": confidence,
    }

def run_phase1_verification():
    """Run Phase 1 auto-verification"""
    print("="*70)
    print("PHASE 1 FOUNDATION VERIFICATION")
    print("="*70)
    print("\nTesting whether AI learned C++ fundamentals correctly...")
    print("(Auto-checking responses for hallucination and understanding)\n")
    
    ace = ACE()
    
    passed = 0
    failed = 0
    hallucinations = 0
    results = []
    
    for idx, test in enumerate(PHASE1_TESTS, 1):
        print(f"[{idx}/{len(PHASE1_TESTS)}] {test['id']}")
        print(f"    Q: {test['question'][:70]}...")
        
        try:
            # Ask ACE
            response = ace.ask(test['question'])
            
            # Handle dict response
            if isinstance(response, dict):
                answer_text = response.get("answer", "")
                confidence = response.get("confidence", 0.0)
            else:
                answer_text = str(response)
                confidence = 0.0
            
            if not answer_text or len(answer_text) < 20:
                print(f"    [FAIL] No meaningful response (got: '{answer_text[:50]}')")
                failed += 1
                results.append({
                    "test": test['id'],
                    "passed": False,
                    "reason": "No response"
                })
                continue
            
            # Score response
            is_first_test = (idx == 1)  # Debug first test only
            score = score_response(response, test, debug=is_first_test)
            
            if score["hallucination"]:
                print(f"    [HALLUCINATION] {score['reason']} (conf: {score['confidence']:.2f})")
                hallucinations += 1
                failed += 1
                results.append({
                    "test": test['id'],
                    "passed": False,
                    "reason": f"Hallucination: {score['reason']}"
                })
            elif score["passed"]:
                print(f"    [PASS] {score['reason']} (conf: {score['confidence']:.2f})")
                passed += 1
                results.append({
                    "test": test['id'],
                    "passed": True,
                    "reason": score['reason']
                })
            else:
                print(f"    [FAIL] {score['reason']} (conf: {score['confidence']:.2f})")
                failed += 1
                results.append({
                    "test": test['id'],
                    "passed": False,
                    "reason": score['reason']
                })
        
        except Exception as e:
            print(f"    [ERROR] {str(e)[:60]}")
            failed += 1
            results.append({
                "test": test['id'],
                "passed": False,
                "reason": str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 1 RESULTS")
    print("="*70)
    print(f"\nPassed:        {passed}/{len(PHASE1_TESTS)} ({100*passed//len(PHASE1_TESTS)}%)")
    print(f"Failed:        {failed}/{len(PHASE1_TESTS)}")
    print(f"Hallucinated:  {hallucinations}")
    
    if hallucinations > 0:
        print(f"\n[WARNING] AI is hallucinating - confident wrong answers detected!")
        print("Recommendation: Review knowledge sources, filter for quality.")
    
    if passed == len(PHASE1_TESTS):
        print(f"\n[SUCCESS] AI mastered Phase 1 fundamentals!")
    elif passed >= len(PHASE1_TESTS) * 0.8:
        print(f"\n[GOOD] AI understands basics, but has gaps - continue training")
    else:
        print(f"\n[WEAK] AI needs more structured learning")
    
    print("\n" + "="*70)
    return {
        "phase": 1,
        "passed": passed,
        "failed": failed,
        "hallucinations": hallucinations,
        "total": len(PHASE1_TESTS),
        "results": results,
    }

if __name__ == "__main__":
    result = run_phase1_verification()
