# evaluation/evaluator.py
import os
import sys
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from evaluation.eval_dataset import EVAL_DATASET

# ── Configuration ──────────────────────────────────────────
# Switch between local and deployed evaluation
EVAL_TARGET = os.getenv("EVAL_TARGET","local")

TARGETS = {
    "local": "http://localhost:5000",
    "render": "https://enterprise-knowledge-assistant-zzt4.onrender.com/"  # replace with your actual URL
}


def get_base_url() -> str:
    url = TARGETS.get(EVAL_TARGET, TARGETS["local"])
    print(f"Evaluating against: {url}")
    return url


def login_and_get_session(base_url: str, username: str,
                           password: str) -> requests.Session:
    """
    Creates an authenticated session for a given user.
    Returns a requests.Session with the Flask session cookie set.
    """
    session = requests.Session()
    response = session.post(
        f"{base_url}/login",
        data={"username": username, "password": password},
        allow_redirects=True,
        timeout=120
    )
    if response.status_code == 200 and "chat" in response.url:
        print(f"  Logged in as {username}")
        return session
    else:
        raise Exception(f"Login failed for {username}. "
                        f"Status: {response.status_code}")


def ask_question(session: requests.Session, base_url: str,
                 question: str) -> dict:
    """
    Sends a question to the /ask endpoint using an authenticated session.
    Returns the JSON response with answer and sources.
    """
    response = session.post(
        f"{base_url}/ask",
        json={"question": question},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


# Role to username mapping
ROLE_USERS = {
    "intern": ("alice", "intern123"),
    "engineer": ("bob", "engineer123"),
    "manager": ("carol", "manager123"),
    "executive": ("dave", "executive123")
}


def evaluate_answer(answer: str, expected_keywords: list,
                    should_answer: bool) -> dict:
    """Evaluates answer quality using keyword matching."""
    answer_lower = answer.lower()

    keywords_found = [
        kw for kw in expected_keywords
        if kw.lower() in answer_lower
    ]
    keyword_hit_rate = len(keywords_found) / len(expected_keywords)

    rbac_correct = True
    if not should_answer:
        refusal_phrases = [
            "don't have information",
            "not available",
            "cannot find",
            "no information",
            "not in the documents",
            "available to your role"
        ]
        rbac_correct = any(
            phrase in answer_lower for phrase in refusal_phrases
        )

    return {
        "keywords_found": keywords_found,
        "keywords_missed": [
            kw for kw in expected_keywords
            if kw.lower() not in answer_lower
        ],
        "keyword_hit_rate": round(keyword_hit_rate, 2),
        "rbac_correct": rbac_correct,
        "answer_length": len(answer)
    }


def run_evaluation() -> dict:
    """
    Runs full evaluation suite against local or Render deployment.
    Uses HTTP requests with session cookies — no direct Python imports needed.
    This means it tests the REAL deployed system end to end.
    """
    base_url = get_base_url()
    results = []

    summary = {
        "total": 0,
        "retrieval_hits": 0,
        "rbac_correct": 0,
        "avg_keyword_hit_rate": 0.0,
        "eval_target": EVAL_TARGET,
        "base_url": base_url,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "by_category": {},
        "by_role": {},
        "failed_cases": []
    }

    print("\n" + "=" * 60)
    print("TECHNOVA RAG EVALUATION")
    print(f"Target:    {EVAL_TARGET} → {base_url}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Test cases: {len(EVAL_DATASET)}")
    print("=" * 60 + "\n")

    # Cache sessions per role — one login per role not per test case
    sessions = {}

    for i, test_case in enumerate(EVAL_DATASET, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        role = test_case["role"]
        expected_source = test_case["expected_source"]
        expected_keywords = test_case["expected_keywords"]
        should_answer = test_case["should_answer"]
        category = test_case["category"]

        print(f"[{i}/{len(EVAL_DATASET)}] {test_id}")
        print(f"  Q: {question}")
        print(f"  Role: {role} | Category: {category}")

        # Get or create session for this role
        if role not in sessions:
            username, password = ROLE_USERS[role]
            sessions[role] = login_and_get_session(
                base_url, username, password
            )

        # Ask the question via HTTP
        try:
            response_data = ask_question(
                sessions[role], base_url, question
            )
            answer = response_data.get("answer", "")
            sources = response_data.get("sources", [])

        except Exception as e:
            print(f"  ERROR: {e}")
            answer = ""
            sources = []

        # Evaluate answer
        answer_result = evaluate_answer(
            answer=answer,
            expected_keywords=expected_keywords,
            should_answer=should_answer
        )

        # Check source hit
        source_hit = False
        if expected_source:
            source_hit = expected_source in sources

        result = {
            "id": test_id,
            "question": question,
            "role": role,
            "category": category,
            "should_answer": should_answer,
            "sources_retrieved": sources,
            "source_hit": source_hit,
            "keyword_hit_rate": answer_result["keyword_hit_rate"],
            "keywords_found": answer_result["keywords_found"],
            "keywords_missed": answer_result["keywords_missed"],
            "rbac_correct": answer_result["rbac_correct"],
            "answer_preview": answer[:200],
            "answer_length": answer_result["answer_length"]
        }
        results.append(result)

        # Print result
        source_status = "✓" if source_hit else "✗"
        if not should_answer:
            source_status = "N/A"

        print(f"  Source: {source_status} "
              f"| Keywords: {answer_result['keyword_hit_rate']:.0%} "
              f"| RBAC: {'✓' if answer_result['rbac_correct'] else '✗'}")
        print(f"  Sources returned: {sources}")
        print()

        # Rate limiting
        time.sleep(2)

    # Calculate summary
    summary["total"] = len(results)
    summary["retrieval_hits"] = sum(
        1 for r in results
        if r["source_hit"] or not r["should_answer"]
    )
    summary["rbac_correct"] = sum(
        1 for r in results if r["rbac_correct"]
    )
    summary["avg_keyword_hit_rate"] = round(
        sum(r["keyword_hit_rate"] for r in results) / len(results), 2
    )

    # By role
    for role in ["intern", "engineer", "manager", "executive"]:
        role_results = [r for r in results if r["role"] == role]
        summary["by_role"][role] = {
            "total": len(role_results),
            "avg_keyword_hit_rate": round(
                sum(r["keyword_hit_rate"] for r in role_results)
                / len(role_results), 2
            )
        }

    # By category
    for cat in set(r["category"] for r in results):
        cat_results = [r for r in results if r["category"] == cat]
        summary["by_category"][cat] = {
            "total": len(cat_results),
            "avg_keyword_hit_rate": round(
                sum(r["keyword_hit_rate"] for r in cat_results)
                / len(cat_results), 2
            )
        }

    summary["failed_cases"] = [
        r["id"] for r in results
        if r["keyword_hit_rate"] < 0.5 or not r["rbac_correct"]
    ]

    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Target:               {EVAL_TARGET.upper()} — {base_url}")
    print(f"Total test cases:     {summary['total']}")
    print(f"Retrieval hits:       "
          f"{summary['retrieval_hits']}/{summary['total']}")
    print(f"RBAC enforcement:     "
          f"{summary['rbac_correct']}/{summary['total']} correct")
    print(f"Avg keyword hit rate: "
          f"{summary['avg_keyword_hit_rate']:.0%}")
    print()
    print("By role:")
    for role, stats in summary["by_role"].items():
        print(f"  {role:12} → "
              f"{stats['avg_keyword_hit_rate']:.0%} keyword hit rate")
    print()
    if summary["failed_cases"]:
        print(f"Failed cases: {summary['failed_cases']}")
    else:
        print("No failed cases.")
    print("=" * 60)

    # Save results with target name so both are preserved
    os.makedirs("evaluation", exist_ok=True)
    output_path = f"evaluation/results_{EVAL_TARGET}.json"
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()