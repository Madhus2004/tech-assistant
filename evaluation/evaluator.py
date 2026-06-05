# evaluation/evaluator.py
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from src.hybrid_retriever import hybrid_retrieve
from src.retriever import format_context
from src.prompt import build_prompt, SYSTEM_PROMPT
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from evaluation.eval_dataset import EVAL_DATASET


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        max_tokens=1024
    )


def evaluate_retrieval(question: str, role: str,
                       expected_source: str) -> dict:
    """
    Evaluates whether the correct document was retrieved.

    Metrics:
    - source_hit: did expected_source appear in retrieved chunks?
    - retrieval_count: how many chunks were retrieved
    - sources_retrieved: which documents were actually retrieved
    """
    documents = hybrid_retrieve(query=question, role=role, top_k=4)
    sources = [doc.metadata.get("source", "") for doc in documents]

    source_hit = False
    if expected_source:
        source_hit = expected_source in sources

    return {
        "documents": documents,
        "sources_retrieved": sources,
        "source_hit": source_hit,
        "retrieval_count": len(documents)
    }


def evaluate_answer(answer: str, expected_keywords: list,
                    should_answer: bool) -> dict:
    """
    Evaluates answer quality using keyword matching.

    Metrics:
    - keyword_hit_rate: fraction of expected keywords found in answer
    - rbac_correct: for restricted queries, did system refuse correctly?
    - answer_length: proxy for answer completeness
    """
    answer_lower = answer.lower()

    # Check keyword presence
    keywords_found = [
        kw for kw in expected_keywords
        if kw.lower() in answer_lower
    ]
    keyword_hit_rate = len(keywords_found) / len(expected_keywords)

    # RBAC check — if should_answer=False, system should have refused
    rbac_correct = True
    if not should_answer:
        # System should have said it doesn't have the information
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
    Runs the full evaluation suite against all test cases.
    Prints results to terminal and saves to evaluation/results.json.
    """
    llm = get_llm()
    results = []
    summary = {
        "total": 0,
        "retrieval_hits": 0,
        "rbac_correct": 0,
        "avg_keyword_hit_rate": 0.0,
        "by_category": {},
        "by_role": {},
        "failed_cases": []
    }

    print("\n" + "=" * 60)
    print("TECHNOVA RAG EVALUATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test cases: {len(EVAL_DATASET)}")
    print("=" * 60 + "\n")

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

        # Step 1: Retrieval evaluation
        retrieval_result = evaluate_retrieval(
            question=question,
            role=role,
            expected_source=expected_source
        )

        # Step 2: Generate answer
        context = format_context(retrieval_result["documents"])
        prompt = build_prompt(
            question=question,
            context=context,
            role=role
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        answer = response.content

        # Step 3: Answer evaluation
        answer_result = evaluate_answer(
            answer=answer,
            expected_keywords=expected_keywords,
            should_answer=should_answer
        )

        # Compile result
        result = {
            "id": test_id,
            "question": question,
            "role": role,
            "category": category,
            "should_answer": should_answer,
            "sources_retrieved": retrieval_result["sources_retrieved"],
            "source_hit": retrieval_result["source_hit"],
            "keyword_hit_rate": answer_result["keyword_hit_rate"],
            "keywords_found": answer_result["keywords_found"],
            "keywords_missed": answer_result["keywords_missed"],
            "rbac_correct": answer_result["rbac_correct"],
            "answer_preview": answer[:200],
            "answer_length": answer_result["answer_length"]
        }
        results.append(result)

        # Print result
        source_status = "✓" if retrieval_result["source_hit"] else "✗"
        if not should_answer:
            source_status = "N/A"

        rbac_status = "✓" if answer_result["rbac_correct"] else "✗"
        kw_rate = answer_result["keyword_hit_rate"]

        print(f"  Source hit: {source_status} "
              f"| Keywords: {kw_rate:.0%} "
              f"| RBAC: {rbac_status}")
        print(f"  Retrieved: {retrieval_result['sources_retrieved']}")
        print()

        # Rate limiting — Groq free tier
        time.sleep(1)

    # Calculate summary
    summary["total"] = len(results)
    summary["retrieval_hits"] = sum(
        1 for r in results if r["source_hit"] or not r["should_answer"]
    )
    summary["rbac_correct"] = sum(
        1 for r in results if r["rbac_correct"]
    )
    summary["avg_keyword_hit_rate"] = round(
        sum(r["keyword_hit_rate"] for r in results) / len(results), 2
    )

    # Group by role
    for role in ["intern", "engineer", "manager", "executive"]:
        role_results = [r for r in results if r["role"] == role]
        summary["by_role"][role] = {
            "total": len(role_results),
            "avg_keyword_hit_rate": round(
                sum(r["keyword_hit_rate"] for r in role_results)
                / len(role_results), 2
            )
        }

    # Group by category
    categories = set(r["category"] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        summary["by_category"][cat] = {
            "total": len(cat_results),
            "avg_keyword_hit_rate": round(
                sum(r["keyword_hit_rate"] for r in cat_results)
                / len(cat_results), 2
            )
        }

    # Failed cases
    summary["failed_cases"] = [
        r["id"] for r in results
        if r["keyword_hit_rate"] < 0.5 or not r["rbac_correct"]
    ]

    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total test cases:       {summary['total']}")
    print(f"Retrieval hits:         "
          f"{summary['retrieval_hits']}/{summary['total']}")
    print(f"RBAC enforcement:       "
          f"{summary['rbac_correct']}/{summary['total']} correct")
    print(f"Avg keyword hit rate:   "
          f"{summary['avg_keyword_hit_rate']:.0%}")
    print()
    print("By role:")
    for role, stats in summary["by_role"].items():
        print(f"  {role:12} → {stats['avg_keyword_hit_rate']:.0%} "
              f"keyword hit rate")
    print()
    if summary["failed_cases"]:
        print(f"Failed cases: {summary['failed_cases']}")
    else:
        print("No failed cases.")
    print("=" * 60)

    # Save results
    os.makedirs("evaluation", exist_ok=True)
    output_path = "evaluation/results.json"
    with open(output_path, "w") as f:
        json.dump(
            {"summary": summary, "results": results},
            f, indent=2
        )
    print(f"\nFull results saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()