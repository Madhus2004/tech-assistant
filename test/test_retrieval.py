# test_retrieval.py
from src.retriever import retrieve_documents, format_context


def test_role_isolation():
    """
    Tests that each role only retrieves documents it should see.
    This is the core RBAC verification test.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Intern asks about financial data")
    print("Expected: No financial/executive documents returned")
    print("=" * 60)
    docs = retrieve_documents(
        query="What is the company revenue and financial performance?",
        role="intern"
    )
    for doc in docs:
        print(f"  Source: {doc.metadata['source']} "
              f"| Role: {doc.metadata['role']}")
    print()

    print("=" * 60)
    print("TEST 2: Executive asks about financial data")
    print("Expected: Financial documents returned")
    print("=" * 60)
    docs = retrieve_documents(
        query="What is the company revenue and financial performance?",
        role="executive"
    )
    for doc in docs:
        print(f"  Source: {doc.metadata['source']} "
              f"| Role: {doc.metadata['role']}")
    print()

    print("=" * 60)
    print("TEST 3: Engineer asks about deployment")
    print("Expected: Deployment runbook returned")
    print("=" * 60)
    docs = retrieve_documents(
        query="How do I deploy a service to production?",
        role="engineer"
    )
    for doc in docs:
        print(f"  Source: {doc.metadata['source']} "
              f"| Role: {doc.metadata['role']}")
    print()

    print("=" * 60)
    print("TEST 4: Intern asks about deployment")
    print("Expected: No deployment docs — intern has no engineer access")
    print("=" * 60)
    docs = retrieve_documents(
        query="How do I deploy a service to production?",
        role="intern"
    )
    for doc in docs:
        print(f"  Source: {doc.metadata['source']} "
              f"| Role: {doc.metadata['role']}")
    print()


if __name__ == "__main__":
    test_role_isolation()