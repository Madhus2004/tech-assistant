# evaluation/eval_dataset.py

# Each test case has:
# - question: what the user asks
# - role: which user is asking
# - expected_source: which document should be retrieved
# - expected_keywords: words that must appear in a good answer
# - should_answer: False means the system should say "not available"

EVAL_DATASET = [

    # ── Intern tests ──────────────────────────────────────────
    {
        "id": "intern_001",
        "question": "How do I apply for leave?",
        "role": "intern",
        "expected_source": "onboarding_guide",
        "expected_keywords": ["leave", "jira", "5 business days"],
        "should_answer": True,
        "category": "policy"
    },
    {
        "id": "intern_002",
        "question": "What is the pull request process?",
        "role": "intern",
        "expected_source": "coding_standards",
        "expected_keywords": ["pull request", "reviewer", "jira"],
        "should_answer": True,
        "category": "process"
    },
    {
        "id": "intern_003",
        "question": "What is the company annual revenue?",
        "role": "intern",
        "expected_source": None,
        "expected_keywords": ["don't have information", "not available"],
        "should_answer": False,
        "category": "rbac_enforcement"
    },

    # ── Engineer tests ────────────────────────────────────────
    {
        "id": "engineer_001",
        "question": "How do I rollback a failed deployment?",
        "role": "engineer",
        "expected_source": "deployment_runbook",
        "expected_keywords": ["rollback", "github actions", "cloudwatch"],
        "should_answer": True,
        "category": "technical"
    },
    {
        "id": "engineer_002",
        "question": "What is a Sev-1 incident?",
        "role": "engineer",
        "expected_source": "incident_response",
        "expected_keywords": ["sev-1", "outage", "15 minutes"],
        "should_answer": True,
        "category": "technical"
    },
    {
        "id": "engineer_003",
        "question": "What are the company growth targets?",
        "role": "engineer",
        "expected_source": None,
        "expected_keywords": ["don't have information", "not available"],
        "should_answer": False,
        "category": "rbac_enforcement"
    },

    # ── Manager tests ─────────────────────────────────────────
    {
        "id": "manager_001",
        "question": "What are the promotion requirements?",
        "role": "manager",
        "expected_source": "performance_reviews",
        "expected_keywords": ["promotion", "rating", "4 or above"],
        "should_answer": True,
        "category": "hr"
    },
    {
        "id": "manager_002",
        "question": "Who approves software purchases?",
        "role": "manager",
        "expected_source": "budget_policy",
        "expected_keywords": ["software", "approval", "it security"],
        "should_answer": True,
        "category": "process"
    },
    {
        "id": "manager_003",
        "question": "What is the company acquisition strategy?",
        "role": "manager",
        "expected_source": None,
        "expected_keywords": ["don't have information", "not available"],
        "should_answer": False,
        "category": "rbac_enforcement"
    },

    # ── Executive tests ───────────────────────────────────────
    {
        "id": "executive_001",
        "question": "What was the annual revenue in FY2024?",
        "role": "executive",
        "expected_source": "financial_overview",
        "expected_keywords": ["48", "revenue", "arr"],
        "should_answer": True,
        "category": "financial"
    },
    {
        "id": "executive_002",
        "question": "What are the top company risks?",
        "role": "executive",
        "expected_source": "governance_handbook",
        "expected_keywords": ["risk", "ml engineers", "retention"],
        "should_answer": True,
        "category": "strategic"
    },
    {
        "id": "executive_003",
        "question": "What are the European market revenue targets?",
        "role": "executive",
        "expected_source": "strategic_growth_plan",
        "expected_keywords": ["amsterdam", "european", "4.2"],
        "should_answer": True,
        "category": "strategic"
    }
]