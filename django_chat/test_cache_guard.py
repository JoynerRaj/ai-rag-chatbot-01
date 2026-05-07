"""
Local test for the strict document-only cache gate.
Run from: m:\INTERN\ai-rag-chatbot\django_chat\
  python test_cache_guard.py
"""

# ── exact copy of the real logic from ai_agent.py + semantic_cache.py ────────

_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "helo", "sup",
    "bye", "goodbye", "ok", "okay", "thanks", "thank",
}

_MEMORY_PATTERNS = (
    "previous conversation", "last message", "what did we talk",
    "summarize our", "summarize the chat", "earlier you said",
    "what was my", "remind me", "our conversation",
)


def should_cache(query, has_document_context):
    """Returns True only when the answer came from an uploaded document."""
    if not has_document_context:
        return False
    q = query.strip().lower().rstrip("!?.,:;")
    words = q.split()
    if len(words) < 3:
        return False
    if any(w in _GREETING_WORDS for w in words):
        return False
    for pattern in _MEMORY_PATTERNS:
        if pattern in q:
            return False
    return True


def should_write_cache(query, has_context, document_id, user_id):
    """
    Mirrors the exact condition used in ai_agent.py before calling semantic_cache_set.
    All four must be True for a write to happen.
    """
    return (
        has_context
        and bool(document_id)           # user must have selected a specific doc
        and should_cache(query, has_document_context=True)
        and user_id is not None
    )


# ── test cases ────────────────────────────────────────────────────────────────
# (label, query, has_context, document_id, user_id, should_write)
cases = [
    # general questions with NO document selected  ->  NEVER cache
    ("weather (no doc)",          "what is the weather today in tamil nadu", True,  None,   1, False),
    ("chemistry (no doc)",        "what is chemistry",                        True,  None,   1, False),
    ("greeting (no doc)",         "hi how are you",                           True,  None,   1, False),
    ("general (no user)",         "what is machine learning",                 True,  "doc1", None, False),

    # general questions WITH a document selected, but answer did NOT come from doc
    ("weather no context",        "what is the weather today",               False, "doc1",  1, False),
    ("news no context",           "what happened in the news today",         False, "doc1",  1, False),

    # greetings and small talk  ->  NEVER cache even with a document selected
    ("hi",                        "hi",                                       True,  "doc1",  1, False),
    ("hi how are you",            "hi how are you",                           True,  "doc1",  1, False),
    ("hello there",               "hello there",                              True,  "doc1",  1, False),
    ("thanks",                    "thanks",                                   True,  "doc1",  1, False),
    ("okay thanks",               "okay thanks",                              True,  "doc1",  1, False),

    # conversation memory  ->  NEVER cache
    ("previous convo",            "what is my previous conversation",         True,  "doc1",  1, False),
    ("summarize chat",            "summarize our chat",                       True,  "doc1",  1, False),

    # real document questions WITH a specific document selected  ->  SHOULD cache
    ("f1 score from doc",         "what is f1 score",                         True,  "doc1",  1, True),
    ("gradient descent from doc", "explain gradient descent",                 True,  "doc1",  1, True),
    ("ML definition from doc",    "what is machine learning",                 True,  "doc1",  1, True),
    ("precision vs recall",       "difference between precision and recall",  True,  "doc1",  1, True),
]

print(f"{'Label':<30} {'got':>5} {'want':>5} {'result':>6}")
print("-" * 55)

all_pass = True
for label, query, has_ctx, doc_id, uid, expected in cases:
    got = should_write_cache(query, has_ctx, doc_id, uid)
    status = "PASS" if got == expected else "FAIL"
    if got != expected:
        all_pass = False
    print(f"{label:<30} {str(got):>5} {str(expected):>5} {status}")

print()
if all_pass:
    print("All tests passed - safe to deploy")
else:
    print("TESTS FAILED - do not push until fixed")
