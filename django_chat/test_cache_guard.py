"""
Local test for the strict document-only cache gate.
Run from: m:\INTERN\ai-rag-chatbot\django_chat\
  python test_cache_guard.py
"""

# ── exact copy of the real should_cache logic in semantic_cache.py ────────────

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


# ── test cases ────────────────────────────────────────────────────────────────
# (query, has_document_context, expected_result)
cases = [
    # should NOT be cached
    ("hi",                                 True,  False),
    ("hi how are you",                     True,  False),
    ("hello",                              True,  False),
    ("hello there",                        True,  False),
    ("hey what is machine learning",       True,  False),
    ("thanks",                             True,  False),
    ("okay thanks",                        True,  False),
    ("ok sounds good",                     True,  False),
    ("what is my previous conversation",   True,  False),
    ("summarize our chat",                 True,  False),
    ("what did we talk about",             True,  False),
    ("remind me what we discussed",        True,  False),
    # no doc context -> never cache regardless
    ("what is chemistry",                  False, False),
    ("explain the water cycle",            False, False),
    # should be cached (document-based)
    ("what is f1 score",                   True,  True),
    ("explain gradient descent",           True,  True),
    ("what is machine learning",           True,  True),
    ("how does backpropagation work",      True,  True),
    ("what is the difference between precision and recall", True, True),
    ("define supervised learning",         True,  True),
]

print(f"{'Query':<52} {'ctx':>4} {'got':>5} {'want':>5} {'result':>6}")
print("-" * 75)

all_pass = True
for query, ctx, expected in cases:
    got = should_cache(query, ctx)
    status = "PASS" if got == expected else "FAIL"
    if got != expected:
        all_pass = False
    print(f"{query:<52} {str(ctx):>4} {str(got):>5} {str(expected):>5} {status}")

print()
if all_pass:
    print("All tests passed — safe to deploy")
else:
    print("TESTS FAILED — do not push until fixed")
