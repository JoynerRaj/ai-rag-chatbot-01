"""
Local test for the strict document-only cache gate.
Run from: m:\INTERN\ai-rag-chatbot\django_chat\
  python test_cache_guard.py
"""

# ── exact copy of the real logic from semantic_cache.py ──────────────────────

_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "helo", "sup",
    "bye", "goodbye", "ok", "okay", "thanks", "thank",
}

_MEMORY_TRIGGER_WORDS = {
    "summarize", "summarise", "remind",
}

_MEMORY_WORD_PAIRS = [
    {"previous", "conversation"},
    {"previous", "convo"},
    {"previous", "chat"},
    {"last", "message"},
    {"last", "conversation"},
    {"earlier", "said"},
    {"what", "talk"},
    {"our", "conversation"},
    {"our", "chat"},
    {"tell", "previous"},
]


def should_cache(query, has_document_context):
    if not has_document_context:
        return False
    q = query.strip().lower().rstrip("!?.,:;")
    word_set = set(q.split())
    if len(word_set) < 3:
        return False
    if word_set & _GREETING_WORDS:
        return False
    if word_set & _MEMORY_TRIGGER_WORDS:
        return False
    for pair in _MEMORY_WORD_PAIRS:
        if pair.issubset(word_set):
            return False
    return True


def should_write_cache(query, has_context, user_id):
    return has_context and should_cache(query, has_document_context=True) and user_id is not None


# ── test cases ────────────────────────────────────────────────────────────────
cases = [
    # --- should NOT be cached ---
    # greetings
    ("hi",                                         True,  1, False),
    ("hi how are you",                             True,  1, False),
    ("hello there",                                True,  1, False),
    ("thanks",                                     True,  1, False),
    ("okay thanks",                                True,  1, False),
    # memory / history  (correct spelling)
    ("previous conversation",                      True,  1, False),
    ("what was my previous conversation",          True,  1, False),
    ("tell me about the previous conversation",    True,  1, False),
    ("summarize our chat",                         True,  1, False),
    ("remind me what we said",                     True,  1, False),
    ("what did we talk about",                     True,  1, False),
    # memory / history  (with TYPOS — the main bug)
    ("tell me about the previous converssation",   True,  1, False),  # double s
    ("previous convrsation",                       True,  1, False),  # missing e
    ("previous convo",                             True,  1, False),  # slang
    ("summarise our last chat",                    True,  1, False),  # British spelling
    # off-topic — no document context
    ("what is the weather today",                  False, 1, False),
    ("what is chemistry",                          False, 1, False),
    # no user
    ("what is machine learning",                   True,  None, False),

    # --- SHOULD be cached (real document questions) ---
    ("what is machine learning",                   True,  1, True),
    ("what is python",                             True,  1, True),
    ("explain deep learning",                      True,  1, True),
    ("what is artificial intelligence",            True,  1, True),
    ("explain gradient descent",                   True,  1, True),
    ("what is f1 score",                           True,  1, True),
    ("difference between precision and recall",    True,  1, True),
    ("define supervised learning",                 True,  1, True),
]

print(f"{'Query':<50} {'got':>5} {'want':>5} {'result':>6}")
print("-" * 70)

all_pass = True
for query, has_ctx, uid, expected in cases:
    got = should_write_cache(query, has_ctx, uid)
    status = "PASS" if got == expected else "FAIL"
    if got != expected:
        all_pass = False
    print(f"{query:<50} {str(got):>5} {str(expected):>5} {status}")

print()
if all_pass:
    print("All tests passed - safe to deploy")
else:
    print("TESTS FAILED - do not push until fixed")
