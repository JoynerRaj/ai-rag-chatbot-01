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
    "conversation", "conversations",
}

_MEMORY_WORD_PAIRS = [
    {"previous", "convo"},
    {"previous", "chat"},
    {"last", "message"},
    {"last", "conversation"},
    {"earlier", "said"},
    {"our", "conversation"},
    {"our", "chat"},
    {"tell", "previous"},
]

_PREVIOUS_PREFIXES = {"previ", "prvio", "previo"}


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
    if any(w[:5] in _PREVIOUS_PREFIXES for w in word_set if len(w) >= 5):
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
    ("hi",                                          True,  1, False),
    ("hi how are you",                              True,  1, False),
    ("hello there",                                 True,  1, False),
    ("thanks",                                      True,  1, False),
    # memory - exact
    ("what is the previous conversation",           True,  1, False),
    ("tell me about the previous conversation",     True,  1, False),
    ("summarize our chat",                          True,  1, False),
    ("remind me what we said",                      True,  1, False),
    # memory - TYPOS (the main bug cases)
    ("what is the prvious conversation",            True,  1, False),  # typo: prvious
    ("tell me about the previous converssation",    True,  1, False),  # typo: converssation
    ("what is the previous convrsation",            True,  1, False),  # typo: convrsation
    ("previous convo",                              True,  1, False),  # slang
    ("summarise our last chat",                     True,  1, False),  # British spelling
    # off-topic (no document context from Pinecone)
    ("what is the weather today",                   False, 1, False),
    ("what is chemistry",                           False, 1, False),
    # no user logged in
    ("what is machine learning",                    True,  None, False),

    # --- SHOULD be cached (genuine document questions) ---
    ("what is machine learning",                    True,  1, True),
    ("what is python",                              True,  1, True),
    ("explain deep learning",                       True,  1, True),
    ("what is artificial intelligence",             True,  1, True),
    ("explain gradient descent",                    True,  1, True),
    ("what is f1 score",                            True,  1, True),
    ("difference between precision and recall",     True,  1, True),
    ("define supervised learning",                  True,  1, True),
    ("what are the types of machine learning",      True,  1, True),
    ("how does backpropagation work",               True,  1, True),
]

print(f"{'Query':<52} {'got':>5} {'want':>5} {'result':>6}")
print("-" * 72)

all_pass = True
for query, has_ctx, uid, expected in cases:
    got = should_write_cache(query, has_ctx, uid)
    status = "PASS" if got == expected else "FAIL"
    if got != expected:
        all_pass = False
    print(f"{query:<52} {str(got):>5} {str(expected):>5} {status}")

print()
if all_pass:
    print("All tests passed - safe to deploy")
else:
    print("TESTS FAILED - do not push until fixed")
