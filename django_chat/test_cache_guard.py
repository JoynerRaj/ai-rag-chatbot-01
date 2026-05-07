"""
Local test for the cache guard logic.
Run from: m:\INTERN\ai-rag-chatbot\django_chat\
  python test_cache_guard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── replicate the exact constants from semantic_cache.py ──────────────────────

_NO_CACHE_PHRASES = {
    "hi", "hello", "hey", "hii", "helo",
    "good morning", "good evening", "good afternoon",
    "how are you", "what's up", "whats up", "sup",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye",
    "great", "nice", "cool", "awesome", "got it",
}

_HISTORY_KEYWORDS = (
    "previous conversation", "last message", "what did we talk",
    "summarize our", "summarize the chat", "earlier you said",
    "what was my", "remind me", "our conversation",
)

# ── OLD implementation (broken) ───────────────────────────────────────────────

def should_cache_OLD(query, has_document_context):
    if not has_document_context:
        return False
    q = query.strip().lower().rstrip("!?.,:;")
    if q in _NO_CACHE_PHRASES:
        return False
    for pattern in _HISTORY_KEYWORDS:
        if pattern in q:
            return False
    return True


# ── NEW implementation ────────────────────────────────────────────────────────

# Greeting words that make a query non-cacheable even if embedded in a longer string
_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "helo", "sup",
    "bye", "goodbye", "ok", "okay",
}

# Short questions below this word count are almost always small talk
_MIN_WORDS_TO_CACHE = 3

def should_cache_NEW(query, has_document_context):
    if not has_document_context:
        return False

    q = query.strip().lower().rstrip("!?.,:;")

    # Exact full-phrase match
    if q in _NO_CACHE_PHRASES:
        return False

    # Any word in the query is a known greeting word → skip
    words = q.split()
    if any(w in _GREETING_WORDS for w in words):
        return False

    # Very short queries are overwhelmingly small talk
    if len(words) < _MIN_WORDS_TO_CACHE:
        return False

    # History / summary references
    for pattern in _HISTORY_KEYWORDS:
        if pattern in q:
            return False

    return True


# ── Test cases ────────────────────────────────────────────────────────────────

cases = [
    # (query, has_doc_context, should_be_cached)
    ("hi",                               True,  False),
    ("hi how are you",                   True,  False),  # BUG in old code — stored!
    ("hii",                              True,  False),
    ("hello there",                      True,  False),
    ("hey what is machine learning",     True,  False),  # starts with greeting
    ("thanks",                           True,  False),
    ("okay thanks",                      True,  False),
    ("what is my previous conversation", True,  False),
    ("summarize our chat",               True,  False),
    ("what is f1 score",                 True,  True),
    ("explain gradient descent",         True,  True),
    ("what is the difference between precision and recall", True, True),
    # no doc context → never cache
    ("what is chemistry",                False, False),
    ("explain gravity",                  False, False),
]

print(f"{'Query':<48} {'old':>5} {'new':>5} {'want':>5} {'pass':>5}")
print("-" * 75)

all_pass = True
for query, has_doc, expected in cases:
    old = should_cache_OLD(query, has_doc)
    new = should_cache_NEW(query, has_doc)
    ok  = "PASS" if new == expected else "FAIL"
    if new != expected:
        all_pass = False
    print(f"{query:<48} {str(old):>5} {str(new):>5} {str(expected):>5} {ok}")

print()
if all_pass:
    print("All tests passed — safe to update semantic_cache.py")
else:
    print("Some tests FAILED — fix the logic before pushing")
