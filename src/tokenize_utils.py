import re

_LEADING_TRAILING_PUNCT = re.compile(
    r"^[^\w]+|[^\w]+$", re.UNICODE
)


_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def clean_word(token):
    cleaned = _LEADING_TRAILING_PUNCT.sub("", token)
    if not _HAS_LETTER.search(cleaned):
        return ""
    return cleaned


def tokenize_line(line):
    return [w for w in (clean_word(tok) for tok in line.split()) if w]
