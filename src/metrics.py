def fertility(tokenized_words):
    n_words = len(tokenized_words)
    n_tokens = sum(len(tokens) for tokens in tokenized_words)
    return n_tokens / n_words if n_words else 0.0


def word_to_boundary_positions(units):
    positions = set()
    offset = 0
    for u in units[:-1]:
        offset += len(u)
        positions.add(offset)
    return positions


def consistency_f1(test_words, tokenized_words, reference_lexicon):
    tp = fp = fn = 0
    n_scored = 0
    for word, tokens in zip(test_words, tokenized_words):
        ref_morphs = reference_lexicon.get(word)
        if not ref_morphs:
            continue
        n_scored += 1
        ref_boundaries = word_to_boundary_positions(ref_morphs)
        pred_boundaries = word_to_boundary_positions(tokens)
        tp += len(ref_boundaries & pred_boundaries)
        fp += len(pred_boundaries - ref_boundaries)
        fn += len(ref_boundaries - pred_boundaries)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_scored_words": n_scored,
        "n_test_words": len(test_words),
    }


def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev_row = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cost = 0 if x == y else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = curr_row
    return prev_row[-1]


def morph_edit_distance(test_words, tokenized_words, reference_lexicon):
    total_distance = 0.0
    n_scored = 0
    for word, tokens in zip(test_words, tokenized_words):
        ref_morphs = reference_lexicon.get(word)
        if not ref_morphs:
            continue
        n_scored += 1
        denom = max(len(tokens), len(ref_morphs))
        total_distance += _levenshtein(tokens, ref_morphs) / denom if denom else 0.0

    mean_distance = total_distance / n_scored if n_scored else 0.0
    return {
        "mean_edit_distance": mean_distance,
        "n_scored_words": n_scored,
        "n_test_words": len(test_words),
    }
