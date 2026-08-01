import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from lm import (
    DIMS,
    TinyNGramLM,
    build_examples,
    build_vocab,
    evaluate_cross_entropy,
    sentence_to_ids,
    train_lm,
)


def test_build_vocab_and_examples():
    sentences = [[["a", "b"], ["c"]], [["a", "b"], ["d"]]]
    stoi, itos = build_vocab(sentences)
    for tok in ("a", "b", "c", "d", "<pad>", "<ws>", "<eos>", "<unk>"):
        assert tok in stoi
    contexts, targets = build_examples(sentences, stoi, context_size=2)
    assert contexts.shape[0] == targets.shape[0]
    assert contexts.shape[1] == 2


def test_sentence_to_ids_marks_word_boundaries_and_eos():
    stoi, itos = build_vocab([[["a", "b"], ["c"]]])
    ids = sentence_to_ids([["a", "b"], ["c"]], stoi, context_size=2)
    assert ids == [stoi["<pad>"], stoi["<pad>"], stoi["a"], stoi["b"], stoi["<ws>"], stoi["c"], stoi["<ws>"], stoi["<eos>"]]


def test_training_reduces_loss_on_a_repeated_toy_pattern():
    sentence = [["a", "b"], ["c"], ["a", "b"], ["d"]]
    sentences = [sentence] * 50

    stoi, itos = build_vocab(sentences)
    context_size = 2
    contexts, targets = build_examples(sentences, stoi, context_size)

    torch.manual_seed(0)
    model = TinyNGramLM(len(itos), embedding_dim=8, hidden_dim=16, context_size=context_size)

    first_epoch_loss = train_lm(model, contexts, targets, epochs=1, batch_size=32)[0]
    later_losses = train_lm(model, contexts, targets, epochs=20, batch_size=32)
    assert later_losses[-1] < first_epoch_loss, (first_epoch_loss, later_losses)

    ce = evaluate_cross_entropy(model, contexts, targets)
    assert ce < first_epoch_loss


def test_dims_defined_for_both_languages():
    assert "cree" in DIMS and "inuktitut" in DIMS
    for lang, d in DIMS.items():
        assert d["embedding_dim"] > 0 and d["hidden_dim"] > 0 and d["context_size"] > 0


if __name__ == "__main__":
    test_build_vocab_and_examples()
    test_sentence_to_ids_marks_word_boundaries_and_eos()
    test_training_reduces_loss_on_a_repeated_toy_pattern()
    test_dims_defined_for_both_languages()
    print("lm.py tests passed")
