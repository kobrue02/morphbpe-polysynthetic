import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from lm import EOS, WS, build_vocab
from lm_transformer import build_model, build_sequences, evaluate_transformer_cross_entropy, flatten_to_ids, train_transformer_lm


def test_flatten_to_ids_marks_word_boundaries_and_eos():
    stoi, itos = build_vocab([[["a", "b"], ["c"]]])
    ids = flatten_to_ids([[["a", "b"], ["c"]]], stoi)
    assert ids == [stoi["a"], stoi["b"], stoi[WS], stoi["c"], stoi[WS], stoi[EOS]]


def test_build_sequences_shape():
    sentences = [[["a", "b"], ["c"]]] * 20
    stoi, itos = build_vocab(sentences)
    seqs = build_sequences(sentences, stoi, seq_len=4)
    assert seqs.dtype == torch.long
    assert seqs.shape[1] == 4
    total_ids = len(flatten_to_ids(sentences, stoi))
    assert seqs.shape[0] == total_ids // 4


def test_training_reduces_loss_on_a_repeated_toy_pattern():
    sentence = [["a", "b"], ["c"], ["a", "b"], ["d"]]
    sentences = [sentence] * 100

    stoi, itos = build_vocab(sentences)
    seqs = build_sequences(sentences, stoi, seq_len=8)
    assert seqs.shape[0] > 0

    torch.manual_seed(0)
    model = build_model(vocab_size=len(itos), eos_id=stoi[EOS], seq_len=8)

    first_epoch_loss = train_transformer_lm(model, seqs, epochs=1, batch_size=8)[0]
    later_losses = train_transformer_lm(model, seqs, epochs=15, batch_size=8)
    assert later_losses[-1] < first_epoch_loss, (first_epoch_loss, later_losses)

    ce = evaluate_transformer_cross_entropy(model, seqs, batch_size=8)
    assert ce < first_epoch_loss


if __name__ == "__main__":
    test_flatten_to_ids_marks_word_boundaries_and_eos()
    test_build_sequences_shape()
    test_training_reduces_loss_on_a_repeated_toy_pattern()
    print("lm_transformer.py tests passed")
