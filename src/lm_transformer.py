import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

import wandb_utils
from lm import EOS, PAD, UNK, WS, build_vocab

SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PYTHIA_14M_HPARAMS = dict(
    hidden_size=128,
    intermediate_size=512,
    num_attention_heads=4,
    num_hidden_layers=6,
    hidden_act="gelu",
    rotary_pct=0.25,
    rotary_emb_base=10000,
    layer_norm_eps=1e-5,
    initializer_range=0.02,
    use_parallel_residual=True,
    tie_word_embeddings=False,
)

PYTHIA_7M_HPARAMS = dict(PYTHIA_14M_HPARAMS, num_hidden_layers=3)

HPARAMS_BY_LANG = {
    "cree": PYTHIA_7M_HPARAMS,
    "guarani": PYTHIA_7M_HPARAMS,
    "inuktitut": PYTHIA_14M_HPARAMS,
}


def flatten_to_ids(tokenized_sentences, stoi):
    ids = []
    for sentence in tokenized_sentences:
        for word_tokens in sentence:
            for tok in word_tokens:
                ids.append(stoi.get(tok, stoi[UNK]))
            ids.append(stoi[WS])
        ids.append(stoi[EOS])
    return ids


def build_sequences(tokenized_sentences, stoi, seq_len):
    ids = flatten_to_ids(tokenized_sentences, stoi)
    n_chunks = len(ids) // seq_len
    if n_chunks == 0:
        return torch.empty((0, seq_len), dtype=torch.long)
    ids = ids[: n_chunks * seq_len]
    return torch.tensor(ids, dtype=torch.long).view(n_chunks, seq_len)


def build_model(vocab_size, eos_id, seq_len=SEQ_LEN, hparams=PYTHIA_14M_HPARAMS):
    config = GPTNeoXConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max(seq_len * 2, 256),
        bos_token_id=eos_id,
        eos_token_id=eos_id,
        **hparams,
    )
    return GPTNeoXForCausalLM(config)


def train_transformer_lm(model, sequences, epochs, batch_size, lr=LEARNING_RATE, log_label="lm", device=DEVICE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = sequences.size(0)
    train_loss_checkpoints = []
    model.train()
    for epoch in tqdm(range(epochs), desc=f"{log_label} epochs", unit="epoch"):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        batch_starts = range(0, n, batch_size)
        for start in tqdm(batch_starts, desc=f"{log_label} epoch {epoch}", unit="batch", leave=False):
            idx = perm[start : start + batch_size]
            batch = sequences[idx].to(device)
            optimizer.zero_grad()
            out = model(input_ids=batch, labels=batch)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            n_batches += 1
        epoch_loss = total_loss / n_batches if n_batches else 0.0
        train_loss_checkpoints.append(epoch_loss)
        wandb_utils.log({f"{log_label}/train_loss": epoch_loss, f"{log_label}/epoch": epoch})
    return train_loss_checkpoints


def evaluate_transformer_cross_entropy(model, sequences, batch_size=BATCH_SIZE, device=DEVICE):
    n = sequences.size(0)
    if n == 0:
        return 0.0
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = sequences[start : start + batch_size].to(device)
            out = model(input_ids=batch, labels=batch)
            total_loss += out.loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches else 0.0


def train_and_evaluate(train_tokenized_sentences, test_tokenized_sentences, lang="inuktitut", variant="lm"):
    hparams = HPARAMS_BY_LANG[lang]
    stoi, itos = build_vocab(train_tokenized_sentences)
    train_sequences = build_sequences(train_tokenized_sentences, stoi, SEQ_LEN)
    test_sequences = build_sequences(test_tokenized_sentences, stoi, SEQ_LEN)

    torch.manual_seed(0)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(0)
    model = build_model(vocab_size=len(itos), eos_id=stoi[EOS], hparams=hparams).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_train_tokens = train_sequences.numel()

    log_label = f"{lang}/{variant}"
    train_loss_checkpoints = train_transformer_lm(
        model, train_sequences, epochs=EPOCHS, batch_size=BATCH_SIZE, log_label=log_label
    )
    test_cross_entropy = evaluate_transformer_cross_entropy(model, test_sequences)
    wandb_utils.log({f"{log_label}/test_cross_entropy": test_cross_entropy})

    arch_name = "Pythia-14M hyperparameters" if hparams is PYTHIA_14M_HPARAMS else "Pythia-14M hyperparameters, depth halved (~7M-scale, not an official Pythia size)"
    return {
        "architecture": f"gpt_neox ({arch_name}, randomly initialized, vocab resized)",
        "vocab_size": len(itos),
        "n_parameters": n_params,
        "n_train_examples": n_train_tokens,
        "tokens_per_parameter": n_train_tokens / n_params if n_params else 0.0,
        "seq_len": SEQ_LEN,
        "train_loss_checkpoints": train_loss_checkpoints,
        "test_cross_entropy": test_cross_entropy,
    }
