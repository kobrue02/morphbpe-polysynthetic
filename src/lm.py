import torch
import torch.nn as nn
from tqdm import tqdm

import wandb_utils

PAD = "<pad>"
WS = "<ws>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, WS, EOS, UNK]


def build_vocab(tokenized_sentences):
    vocab = set()
    for sentence in tokenized_sentences:
        for word_tokens in sentence:
            vocab.update(word_tokens)
    itos = SPECIAL_TOKENS + sorted(vocab)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos


def sentence_to_ids(sentence_tokens, stoi, context_size):
    ids = [stoi[PAD]] * context_size
    for word_tokens in sentence_tokens:
        for tok in word_tokens:
            ids.append(stoi.get(tok, stoi[UNK]))
        ids.append(stoi[WS])
    ids.append(stoi[EOS])
    return ids


def build_examples(tokenized_sentences, stoi, context_size):
    contexts = []
    targets = []
    for sentence in tokenized_sentences:
        if not sentence:
            continue
        ids = sentence_to_ids(sentence, stoi, context_size)
        for i in range(context_size, len(ids)):
            contexts.append(ids[i - context_size : i])
            targets.append(ids[i])
    if not contexts:
        return torch.empty((0, context_size), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class TinyNGramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, context_ids):
        batch_size = context_ids.size(0)
        emb = self.embedding(context_ids).view(batch_size, -1)
        h = torch.tanh(self.hidden(emb))
        proj = self.project(h)
        return proj @ self.embedding.weight.T

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_lm(model, contexts, targets, epochs, batch_size, lr=1e-3, log_label="lm"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = contexts.size(0)
    train_loss_checkpoints = []
    model.train()
    for epoch in tqdm(range(epochs), desc=f"{log_label} epochs", unit="epoch"):
        perm = torch.randperm(n)
        total_loss = 0.0
        batch_starts = range(0, n, batch_size)
        for start in tqdm(batch_starts, desc=f"{log_label} epoch {epoch}", unit="batch", leave=False):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(contexts[idx])
            loss = loss_fn(logits, targets[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        epoch_loss = total_loss / n if n else 0.0
        train_loss_checkpoints.append(epoch_loss)
        wandb_utils.log({f"{log_label}/train_loss": epoch_loss, f"{log_label}/epoch": epoch})
    return train_loss_checkpoints


def evaluate_cross_entropy(model, contexts, targets, batch_size=512):
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    n = contexts.size(0)
    if n == 0:
        return 0.0
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            logits = model(contexts[start : start + batch_size])
            total_loss += loss_fn(logits, targets[start : start + batch_size]).item()
    return total_loss / n


DIMS = {
    "cree": {"embedding_dim": 16, "hidden_dim": 32, "context_size": 3},
    "inuktitut": {"embedding_dim": 32, "hidden_dim": 64, "context_size": 4},
}
EPOCHS = 20
BATCH_SIZE = 256


def train_and_evaluate(lang, train_tokenized_sentences, test_tokenized_sentences, variant="lm"):
    dims = DIMS[lang]
    stoi, itos = build_vocab(train_tokenized_sentences)
    train_contexts, train_targets = build_examples(train_tokenized_sentences, stoi, dims["context_size"])
    test_contexts, test_targets = build_examples(test_tokenized_sentences, stoi, dims["context_size"])

    torch.manual_seed(0)
    model = TinyNGramLM(len(itos), dims["embedding_dim"], dims["hidden_dim"], dims["context_size"])
    n_params = model.count_parameters()
    n_train_examples = train_contexts.size(0)

    log_label = f"{lang}/{variant}"
    train_loss_checkpoints = train_lm(
        model, train_contexts, train_targets, epochs=EPOCHS, batch_size=BATCH_SIZE, log_label=log_label
    )
    test_cross_entropy = evaluate_cross_entropy(model, test_contexts, test_targets)
    wandb_utils.log({f"{log_label}/test_cross_entropy": test_cross_entropy})

    return {
        "vocab_size": len(itos),
        "n_parameters": n_params,
        "n_train_examples": n_train_examples,
        "tokens_per_parameter": n_train_examples / n_params if n_params else 0.0,
        "train_loss_checkpoints": train_loss_checkpoints,
        "test_cross_entropy": test_cross_entropy,
    }
