"""
Fine-tune DistilBERT for scam vs safe (Run 2).

Data (made by clean_data.py -> split_data.py):
  data/splits/train.csv   real 80% + all synthetic   -> learning
  data/splits/val.csv     real only                  -> checking during training / early stop
  data/splits/test.csv    real only, LOCKED          -> final score, used once at the end

Outputs:
  models/scam-distilbert-run4/           best fine-tuned model + tokenizer
  results/run4/history.json              losses, scores, settings
  results/run4/test_mistakes.csv         every wrong test message
  results/run4/results.png               one combined results graph (needs matplotlib)

Run: .venv/bin/python finetune.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

RUN = "run4"
# Pretrained base model: use a local copy if present, else download from Hugging Face
BASE_MODEL = ("models/distilbert-base-uncased" if Path("models/distilbert-base-uncased").exists()
              else "distilbert/distilbert-base-uncased")
OUT_DIR = Path(f"models/scam-distilbert-{RUN}")  # fine-tuned model is saved here
RESULTS = Path(f"results/{RUN}")

MAX_LEN = 128        # max tokens per message (SMS are short)
BATCH_SIZE = 16      # messages per step
EPOCHS = 4           # max passes over the training data
LR = 2e-5            # very small learning rate: adjust, don't overwrite
PATIENCE = 2         # stop if val loss doesn't improve 2 epochs in a row
EVAL_EVERY = 100     # check validation loss every N steps (for the graph)
SEED = 42

torch.manual_seed(SEED)

# Use the Mac GPU if available, else NVIDIA GPU, else CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")

# ---------------------------------------------------------------
# 1. Load the ready-made splits
# ---------------------------------------------------------------
def load(name):
    frame = pd.read_csv(f"data/splits/{name}.csv", keep_default_na=False)
    frame["y"] = (frame["label"] == "scam").astype(int)   # scam = 1, safe = 0
    return frame


train_df, val_df, test_df = load("train"), load("val"), load("test")
for name, frame in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    counts = frame["label"].value_counts().to_dict()
    print(f"{name:5s}: {len(frame):6d} rows | scam {counts.get('scam', 0)} | safe {counts.get('safe', 0)}")

# ---------------------------------------------------------------
# 2. Tokenizer + model (pretrained body + new 2-class head)
# ---------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Our masking tags become single tokens instead of being split into pieces
tokenizer.add_tokens(["<URL>", "<PHONE_NUMBER>", "<EMAIL_ADDRESS>", "<NUMBER>"])
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=2, id2label={0: "safe", 1: "scam"}, label2id={"safe": 0, "scam": 1}
)
model.resize_token_embeddings(len(tokenizer))   # make room for the 4 new tag tokens
model.to(device)


def batches(frame, shuffle=False):
    """Yield tokenized batches: input_ids, attention_mask, labels."""
    idx = torch.randperm(len(frame)).tolist() if shuffle else list(range(len(frame)))
    texts, labels = frame["text"].tolist(), frame["y"].tolist()
    for i in range(0, len(idx), BATCH_SIZE):
        b = idx[i:i + BATCH_SIZE]
        enc = tokenizer([texts[j] for j in b], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[j] for j in b])
        yield {k: v.to(device) for k, v in enc.items()}


def evaluate(frame):
    """Return average loss, predictions, and scam probabilities (no learning)."""
    model.eval()
    total_loss, preds, probs = 0.0, [], []
    with torch.no_grad():
        for batch in batches(frame):
            out = model(**batch)
            total_loss += out.loss.item() * len(batch["labels"])
            p = torch.softmax(out.logits, dim=-1)[:, 1]
            probs += p.cpu().tolist()
            preds += (p > 0.5).int().cpu().tolist()
    return total_loss / len(frame), preds, probs


# ---------------------------------------------------------------
# 3. Optimizer + learning-rate schedule
#    Warm up for the first 10% of steps, then slowly lower the LR.
# ---------------------------------------------------------------
steps_per_epoch = (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE
total_steps = steps_per_epoch * EPOCHS
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

# ---------------------------------------------------------------
# 4. Fine-tuning loop with early stopping
# ---------------------------------------------------------------
history = {"run": RUN, "settings": {"base_model": BASE_MODEL, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
                                    "epochs": EPOCHS, "lr": LR, "patience": PATIENCE, "seed": SEED},
           "steps_per_epoch": steps_per_epoch, "points": [], "epochs": []}

best_val, bad = float("inf"), 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    start, running, window = time.time(), 0.0, []
    for step, batch in enumerate(batches(train_df, shuffle=True), 1):
        out = model(**batch)          # forward pass (loss is computed inside)
        optimizer.zero_grad()
        out.loss.backward()           # backprop
        optimizer.step()              # update weights
        scheduler.step()              # adjust learning rate
        running += out.loss.item()
        window.append(out.loss.item())
        if step % EVAL_EVERY == 0:
            v_loss, _, _ = evaluate(val_df)
            model.train()
            history["points"].append({"step": (epoch - 1) * steps_per_epoch + step,
                                      "train_loss": sum(window) / len(window), "val_loss": v_loss})
            print(f"  epoch {epoch} step {step}/{steps_per_epoch} | train loss {sum(window) / len(window):.4f} | val loss {v_loss:.4f}")
            window = []

    train_loss = running / steps_per_epoch
    val_loss, _, _ = evaluate(val_df)
    elapsed = time.time() - start
    is_best = val_loss < best_val
    history["epochs"].append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                              "seconds": round(elapsed), "best": is_best})
    print(f"Epoch {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {elapsed:.0f}s")

    if is_best:
        best_val, bad = val_loss, 0
        model.save_pretrained(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print(f"  saved best model -> {OUT_DIR}")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("Early stop: val loss stopped improving.")
            break

# ---------------------------------------------------------------
# 5. Test the best model ONCE on the locked test set
# ---------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).to(device)
_, preds, probs = evaluate(test_df)
y = test_df["y"].tolist()
tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
scores = {
    "accuracy": accuracy_score(y, preds),
    "precision": precision_score(y, preds, zero_division=0),
    "recall": recall_score(y, preds),                # share of scams caught
    "f1": f1_score(y, preds),
    "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,   # share of safe messages wrongly flagged
}
history["test"] = {**scores, "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn), "rows": len(test_df)}

print(f"\n=== Test set ({len(test_df)} real messages) ===")
for k, v in scores.items():
    print(f"{k:17s}: {v:.3f}")
print(f"Scams caught {tp} of {tp + fn} | Safe messages wrongly flagged {fp} of {fp + tn}")

# Save every mistake for review
test_df = test_df.assign(scam_prob=probs, pred=["scam" if p else "safe" for p in preds])
mistakes = test_df[test_df["pred"] != test_df["label"]]
RESULTS.mkdir(parents=True, exist_ok=True)
mistakes[["text", "label", "pred", "scam_prob", "source", "scam_type"]].sort_values("scam_prob").to_csv(
    RESULTS / "test_mistakes.csv", index=False)
with open(RESULTS / "history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\nSaved {len(mistakes)} mistakes -> {RESULTS / 'test_mistakes.csv'}")
print(f"Saved history -> {RESULTS / 'history.json'}")

# ---------------------------------------------------------------
# 6. One combined results graph (local image)
# ---------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed: skipped the graph (pip install matplotlib)")
else:
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 0.9]})
    fig.suptitle(f"Scam detector - DistilBERT fine-tune {RUN}", fontsize=13)

    # Loss curves
    xs = [p["step"] / steps_per_epoch for p in history["points"]]
    a1.plot(xs, [p["train_loss"] for p in history["points"]], marker="o", ms=3, lw=2, label="Train loss")
    a1.plot(xs, [p["val_loss"] for p in history["points"]], marker="o", ms=3, lw=2, label="Validation loss")
    best = [e["epoch"] for e in history["epochs"] if e["best"]][-1]
    a1.axvline(best, ls="--", color="grey", lw=1)
    a1.text(best, a1.get_ylim()[1] * 0.95, f" best model (epoch {best})", color="grey", va="top")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Loss (lower is better)")
    a1.set_title("Loss during training")
    a1.grid(alpha=0.3)
    a1.legend()

    # Test scores
    names = ["Accuracy", "Precision", "Recall", "F1", "False alarm"]
    vals = [scores["accuracy"], scores["precision"], scores["recall"], scores["f1"], scores["false_alarm_rate"]]
    bars = a2.bar(names, vals, color=["#4C72B0"] * 4 + ["#C44E52"])
    a2.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=2)
    a2.set_ylim(0, 1.1)
    a2.set_title(f"Test scores ({len(test_df)} real messages)")
    a2.tick_params(axis="x", rotation=20)
    a2.grid(axis="y", alpha=0.3)

    # Confusion matrix
    grid = [[tn, fp], [fn, tp]]
    a3.imshow(grid, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a3.text(j, i, grid[i][j], ha="center", va="center", fontsize=14,
                    color="white" if grid[i][j] > max(tn, fp, fn, tp) / 2 else "black")
    a3.set_xticks([0, 1], ["Said safe", "Said scam"])
    a3.set_yticks([0, 1], ["Really safe", "Really scam"])
    a3.set_title("Right vs wrong")

    fig.tight_layout()
    fig.savefig(RESULTS / "results.png", dpi=130)
    print(f"Saved graph -> {RESULTS / 'results.png'}")
