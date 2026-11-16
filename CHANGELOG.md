# Changelog

## 1.1.0 — 2026-11-16

Better model, same way to use it.

- New training data: 4,000 written hard cases, scams and safe messages (`data/raw/synthetic_v2/`), used for training only.
- Removed 163 IMC reports that are not scams after a hand review (`data/raw/imc2025/label_fixes.csv`).
- Validation and test sets now stay the same between runs (`split_data.py`; `--fresh` makes a new split).
- Masking also treats `.example` addresses as links.
- New **Test B**: 400 unseen messages written by a different AI, with an HTML report (`make_test_b_report.py`).
- Released model is run 4. `predict.py` uses the local run 4 model, or v1.1.0 from Hugging Face.
- Test B: accuracy 75.2% → **91.5%**, scams caught 67% → **89%**, false alarms 17% → **6%**.
- Held-out test set (1,510 real messages): accuracy 98.5%, precision 99.0%, recall 98.8%, F1 98.9%, false alarm rate 2.2%.

## 1.0.0 — 2026-11-12

First public release.

- Data pipeline: cleaning, masking and de-duplication of 3 sources (`clean_data.py`, `text_utils.py`).
- Train / validation / test split with real-only validation and test sets (`split_data.py`).
- Fine-tuned DistilBERT classifier: scam vs safe (`finetune.py`).
- Results report as an image and a local HTML page (`make_report.py`).
- Live checker for single and pasted multi-line messages (`predict.py`).
- Test results on 1,531 real held-out messages: accuracy 97.8%, precision 98.7%, recall 98.2%, F1 98.4%, false alarm rate 3.0%.
