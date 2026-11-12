# Changelog

## 1.0.0 — 2026-11-12

First public release.

- Data pipeline: cleaning, masking and de-duplication of 3 sources (`clean_data.py`, `text_utils.py`).
- Train / validation / test split with real-only validation and test sets (`split_data.py`).
- Fine-tuned DistilBERT classifier: scam vs safe (`finetune.py`).
- Results report as an image and a local HTML page (`make_report.py`).
- Live checker for single and pasted multi-line messages (`predict.py`).
- Test results on 1,531 real held-out messages: accuracy 97.8%, precision 98.7%, recall 98.2%, F1 98.4%, false alarm rate 3.0%.
