# Scam Shield

An open-source model that reads a text message and says **SCAM** or **SAFE**.

It is a fine-tuned [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) model trained on real scam reports from around the world, real everyday SMS, and written examples of both scams and safe messages (OTP codes, bank alerts, deliveries, family chats and more). It runs on a normal laptop, so messages never need to leave your device.

**Version 1.1.0** · English only · Code: MIT · Data: CC BY 4.0

**Model:** [rasheq/scam-shield-distilbert](https://huggingface.co/rasheq/scam-shield-distilbert) · **Dataset:** [rasheq/scam-shield-sms](https://huggingface.co/datasets/rasheq/scam-shield-sms)

```
$ python predict.py "USPS: Your package is on hold due to an incomplete address. Update within 12 hours: usps-redeliver.top/track"

🚨 SCAM  (100% scam)
  Model saw: USPS: Your package is on hold due to an incomplete address. Update within 12 hours: <URL>
```

## Results

### Test B: 400 new messages (the honest score)

Written by a different AI from a generic prompt, never used for training or tuning: 199 scams and 201 safe messages from 18 countries, both easy and hard.

| Model | Accuracy | Scams caught | False alarms |
|---|---|---|---|
| v1.0.0 | 75.2% | 67% (134 of 199) | 17% (34 of 201) |
| **v1.1.0** | **91.5%** | **89% (178 of 199)** | **6% (13 of 201)** |

| v1.1.0 on | Accuracy | Scams caught | False alarms |
|---|---|---|---|
| Easy messages | 95.7% | 94% | 3% |
| Hard messages | 81.4% | 79% | 16% |
| Messages with a link | 88.6% | 91% | 18% |
| Messages without a link | 93.1% | 88% | 4% |

Full report: [`results/test_b/results.html`](results/test_b/results.html) (open in a browser) · every score: [`results/test_b_scores.csv`](results/test_b_scores.csv)

### Held-out test set: 1,510 real messages

| Metric | Score | Meaning |
|---|---|---|
| Accuracy | 98.5% | All answers right |
| Precision | 99.0% | When it says scam, how often it is right |
| Recall | 98.8% | Share of scams caught (1,032 of 1,045) |
| F1 | 98.9% | Balance of precision and recall |
| False alarm rate | 2.2% | Safe messages wrongly flagged (10 of 465) |

![Run 4 results](results/run4/results.png)

Full report: [`results/run4/results.html`](results/run4/results.html) · every mistake: [`results/run4/test_mistakes.csv`](results/run4/test_mistakes.csv)

### Read this before trusting the numbers

- The held-out test set has real scams but only **personal chats** as safe messages, so its scores are higher than real life. **Test B (about 90%) is closer to what to expect.**
- On a hand-made set of 60 harder messages the model got **55 of 60** right (v1.0.0: 51 of 60). This set was used to find error patterns, so treat it as a rough check only.

## Quick start

Needs Python 3.10+ (tested on 3.14, macOS, Apple Silicon).

```bash
git clone https://github.com/Mohamedrasheq/scam-shield.git
cd scam-shield
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The trained model is not stored in this repo (it is 256 MB). `predict.py` downloads it automatically from Hugging Face the first time you run it, or you can train it yourself (below, about 15 minutes on a laptop GPU).

## Check messages

```bash
# One message
python predict.py "Hi Mum, I dropped my phone, this is my new number. Can you send me \$300 today?"

# Type or paste messages one by one (multi-line pastes work). Type q to quit.
python predict.py
```

You see the verdict, how sure the model is, and the message as the model saw it after masking (links, phone numbers and long numbers are replaced with tags). Messages under 3 words are reported as too short to judge.

## Train it yourself

```bash
python clean_data.py   # data/raw/      -> data/processed/clean.csv
python split_data.py   # clean.csv      -> data/splits/train|val|test.csv
python finetune.py     # splits         -> models/scam-distilbert-run4/ + results/run4/
python make_report.py run4   # results  -> results/run4/results.html
python make_test_b_report.py # Test B   -> results/test_b/results.html (needs results/test_b_scores.csv)
```

`split_data.py` keeps the existing validation and test sets, so scores stay comparable between runs; use `--fresh` for a new random split. `finetune.py` downloads the base model from Hugging Face if it is not in `models/`. It uses the Apple GPU (MPS), an NVIDIA GPU (CUDA) or the CPU, whichever is available. Peak memory is about 7 GB.

| Setting | Value |
|---|---|
| Base model | distilbert-base-uncased (67M parameters, all trained) |
| Max length | 128 tokens |
| Batch size | 16 |
| Learning rate | 2e-5, 10% warm-up, then linear decay |
| Epochs | up to 4, early stop after 2 epochs without validation improvement |
| Best epoch (run 4) | 2 |

## How it works

1. **Clean** (`text_utils.py`): fix broken characters, then mask links, phone numbers, emails and long numbers, so `www.bank-verify.top` and `bit.ly/x` both become `<URL>`.
2. **Read:** DistilBERT turns the message into a 768-number summary of its meaning.
3. **Decide:** a small classification head turns that summary into two scores, safe and scam. Above 50% scam means **SCAM**.

## Data

| Source | Messages used | Label | License |
|---|---|---|---|
| [IMC 2025 smishing reports](https://github.com/reportsmishing/Smishing-Dataset-IMC25) | 10,167 | scam | CC BY 4.0 |
| [Mishra & Soni SMS phishing](https://data.mendeley.com/datasets/f45bkkt8pr/1) | 4,650 safe, 333 scam | both | CC BY 4.0 |
| Synthetic safe service messages (this project) | 4,953 | safe | CC BY 4.0 |
| Synthetic hard cases, v2 (this project) | 1,970 safe, 1,999 scam | both | CC BY 4.0 |

Synthetic messages are used for training only. 163 IMC reports that are not scams (real service messages, comments about scams, or too vague to judge) were removed by hand.

Spam (ads) is left out on purpose: the model answers "is this a scam?", not "is this unwanted?". Details on cleaning, masking, de-duplication and the split: [`data/README.md`](data/README.md).

## Known limitations

- **English only**, trained mostly on SMS-style messages. Emails and long chat messages are outside what it has learned.
- **It can't see link addresses.** Every link becomes `<URL>`, so a fake-looking domain (`bank-unlock-verify.top`) and a real one look the same. Scams whose only clue is the link can slip through.
- **Long, multi-line real notices** (bank statements, sign-in alerts) can still be flagged as scams.
- **Real alerts with links or money** (receipts, shop promos, fines, charity appeals) are the most common false alarms.
- **Some scams slip through:** slow "wrong number" chats that turn into investment pitches, and some bank-alert scams that ask you to call a number.
- **Sensitive to wording:** small changes in wording can flip the answer.
- **Label noise:** some public scam reports are still real messages that users were unsure about.
- **It is a helper, not a guarantee.** Never treat a SAFE result as proof that a message is safe.

## Roadmap

- Keep the link's domain name instead of a plain `<URL>` tag (v2.0.0).
- Real safe service messages and long messages in the test set.
- A fresh test set for every release.
- More languages (multilingual base model).
- Smaller model for browsers and phones.

## Project layout

```
text_utils.py     shared cleaning and masking (training and live checks)
clean_data.py     build data/processed/clean.csv from data/raw/
split_data.py     build data/splits/ (train / val / test)
finetune.py       fine-tune DistilBERT, test once, save results
make_report.py    build the HTML results page for a run
make_test_b_report.py  build the HTML results page for Test B
predict.py        check messages live
data/             datasets and their licenses (see data/README.md)
data/test_b/      Test B: 400 unseen messages for the honest score
results/run4/     scores, mistakes and graphs for the released model (v1.1.0)
results/test_b/   Test B report comparing runs 2, 3 and 4
```

## Credits

- **Datasets:** Agarwal, Papasavva, Suarez-Tangil and Vasek (IMC 2025); Mishra and Soni (Mendeley Data, 2022). Full citations in [`data/README.md`](data/README.md).
- **Base model:** DistilBERT by Hugging Face (Sanh et al., 2019), Apache-2.0.

## License

Code: [MIT](LICENSE). Data: CC BY 4.0 (credit the original authors). Base model: Apache-2.0.
