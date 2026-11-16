# Data

All training and test data used by this project, with sources and licenses.

## Folders

| Folder | What | Made by |
|---|---|---|
| `raw/imc2025/` | Real scam (smishing) messages reported by the public, 66 languages | Agarwal et al., IMC 2025 |
| `raw/mishra_soni/` | Real SMS messages labelled ham / spam / smishing | Mishra & Soni, 2022 |
| `raw/synthetic_safe/` | 5,000 written safe service messages (OTP codes, bank alerts, deliveries, bills, travel …) from 21 countries | This project (AI-generated, then checked) |
| `raw/synthetic_v2/` | 4,000 written hard cases, 2,000 scam and 2,000 safe, about 180 message types from 50 countries | This project (AI-generated from a generic prompt, then checked) |
| `raw/imc2025/label_fixes.csv` | 163 IMC reports removed after a hand review (not scams, or too vague to judge) | This project |
| `test_b/` | Test B: 400 unseen messages (199 scam, 201 safe) for the honest score. Never used for training | Written by a different AI from a generic prompt; brand notes it left in the text were removed |
| `manual_tests/` | 60 hand-made hard messages, and the review sheet behind `label_fixes.csv` | This project |
| `processed/clean.csv` | All sources cleaned and combined | `clean_data.py` |
| `splits/` | `train.csv`, `val.csv`, `test.csv` | `split_data.py` |

## How the data is built

1. **Pick:** English messages only. Columns kept: text and label.
2. **Labels:** `ham` → `safe`; `smishing` and all IMC reports → `scam`. Spam (ads) is **dropped**.
3. **Fix text:** repair broken characters and HTML codes.
4. **Mask:** links → `<URL>`, phone numbers → `<PHONE_NUMBER>`, emails → `<EMAIL_ADDRESS>`, long numbers → `<NUMBER>`. The same rules apply to every source and to live messages (`text_utils.py`), so the model can't learn a shortcut from one source's format.
5. **Remove junk:** messages with fewer than 3 real words.
6. **Fix labels:** remove IMC reports that are clearly real messages (real OTP codes, fraud warnings), plus the hand-reviewed list in `raw/imc2025/label_fixes.csv`. Rule: the label must be clear from the text alone.
7. **Remove duplicates:** exact copies, same-pattern copies, and near-duplicates (90%+ similar).
8. **Split:** validation and test come **only from real messages**. Synthetic messages are used for training only. Once made, validation and test stay the same between runs (`split_data.py --fresh` makes a new split).

## Final counts

| Split | Scam | Safe |
|---|---|---|
| Train | 10,411 | 10,643 |
| Validation | 1,043 | 465 |
| Test | 1,045 | 465 |

## Known issues

- Some IMC reports are still real messages rather than scams (users report messages they are unsure about). Only the ones the model doubted were reviewed by hand.
- Real safe messages in validation/test are personal chats only. There are no real safe *service* messages yet.
- The first synthetic safe set often starts with a brand name ("Uber: …"). The v2 set balances links, phone numbers, lengths and brand prefixes across both labels.
- Test B has no long or multi-line messages.

## Sources and licenses

**IMC 2025 smishing dataset** (CC BY 4.0) — https://github.com/reportsmishing/Smishing-Dataset-IMC25

> Sharad Agarwal, Antonis Papasavva, Guillermo Suarez-Tangil, Marie Vasek. *Fishing for Smishing: Understanding SMS Phishing Infrastructure and Strategies by Mining Public User Reports.* ACM Internet Measurement Conference (IMC '25), 2025. https://doi.org/10.1145/3730567.3764431

**SMS Phishing Dataset for Machine Learning and Pattern Recognition** (CC BY 4.0) — https://data.mendeley.com/datasets/f45bkkt8pr/1

> Sandhya Mishra, Devpriya Soni. *SMS Phishing Dataset for Machine Learning and Pattern Recognition.* Mendeley Data, V1, 2022. https://doi.org/10.17632/f45bkkt8pr.1

**Synthetic safe service messages and synthetic hard cases v2** (CC BY 4.0, this project) — written with an AI model for this project and checked for duplicates, reused templates, label shortcuts and overlap with the test sets. Brands and details are fictional examples.

**Test B** (CC BY 4.0, this project) — written with a different AI model from a generic prompt. Brands, names, numbers and links are fictional.

Files in `processed/` and `splits/` are derived from the sources above and keep the CC BY 4.0 license. Please credit the original authors when you use them.
