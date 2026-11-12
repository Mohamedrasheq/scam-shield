# Data

All training and test data used by this project, with sources and licenses.

## Folders

| Folder | What | Made by |
|---|---|---|
| `raw/imc2025/` | Real scam (smishing) messages reported by the public, 66 languages | Agarwal et al., IMC 2025 |
| `raw/mishra_soni/` | Real SMS messages labelled ham / spam / smishing | Mishra & Soni, 2022 |
| `raw/synthetic_safe/` | 5,000 written safe service messages (OTP codes, bank alerts, deliveries, bills, travel …) from 21 countries | This project (AI-generated, then checked) |
| `processed/clean.csv` | All sources cleaned and combined | `clean_data.py` |
| `splits/` | `train.csv`, `val.csv`, `test.csv` | `split_data.py` |

## How the data is built

1. **Pick:** English messages only. Columns kept: text and label.
2. **Labels:** `ham` → `safe`; `smishing` and all IMC reports → `scam`. Spam (ads) is **dropped**.
3. **Fix text:** repair broken characters and HTML codes.
4. **Mask:** links → `<URL>`, phone numbers → `<PHONE_NUMBER>`, emails → `<EMAIL_ADDRESS>`, long numbers → `<NUMBER>`. The same rules apply to every source and to live messages (`text_utils.py`), so the model can't learn a shortcut from one source's format.
5. **Remove junk:** messages with fewer than 3 real words.
6. **Fix labels:** remove IMC reports that are clearly real messages (real OTP codes, fraud warnings).
7. **Remove duplicates:** exact copies, same-pattern copies, and near-duplicates (90%+ similar).
8. **Split:** validation and test come **only from real messages**. Synthetic messages are used for training only.

## Final counts

| Split | Scam | Safe |
|---|---|---|
| Train | 8,527 | 8,672 |
| Validation | 1,066 | 465 |
| Test | 1,066 | 465 |

## Known issues

- About 10–20% of IMC reports look like real messages rather than scams (users report messages they are unsure about). Most are kept for now.
- Real safe messages in validation/test are personal chats only. There are no real safe *service* messages yet.
- The synthetic safe messages often start with a brand name ("Uber: …") and are shorter than scams.

## Sources and licenses

**IMC 2025 smishing dataset** (CC BY 4.0) — https://github.com/reportsmishing/Smishing-Dataset-IMC25

> Sharad Agarwal, Antonis Papasavva, Guillermo Suarez-Tangil, Marie Vasek. *Fishing for Smishing: Understanding SMS Phishing Infrastructure and Strategies by Mining Public User Reports.* ACM Internet Measurement Conference (IMC '25), 2025. https://doi.org/10.1145/3730567.3764431

**SMS Phishing Dataset for Machine Learning and Pattern Recognition** (CC BY 4.0) — https://data.mendeley.com/datasets/f45bkkt8pr/1

> Sandhya Mishra, Devpriya Soni. *SMS Phishing Dataset for Machine Learning and Pattern Recognition.* Mendeley Data, V1, 2022. https://doi.org/10.17632/f45bkkt8pr.1

**Synthetic safe service messages** (CC BY 4.0, this project) — written with an AI model for this project and checked for duplicates, reused templates and scam-like content. Brands and details are fictional examples.

Files in `processed/` and `splits/` are derived from the sources above and keep the CC BY 4.0 license. Please credit the original authors when you use them.
