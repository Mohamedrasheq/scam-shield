"""
Clean and combine the raw datasets into ONE training file.

Inputs (never modified):
  data/raw/imc2025/final_dataset_output.csv   -> modern global scam reports (scam only)
  data/raw/mishra_soni/Dataset_5971.csv       -> ham / spam / smishing
  data/raw/synthetic_safe/safe_service_messages.csv -> written safe service messages (safe only)

Output:
  data/processed/clean.csv   columns: text, label (safe|scam), source, scam_type

Steps:
  2. Pick       English only, keep text + label
  3. Labels     ham -> safe, smishing/IMC -> scam, spam -> DROPPED
  4. Fix text   repair broken characters and HTML codes
  5. Junk       drop messages with almost no words
  6. Mask       same tags for links, phones, emails, long numbers in ALL data
  7. Dupes      exact copies, then same-pattern copies
  8. Labels     drop IMC rows that are clearly real (not scam) messages
  9. Balance    report counts
 10. Save       write clean.csv + print a report

Run: .venv/bin/python clean_data.py
"""

import html
import re
from pathlib import Path

import pandas as pd

OUT = Path("data/processed/clean.csv")
pd.set_option("display.width", 200)
report = []


def log(step, msg):
    report.append((step, msg))
    print(f"[{step}] {msg}")


# ---------------------------------------------------------------
# Step 2 + 3: load, pick, unify labels
# ---------------------------------------------------------------
imc = pd.read_csv("data/raw/imc2025/final_dataset_output.csv")
log("load", f"IMC 2025: {len(imc)} rows")
imc = imc[imc["language"].str.lower() == "english"]
log("2 pick", f"IMC English only: {len(imc)} rows")
n = len(imc)
imc = imc[imc["scam_type"].str.lower() != "spam"]
log("3 labels", f"IMC dropped 'spam' type: {n - len(imc)} rows")
imc = pd.DataFrame({"text": imc["text"].astype(str), "label": "scam",
                    "source": "imc2025", "scam_type": imc["scam_type"].fillna("unknown")})

ms = pd.read_csv("data/raw/mishra_soni/Dataset_5971.csv", encoding="latin-1")
ms.columns = [c.strip().upper() for c in ms.columns]
ms["LABEL"] = ms["LABEL"].str.strip().str.lower()
log("load", f"Mishra & Soni: {len(ms)} rows")
n = len(ms)
ms = ms[ms["LABEL"] != "spam"]
log("3 labels", f"Mishra & Soni dropped 'spam': {n - len(ms)} rows")
ms = pd.DataFrame({"text": ms["TEXT"].astype(str),
                   "label": ms["LABEL"].map({"ham": "safe", "smishing": "scam"}),
                   "source": "mishra_soni", "scam_type": ms["LABEL"].map({"ham": "", "smishing": "unknown"})})

syn = pd.read_csv("data/raw/synthetic_safe/safe_service_messages.csv")
log("load", f"Synthetic safe service messages: {len(syn)} rows")
syn = pd.DataFrame({"text": syn["text"].astype(str), "label": "safe",
                    "source": "synthetic", "scam_type": ""})

df = pd.concat([imc, ms, syn], ignore_index=True)
log("3 labels", f"combined: {len(df)} rows | " + ", ".join(f"{k} {v}" for k, v in df['label'].value_counts().items()))

# ---------------------------------------------------------------
# Step 4: fix broken text
# ---------------------------------------------------------------
from text_utils import KEEP_TAGS, fix_text, mask  # shared with predict.py

before = df["text"].copy()
df["text"] = df["text"].map(fix_text)
log("4 fix text", f"messages changed by repair: {(before != df['text']).sum()}")

df["text"] = df["text"].map(mask)
for tag in KEEP_TAGS:
    share = df.groupby("label")["text"].apply(lambda s: s.str.contains(f"<{tag}>", regex=False).mean())
    log("6 mask", f"<{tag}> in safe {share.get('safe', 0):.0%} | in scam {share.get('scam', 0):.0%}")
left = df["text"].str.findall(r"<[A-Z_]+>").explode().dropna()
left = left[~left.str.strip("<>").isin(KEEP_TAGS)]
log("6 mask", f"other tags left over: {left.value_counts().to_dict() or 'none'}")

# ---------------------------------------------------------------
# Step 5: remove junk (too few real words)
# ---------------------------------------------------------------
def word_count(s):
    return len(re.findall(r"[A-Za-z]{2,}", re.sub(r"<[A-Z_]+>", " ", s)))


n = len(df)
junk = df[df["text"].map(word_count) < 3]
df = df[df["text"].map(word_count) >= 3]
log("5 junk", f"removed {n - len(df)} messages with fewer than 3 real words, e.g. {junk['text'].head(3).tolist()}")

# ---------------------------------------------------------------
# Step 8: remove IMC rows that are clearly real messages, not scams
#   - real OTP / code messages ("do not share") with no link and no request to send it
#   - fraud warnings ("beware of ...", "unsaved number") with no link
# ---------------------------------------------------------------
low = df["text"].str.lower()
no_link = ~df["text"].str.contains("<URL>", regex=False)
real_otp = (low.str.contains(r"\b(?:otp|code|passcode|pin)\b") & low.str.contains(r"(?:do not|don't|never) share")
            & ~low.str.contains(r"forward|send (?:it|me|us|the)|reply with|share (?:it )?with (?:me|us)|tell me"))
warning = low.str.contains(r"beware of|unsaved number|we will never ask|never ask (?:you )?for")
wrong = (df["source"] == "imc2025") & no_link & (real_otp | warning)
log("8 labels", f"removed {wrong.sum()} IMC rows that look like real messages, e.g.:")
for t in df.loc[wrong, "text"].head(5):
    log("8 labels", f"   - {t[:110]}")
df = df[~wrong]

# ---------------------------------------------------------------
# Step 7: remove duplicates
# ---------------------------------------------------------------
df["exact_key"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
df["pattern_key"] = (df["exact_key"].str.replace(r"<[a-z_]+>", " ", regex=True)
                     .str.replace(r"\d+", "#", regex=True)
                     .str.replace(r"[^a-z#]+", " ", regex=True).str.strip())

# same pattern but different labels -> unclear, drop all copies
conflict = df.groupby("pattern_key")["label"].nunique()
conflict = set(conflict[conflict > 1].index)
n = len(df)
df = df[~df["pattern_key"].isin(conflict)]
log("7 dupes", f"removed {n - len(df)} rows whose pattern has both labels ({len(conflict)} patterns)")

n = len(df)
df = df.drop_duplicates("exact_key")
log("7 dupes", f"removed {n - len(df)} exact copies")
n = len(df)
df = df.drop_duplicates("pattern_key")
log("7 dupes", f"removed {n - len(df)} same-pattern copies")

# Level 3: near-duplicates (90%+ similar wording, e.g. many SBI "account blocked" variants)
#   Compare messages by overlapping letter-groups (TF-IDF), keep the first of each group.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = df.reset_index(drop=True)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform(df["pattern_key"])
nn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute").fit(vec)
dist, idx = nn.kneighbors(vec)
keep = [True] * len(df)
for i in range(len(df)):
    for d, j in zip(dist[i], idx[i]):
        if j < i and keep[j] and 1 - d >= 0.90 and df.at[j, "label"] == df.at[i, "label"]:
            keep[i] = False
            break
n = len(df)
df = df[keep]
log("7 dupes", f"removed {n - len(df)} near-duplicates (90%+ similar)")

# ---------------------------------------------------------------
# Step 9: balance report
# ---------------------------------------------------------------
log("9 balance", "final counts: " + ", ".join(f"{k} {v}" for k, v in df["label"].value_counts().items()))
for (src, lab), c in df.groupby(["source", "label"]).size().items():
    log("9 balance", f"   {src:12s} {lab:5s} {c}")
types = df[df["label"] == "scam"]["scam_type"].value_counts()
log("9 balance", "scam types: " + ", ".join(f"{k} {v}" for k, v in types.items()))
sbi = df["text"].str.contains(r"\bsbi\b|\byono\b", case=False).sum()
log("9 balance", f"messages mentioning SBI/YONO (Indian bank): {sbi} ({sbi / len(df):.1%})")

# ---------------------------------------------------------------
# Step 10: save
# ---------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
df[["text", "label", "source", "scam_type"]].to_csv(OUT, index=False)
log("10 save", f"saved {len(df)} rows -> {OUT}")

print("\n--- 15 random rows ---")
for _, r in df.sample(15, random_state=11).iterrows():
    print(f"  [{r['label']:4s} | {r['source']:11s}] {r['text'][:120]}")
