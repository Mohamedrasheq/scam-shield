"""
Shared text cleaning used by BOTH training data (clean_data.py) and live checks (predict.py),
so the model always sees messages in the same format.
"""

import html
import re

def fix_text(s):
    # Undo "UTF-8 read as Latin-1" damage (e.g. "Ã¼" -> "ü")
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    s = html.unescape(s)                       # "&amp;" -> "&", "&lt;" -> "<"
    s = s.replace("\ufffd", " ")               # unreadable symbol
    s = re.sub(r"\s+", " ", s).strip()         # tidy spaces / line breaks
    return s



# IMC number-like tags the masking tool used (often wrongly) -> one <NUMBER> tag
NUMBER_TAGS = ["US_DRIVER_LICENSE", "US_BANK_NUMBER", "US_ITIN", "US_PASSPORT", "US_SSN",
               "CREDIT_CARD", "UK_NHS", "MEDICAL_LICENSE", "CRYPTO", "IP_ADDRESS", "DECIMAL", "UKP"]
# IMC word tags we can't reproduce in the safe data -> removed
WORD_TAGS = ["DATE_TIME", "TIME", "NAMED_ENTITY", "NAME", "LOCATION", "NRP", "EPIC", "ADV",
             "ATTENTION", "MRP", "UNBW"]
KEEP_TAGS = ["URL", "PHONE_NUMBER", "EMAIL_ADDRESS", "NUMBER"]

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+(\.[\w-]+)*\.(com|net|org|co|uk|info|xyz|top|ly|do|me|io|"
                 r"biz|us|ca|au|shop|online|site|live|app|link|club|vip|cc|tk|ml|ga|gl|gd|ws)\b(/\S*)?)", re.I)
PHONE = re.compile(r"(?<![\w<])\+?\d[\d\s().-]{6,}\d(?![\w>])")
LONGNUM = re.compile(r"(?<![\w<])\d{4,}(?![\w>])")


def mask(s):
    for t in NUMBER_TAGS:
        s = s.replace(f"<{t}>", "<NUMBER>")
    for t in WORD_TAGS:
        s = s.replace(f"<{t}>", " ")
    s = s.replace("<EMAIL>", "<EMAIL_ADDRESS>")
    s = EMAIL.sub("<EMAIL_ADDRESS>", s)
    s = URL.sub("<URL>", s)
    s = PHONE.sub("<PHONE_NUMBER>", s)
    s = LONGNUM.sub("<NUMBER>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare(s):
    """Fix broken text, then mask links/phones/emails/long numbers."""
    return mask(fix_text(str(s)))
