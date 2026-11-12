"""
Live checker: is this message a scam?

Usage:
  .venv/bin/python predict.py "Your parcel is on hold, pay $1.99 at usps-redeliver.top"   -> one message
  .venv/bin/python predict.py                                                           -> type or paste messages

In the typing mode, a pasted message with several lines is checked as ONE message.
Type q (or press Ctrl+D) to quit.

Uses the fine-tuned model in models/scam-distilbert-run2 and the SAME cleaning as training
(text_utils.prepare), so links / phones / numbers become <URL>, <PHONE_NUMBER>, <NUMBER>.
"""

import logging as pylog
import os
import re
import select
import sys

# Hide a harmless PyTorch warning on macOS
pylog.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(pylog.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging

from text_utils import prepare

logging.set_verbosity_error()
logging.disable_progress_bar()

MODEL_DIR = "models/scam-distilbert-run2"
MIN_WORDS = 3   # same rule as training: shorter messages were removed, so the model can't judge them
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()


def check(text):
    cleaned = prepare(text)
    words = len(re.findall(r"[A-Za-z]{2,}", re.sub(r"<[A-Z_]+>", " ", cleaned)))
    if words < MIN_WORDS:
        print(f"\n➖ TOO SHORT to judge (needs at least {MIN_WORDS} words)\n  Model saw: {cleaned}\n")
        return
    enc = tokenizer(cleaned, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        prob = torch.softmax(model(**enc).logits, dim=-1)[0, 1].item()
    tag = "🚨 SCAM" if prob > 0.5 else "✅ SAFE"
    print(f"\n{tag}  ({prob:.0%} scam)\n  Model saw: {cleaned}\n")


PASTE_START, PASTE_END = b"\x1b[200~", b"\x1b[201~"


def enable_paste_mode():
    """Ask the terminal to wrap pastes in start/end markers ("bracketed paste"),
    so a multi-line paste is always read as ONE complete message."""
    if not sys.stdin.isatty():
        return
    import atexit
    import termios
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ECHOCTL          # don't print the markers as ^[[200~
    termios.tcsetattr(fd, termios.TCSANOW, new)
    sys.stdout.write("\x1b[?2004h")
    sys.stdout.flush()

    def restore():
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSANOW, old)

    atexit.register(restore)


def read_message():
    """Read one message.
    - Pasted text: keep reading until the paste END marker arrives (it comes after the
      last pasted line once Enter is pressed), so no line is left behind.
    - Typed text: one line, sent with Enter.
    - Piped input (tests): read whatever arrives together."""
    fd = sys.stdin.fileno()
    buf = os.read(fd, 65536)             # wait for input
    if not buf:
        return None                      # Ctrl+D
    while True:
        if PASTE_START in buf and PASTE_END not in buf:
            more = os.read(fd, 65536)    # paste not finished yet: wait for the rest
        elif select.select([fd], [], [], 0.05)[0]:
            more = os.read(fd, 65536)    # more input already waiting
        else:
            break
        if not more:
            break
        buf += more
    buf = buf.replace(PASTE_START, b"").replace(PASTE_END, b"")
    return buf.decode("utf-8", errors="replace").strip()


if len(sys.argv) > 1:
    check(" ".join(sys.argv[1:]))
else:
    enable_paste_mode()
    print("Type or paste a message, then press Enter. Type q to quit.")
    while True:
        print("> ", end="", flush=True)
        text = read_message()
        if text is None or text.lower() in {"q", "quit", "exit"}:
            break
        if text:
            check(text)
