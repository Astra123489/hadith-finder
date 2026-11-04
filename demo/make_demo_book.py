"""Generates demo/demo-book-of-intentions.pdf — a SYNTHETIC bilingual
sample used for trying the app and for docs. All narrations are invented
test sentences, not real hadith quotations. Arabic is laid out with
PyMuPDF's HTML engine (shaped), matching how real Arabic PDFs look."""

import html
import os
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent / "demo-book-of-intentions.pdf"
FONTS_DIR = r"C:\Windows\Fonts"

PAGES = [
    [
        ("Book One — Intentions (demo)", "h1"),
        ("Chapter 1 — Book of Intentions", "h2"),
        ("1. Narrated Tester: The Prophet ﷺ said, 'Deeds are judged by intentions, and every person shall have only what he intended.'", "p"),
        ("2. Narrated Tester: The Prophet ﷺ said, 'A person may perform two acts that appear equal, and one of them raises him while the other is weightless, because of the intention.'", "p"),
        ("3. Narrated Tester: The Prophet ﷺ said, 'Whoever intends a good deed but is unable to perform it, it is recorded for him.' This hadith is sahih.", "p"),
        ("Chapter 2 — Book of Sincerity", "h2"),
        ("4. Narrated Tester: The Prophet ﷺ said, 'The first matter people will be questioned about on the Day of Judgment is their intentions.'", "p"),
    ],
    [
        ("5. قال رسول الله ﷺ: إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى. حديث صحيح.", "p-rtl"),
        ("5. The Messenger of Allah ﷺ said, 'Actions are by intentions, and each person shall have what they intended.' This hadith is sahih.", "p"),
        ("6. Narrated Tester: The Prophet ﷺ said, 'Sometimes a sleeper's intention earns more than a worker's labor, when the intention is pure.'", "p"),
        ("7. Narrated Tester: His companion said: this narration is weak, do not rely on it. حديث ضعيف.", "p"),
        ("Chapter 3 — Book of Mercy", "h2"),
        ("8. Narrated Tester: The Prophet ﷺ said, 'Whoever does not show mercy will not be shown mercy.' This hadith is sahih.", "p"),
        ("9. Narrated Tester: The Prophet ﷺ said, 'The merciful are shown mercy by the Merciful.'", "p"),
    ],
    [
        ("10. Narrated Tester: The Prophet ﷺ said, 'Feed the hungry, greet others warmly, and pray at night while people sleep, and you will enter Paradise in peace.'", "p"),
        ("11. قال رسول الله ﷺ: من لا يرحم لا يُرحم.", "p-rtl"),
        ("11. The Messenger of Allah ﷺ said, 'Whoever is not merciful will not be shown mercy.'", "p"),
        ("12. This final demo narration carries no grading in this book.", "p"),
    ],
]

CSS = """
@font-face { font-family: arial; src: url(arial.ttf); }
body { font-family: arial; font-size: 11pt; color: #111; }
h1 { font-size: 21pt; margin: 0 0 14pt 0; }
h2 { font-size: 14pt; margin: 10pt 0 6pt 0; }
p { margin: 0 0 7pt 0; text-align: justify; }
.rtl { direction: rtl; text-align: right; }
"""


def main() -> None:
    arch = pymupdf.Archive(FONTS_DIR)
    doc = pymupdf.open()
    for blocks in PAGES:
        page = doc.new_page(width=595, height=842)
        parts = ["<style>" + CSS + "</style>"]
        for text, kind in blocks:
            cls = "rtl" if kind == "p-rtl" else ""
            tag = "h1" if kind == "h1" else "h2" if kind == "h2" else "p"
            attr = f' class="{cls}"' if cls else ""
            parts.append(f"<{tag}{attr}>{html.escape(text)}</{tag}>")
        page.insert_htmlbox(pymupdf.Rect(54, 54, 541, 800), "".join(parts), archive=arch)
    doc.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
