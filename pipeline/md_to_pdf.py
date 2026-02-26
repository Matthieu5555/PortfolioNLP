"""Convert a Markdown file to PDF via markdown + weasyprint."""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from weasyprint import HTML

CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;
}
body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 {
    font-size: 20pt;
    border-bottom: 2px solid #333;
    padding-bottom: 6px;
    margin-top: 0;
}
h2 {
    font-size: 14pt;
    color: #2c3e50;
    margin-top: 28px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
}
h3 {
    font-size: 12pt;
    color: #34495e;
    margin-top: 20px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 9.5pt;
}
th, td {
    border: 1px solid #bbb;
    padding: 5px 8px;
    text-align: left;
}
th {
    background-color: #f0f0f0;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #fafafa;
}
blockquote {
    border-left: 3px solid #888;
    margin: 12px 0;
    padding: 4px 16px;
    color: #444;
    background: #f9f9f9;
    font-size: 10pt;
}
code {
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
}
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 24px 0;
}
strong {
    color: #1a1a1a;
}
"""


def convert(md_path: str, pdf_path: str) -> None:
    md_text = Path(md_path).read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code"],
    )
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{html_body}</body></html>"""
    HTML(string=full_html).write_pdf(pdf_path)
    print(f"Written {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT.md OUTPUT.pdf")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
