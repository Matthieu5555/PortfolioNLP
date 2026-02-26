"""Render docs/research-brief.md to PDF via markdown + weasyprint."""

import markdown
from weasyprint import HTML
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "docs" / "research-brief.md"
PDF_PATH = ROOT / "docs" / "research-brief.pdf"

md_text = MD_PATH.read_text()
html_body = markdown.markdown(md_text, extensions=["tables", "smarty"])

html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4;
    margin: 2.5cm 2.5cm 2.5cm 2.5cm;
  }}
  body {{
    font-family: "Georgia", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
  }}
  h1 {{
    font-size: 20pt;
    margin-bottom: 0.2em;
    color: #111;
  }}
  h2 {{
    font-size: 14pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    color: #222;
    border-bottom: 1px solid #ccc;
    padding-bottom: 0.2em;
  }}
  h3 {{
    font-size: 12pt;
    margin-top: 1.2em;
    margin-bottom: 0.3em;
    color: #333;
  }}
  blockquote {{
    border-left: 3px solid #2a6496;
    padding: 0.8em 1em;
    margin: 1em 0;
    background: #f4f8fb;
    font-size: 10.5pt;
  }}
  blockquote p {{
    margin: 0;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 10pt;
    margin: 0.8em 0;
  }}
  th, td {{
    border: 1px solid #ccc;
    padding: 0.4em 0.6em;
    text-align: left;
  }}
  th {{
    background: #f0f0f0;
    font-weight: bold;
  }}
  ul, ol {{
    margin: 0.5em 0;
    padding-left: 1.5em;
  }}
  li {{
    margin-bottom: 0.3em;
  }}
  hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 1.5em 0;
  }}
  code {{
    font-family: "Courier New", monospace;
    font-size: 10pt;
    background: #f5f5f5;
    padding: 0.1em 0.3em;
    border-radius: 2px;
  }}
  em {{
    font-style: italic;
  }}
  strong {{
    font-weight: bold;
  }}
  p {{
    margin: 0.6em 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

HTML(string=html_doc).write_pdf(str(PDF_PATH))
print(f"PDF written to {PDF_PATH}")
