"""md2html.py —— 把 report/REPORT.md 转成打印用 HTML（供浏览器 printToPDF）"""
import pathlib
import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
md_path = ROOT / 'report' / 'REPORT.md'
out_path = ROOT / 'report' / '_print.html'

CSS = """
@page { size: A4; margin: 2.2cm 2.4cm; }
* { box-sizing: border-box; }
body {
  font-family: "Georgia", "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.62; color: #1a1a1a;
  max-width: 17cm; margin: 0 auto; padding: 0;
}
h1 { font-size: 19pt; line-height: 1.3; margin: 0 0 4pt; }
h2 {
  font-size: 13.5pt; margin: 22pt 0 8pt; padding-bottom: 3pt;
  border-bottom: 1.5px solid #333;
}
p { margin: 7pt 0; text-align: justify; }
strong { font-weight: 700; }
a { color: #1a1a1a; text-decoration: none; border-bottom: 1px dotted #999; }
img { max-width: 100%; display: block; margin: 10pt auto 4pt; }
em { color: inherit; }
h1 + p em, h1 + p + p em { color: #444; }
table {
  border-collapse: collapse; width: 100%; margin: 10pt 0 4pt;
  font-size: 9.5pt; page-break-inside: avoid;
}
th, td { border: 1px solid #bbb; padding: 4.5pt 7pt; text-align: left; }
th { background: #f0ede8; }
tr:nth-child(even) td { background: #faf9f7; }
/* 表格标题（表上方的粗体行）不断页 */
p:has(strong:only-child) { page-break-after: avoid; }
img { page-break-inside: avoid; }
h2 { page-break-after: avoid; }
ul { margin: 7pt 0; padding-left: 22pt; }
li { margin: 3pt 0; }
::selection { background: #ffe9b0; }
"""

html_body = markdown.markdown(
    md_path.read_text(encoding='utf-8'),
    extensions=['tables', 'fenced_code'],
)

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Terraflux — Technical Report</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

out_path.write_text(html, encoding='utf-8')
print(f'[ok] {out_path}')
