"""Render INFORME_AURUM_MARKET.md into the deliverable PDF with WeasyPrint."""

from __future__ import annotations

from aurum_discovery.config import PROJECT_ROOT

INFORME_MARKDOWN = PROJECT_ROOT / "INFORME_AURUM_MARKET.md"
INFORME_PDF = PROJECT_ROOT / "INFORME_AURUM_MARKET.pdf"

PAGE_CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm;
    @bottom-center { content: counter(page) " / " counter(pages); font-size: 9px; color: #718096; }
}
body { font-family: "DejaVu Sans", "Helvetica Neue", Arial, sans-serif; font-size: 10.5px; line-height: 1.5; color: #1a202c; }
h1 { font-size: 20px; margin-bottom: 2px; }
h1 + p strong { color: #2b6cb0; }
h2 { font-size: 14px; color: #2b6cb0; border-bottom: 1px solid #cbd5e0; padding-bottom: 3px; margin-top: 18px; }
h3 { font-size: 12px; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5px; }
th, td { border: 1px solid #cbd5e0; padding: 4px 6px; text-align: left; }
th { background: #edf2f7; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9px; background: #edf2f7; padding: 0 3px; border-radius: 3px; }
pre { background: #f7fafc; border: 1px solid #e2e8f0; padding: 8px; font-size: 9px; overflow-x: hidden; }
pre code { background: none; padding: 0; }
img { max-width: 100%; }
blockquote { border-left: 3px solid #2b6cb0; margin-left: 0; padding-left: 10px; color: #4a5568; }
hr { border: none; border-top: 1px solid #cbd5e0; margin: 14px 0; }
a { color: #2b6cb0; text-decoration: none; }
li { margin-bottom: 2px; }
"""


def main() -> None:
    """Convert the report markdown to a paginated A4 PDF."""
    try:
        import markdown
        from weasyprint import HTML
    except ImportError as error:
        raise RuntimeError(
            "Faltan las dependencias del extra 'pdf'. Ejecuta "
            "`uv sync --all-extras` o `make informe` (usa --extra pdf)."
        ) from error
    if not INFORME_MARKDOWN.exists():
        raise FileNotFoundError(f"No existe {INFORME_MARKDOWN}.")
    body = markdown.markdown(
        INFORME_MARKDOWN.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    document = (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        f"<style>{PAGE_CSS}</style></head><body>{body}</body></html>"
    )
    HTML(string=document, base_url=str(PROJECT_ROOT)).write_pdf(INFORME_PDF)
    print(f"Informe escrito en {INFORME_PDF.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
