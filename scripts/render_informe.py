"""Compile the LaTeX report into INFORME_AURUM_MARKET.pdf with Tectonic."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from aurum_discovery.config import ARTIFACTS_DIRECTORY, PROJECT_ROOT

INFORME_SOURCE = PROJECT_ROOT / "docs" / "informe" / "INFORME_AURUM_MARKET.tex"
DIAGRAM_SVG = PROJECT_ROOT / "docs" / "images" / "arquitectura.svg"
DIAGRAM_PDF = INFORME_SOURCE.parent / "arquitectura.pdf"
INFORME_PDF = PROJECT_ROOT / "INFORME_AURUM_MARKET.pdf"

TECTONIC_VERSION = "0.15.0"
TECTONIC_URL = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    f"tectonic%40{TECTONIC_VERSION}/"
    f"tectonic-{TECTONIC_VERSION}-x86_64-unknown-linux-gnu.tar.gz"
)
TECTONIC_LOCAL = ARTIFACTS_DIRECTORY / "bin" / "tectonic"


def convert_diagram() -> None:
    """Render the SVG architecture diagram as the PDF the report embeds."""
    try:
        import cairosvg
    except ImportError as error:
        raise RuntimeError(
            "Falta cairosvg (extra 'pdf'). Ejecuta `uv sync --all-extras`."
        ) from error
    cairosvg.svg2pdf(url=str(DIAGRAM_SVG), write_to=str(DIAGRAM_PDF))


def resolve_tectonic() -> Path:
    """Return a Tectonic binary, downloading a pinned release if needed."""
    on_path = shutil.which("tectonic")
    if on_path:
        return Path(on_path)
    if TECTONIC_LOCAL.exists():
        return TECTONIC_LOCAL
    print(f"Descargando Tectonic {TECTONIC_VERSION} (una sola vez)...")
    TECTONIC_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    archive_path = TECTONIC_LOCAL.parent / "tectonic.tar.gz"
    urllib.request.urlretrieve(TECTONIC_URL, archive_path)
    with tarfile.open(archive_path) as archive:
        member = archive.getmember("tectonic")
        archive.extract(member, TECTONIC_LOCAL.parent)
    archive_path.unlink()
    TECTONIC_LOCAL.chmod(0o755)
    return TECTONIC_LOCAL


def main() -> None:
    """Convert the diagram, compile the LaTeX source, and place the PDF."""
    if not INFORME_SOURCE.exists():
        raise FileNotFoundError(f"No existe {INFORME_SOURCE}.")
    convert_diagram()
    tectonic = resolve_tectonic()
    output_directory = ARTIFACTS_DIRECTORY / "informe"
    output_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(tectonic),
            "--outdir",
            str(output_directory),
            "--chatter",
            "minimal",
            str(INFORME_SOURCE),
        ],
        check=True,
    )
    shutil.copy2(output_directory / INFORME_PDF.name, INFORME_PDF)
    print(f"Informe escrito en {INFORME_PDF.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
