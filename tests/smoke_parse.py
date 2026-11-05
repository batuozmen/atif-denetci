"""Hızlı duman testi: bir PDF'i ayrıştırır ve özet yazdırır.

Kullanım:
    .venv\\Scripts\\python.exe tests\\smoke_parse.py .cache\\pdfs\\ornek.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_parser import DocumentParser  # noqa: E402


def main() -> int:
    # Windows konsolu cp1254 olabilir; Türkçe/Unicode karakterler için UTF-8'e geç.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if len(sys.argv) < 2:
        print("kullanım: smoke_parse.py <pdf|docx>")
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"dosya yok: {path}")
        return 2

    parsed = DocumentParser().parse_file(path, path.name)
    summary = parsed.to_dict()
    summary.pop("warnings", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for ref in parsed.references[:5]:
        print(f"  {ref.ref_id:>6} | {ref.year or '-':<5} | {ref.title[:60]:<60} | {ref.doi or '-'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
