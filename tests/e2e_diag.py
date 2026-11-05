"""ADIM 22 tanı: tek bir atıf için kanıtları ve LLM çıktısını gösterir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from academic_fetcher import AcademicFetcher  # noqa: E402
from document_parser import DocumentParser  # noqa: E402
from rag_verifier import SemanticRAGVerifier  # noqa: E402

DOC = Path(r"C:\Users\btuoz\Desktop\Manuscript.docx")
parsed = DocumentParser().parse_file(DOC, DOC.name)

target = sys.argv[1] if len(sys.argv) > 1 else "1"
cit = next(c for c in parsed.citations if c.matched_ref_id == target)

fetcher = AcademicFetcher(timeout=20, enable_scihub=False)
paper = fetcher.verify_and_fetch(cit.matched_reference, None)

print("İDDİA:", cit.claim_sentence)
print("\nKAYNAK:", paper.verified_title, "| özet:", len(paper.abstract), "| tam metin:", len(paper.full_text))

v = SemanticRAGVerifier(embedding_device="cpu")
chunks = v.retrieve_top_evidence(cit.claim_sentence, paper.combined_evidence_text, top_k=4)
for i, ch in enumerate(chunks, 1):
    print(f"\n--- KANIT {i} (skor {ch.semantic_score:.3f}) ---\n{ch.text[:600]}")

h = v._heuristic_semantic_nli(cit.claim_sentence, chunks, paper)
print("\nHEURISTIC:", json.dumps({k: h[k] for k in ("verdict", "confidence", "signal", "objective")}, ensure_ascii=False))
print("HINT:", h.get("hint"))
