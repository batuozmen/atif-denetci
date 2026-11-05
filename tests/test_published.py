"""Yayınlanmış makale üzerinde hızlı denetim: ayrıştırma + ilk N atıf."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from academic_fetcher import AcademicFetcher
from document_parser import DocumentParser
from rag_verifier import SemanticRAGVerifier

PDF = Path(r"C:\Users\btuoz\Desktop\YAYINLANMIŞ MAKALE.pdf")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

t0 = time.time()
p = DocumentParser().parse_file(PDF, PDF.name)
print(f"DOSYA: {PDF.name}")
print(f"kaynak={len(p.references)} atıf={len(p.citations)} hayalet={len(p.ghost_citations)} yetim={len(p.orphan_references)} ({time.time()-t0:.1f}s)")
print("uyarılar:", p.warnings[:3])
print("\nİlk 5 atıf:")
for c in p.citations[:5]:
    print(f"  {c.citation_id} ref={c.matched_ref_id} len={len(c.claim_sentence)} | {c.claim_sentence[:100]}")

seen = {}
for c in p.citations:
    if c.matched_ref_id and c.matched_ref_id not in seen:
        seen[c.matched_ref_id] = c
    if len(seen) >= N:
        break

print(f"\n{len(seen)} tekil kaynak çekiliyor (Sci-Hub kapalı)...")
f = AcademicFetcher(timeout=20, enable_scihub=False)
fetched = {}
for rid, c in seen.items():
    t = time.time()
    fp = f.verify_and_fetch(c.matched_reference, None)
    fetched[rid] = fp
    print(f"  [{rid:>3}] {time.time()-t:5.1f}s durum={fp.lookup_status:9} gerçek={fp.exists_online} skor={fp.title_match_score:.2f} özet={len(fp.abstract)}")

v = SemanticRAGVerifier(embedding_device="cpu")
print(f"\nDoğrulama (Ollama: {v.ollama_model})")
rows = []
for rid, c in seen.items():
    t = time.time()
    r = v.verify_claim_against_paper(c.claim_sentence, fetched[rid], c.citation_id, c.marker, c.context_text)
    rows.append((rid, c, r))
    print(f"  [{rid:>3}] {time.time()-t:5.1f}s {r.verdict:22} güven={r.confidence_score:.2f} eşleşme={r.match_score:.2f} | {c.claim_sentence[:60]}")

print("\nDAĞILIM:", dict(Counter(r.verdict for _, _, r in rows)))
print(f"TOPLAM: {time.time()-t0:.1f}s")
