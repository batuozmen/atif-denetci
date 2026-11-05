"""ADIM 22 — Uçtan uca gerçek test (Manuscript.docx -> internet -> Ollama).

Kod değiştirmez; yalnızca ölçer:
    .venv\\Scripts\\python.exe tests\\e2e_smoke.py [--n 5]

Not: Sci-Hub kapalı, Jina açık. Ağ ve Ollama gerektirir.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from academic_fetcher import AcademicFetcher  # noqa: E402
from document_parser import DocumentParser  # noqa: E402
from rag_verifier import SemanticRAGVerifier  # noqa: E402

MANUSCRIPT = Path(r"C:\Users\btuoz\Desktop\Manuscript.docx")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="denenecek tekil kaynak sayısı")
    ap.add_argument("--doc", default=str(MANUSCRIPT))
    args = ap.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        print(f"Belge bulunamadı: {doc_path}")
        return 2

    t_start = time.time()
    parser = DocumentParser()
    t0 = time.time()
    parsed = parser.parse_file(doc_path, doc_path.name)
    t_parse = time.time() - t0
    print(f"[1] Ayrıştırma: {len(parsed.references)} kaynak, {len(parsed.citations)} atıf "
          f"({t_parse:.1f} sn), hayalet={len(parsed.ghost_citations)}, yetim={len(parsed.orphan_references)}")

    # İlk N tekil kaynağı seç
    picked: dict[str, object] = {}
    for cit in parsed.citations:
        if cit.matched_ref_id and cit.matched_ref_id not in picked:
            picked[cit.matched_ref_id] = cit
        if len(picked) >= args.n:
            break

    # ---- Çekim ----
    fetcher = AcademicFetcher(timeout=20, enable_scihub=False, enable_jina=True)
    fetched = {}
    print(f"\n[2] İnternetten çekim ({len(picked)} kaynak, Sci-Hub KAPALI)")
    for rid, cit in picked.items():
        ref = cit.matched_reference
        t0 = time.time()
        fp = fetcher.verify_and_fetch(ref, None)
        dt = time.time() - t0
        fetched[rid] = fp
        print(
            f"  [{rid:>4}] {dt:5.1f}s | durum={fp.lookup_status:9} | gerçek={str(fp.exists_online):5} | "
            f"skor={fp.title_match_score:.2f} | özet={len(fp.abstract):4} | tam metin={len(fp.full_text):5} | "
            f"{(ref.first_author_surname or '')[:18]}"
        )
        if not fp.exists_online and fp.lookup_error:
            print(f"         hata: {fp.lookup_error[:120]}")

    # ---- Doğrulama ----
    verifier = SemanticRAGVerifier(embedding_device="cpu")
    print(f"\n[3] Anlam doğrulaması (Ollama: {verifier.ollama_model})")
    results = []
    for rid, cit in picked.items():
        t0 = time.time()
        res = verifier.verify_claim_against_paper(
            claim_sentence=cit.claim_sentence,
            paper=fetched[rid],
            citation_id=cit.citation_id,
            marker=cit.marker,
            context_text=cit.context_text,
        )
        dt = time.time() - t0
        results.append((rid, res, dt))
        print(
            f"  [{rid:>4}] {dt:5.1f}s | {res.verdict:22} | güven={res.confidence_score:.2f} | "
            f"eşleşme={res.match_score:.2f} | kaynak={res.decision_source:9} | {res.used_llm_model[:28]}"
        )

    total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"TOPLAM SÜRE: {total:.1f} sn")
    print(f"Bulunan kaynak: {sum(1 for p in fetched.values() if p.exists_online)}/{len(fetched)}")
    from collections import Counter
    print("Kararlar:", dict(Counter(r.verdict for _, r, _ in results)))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
