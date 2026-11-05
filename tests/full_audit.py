"""Tam ölçekli denetim (sağlam sürüm): kaldığı yerden devam, satır satır kayıt, zaman aşımı.

Kullanım:
    .venv\\Scripts\\python.exe -u tests\\full_audit.py --doc "...docx" [--timeout 150]
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from academic_fetcher import AcademicFetcher, FetchedPaper  # noqa: E402
from document_parser import DocumentParser  # noqa: E402
from rag_verifier import SemanticRAGVerifier, VerificationResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "raporlar"
CSV_HEADER = ["citation_id", "Atıf", "Kaynak No", "Karar", "Güven", "Eşleşme",
              "Karar Kaynağı", "İddia", "Kanıt (ilk 300)", "Açıklama", "Öneri"]


def run_with_timeout(fn, timeout: float):
    """fn'i ayrı (daemon) iş parçacığında çalıştırır; süre aşımında ('timeout', None) döner."""
    q: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            q.put(("ok", fn()))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    threading.Thread(target=worker, daemon=True).start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return ("timeout", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=150.0)
    args = ap.parse_args()

    doc_path = Path(args.doc)
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    csv_path = OUT_DIR / f"tam-denetim-{stamp}.csv"
    md_path = OUT_DIR / f"tam-denetim-{stamp}.md"

    t_start = time.time()
    parsed = DocumentParser().parse_file(doc_path, doc_path.name)
    print(
        f"[1/3] Ayrıştırma: {len(parsed.references)} kaynak, {len(parsed.citations)} atıf, "
        f"hayalet={len(parsed.ghost_citations)}, yetim={len(parsed.orphan_references)}",
        flush=True,
    )

    fetcher = AcademicFetcher(timeout=20, enable_scihub=False, enable_jina=True)
    fetched: dict[str, FetchedPaper] = {}
    total_refs = len(parsed.references)
    t_fetch = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetcher.verify_and_fetch, ref, None): ref for ref in parsed.references}
        for fut in as_completed(futures):
            ref = futures[fut]
            done += 1
            try:
                fetched[ref.ref_id] = fut.result()
            except Exception as exc:  # noqa: BLE001
                fetched[ref.ref_id] = FetchedPaper(
                    ref_id=ref.ref_id, query_title=ref.title or ref.raw_text,
                    query_doi=ref.doi, lookup_status="error",
                    lookup_error=f"{type(exc).__name__}: {exc}"[:200],
                )
            if done % 25 == 0 or done == total_refs:
                found = sum(1 for p in fetched.values() if p.exists_online)
                print(f"      çekim [{done}/{total_refs}] bulunan={found} ({time.time()-t_fetch:.0f}s)", flush=True)

    print(f"[2/3] Çekim bitti: {sum(1 for p in fetched.values() if p.exists_online)}/{total_refs} "
          f"({time.time()-t_fetch:.0f}s)", flush=True)

    verifier = SemanticRAGVerifier(embedding_device="cpu")

    new_file = not csv_path.exists()
    csv_fh = csv_path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.writer(csv_fh)
    if new_file:
        writer.writerow(CSV_HEADER)
        csv_fh.flush()

    counts: Counter = Counter()
    rows: list[tuple] = []
    total_cits = len(parsed.citations)
    t_ver = time.time()
    timeouts = 0
    for i, cit in enumerate(parsed.citations, start=1):
        paper = fetched.get(cit.matched_ref_id) if cit.matched_ref_id else None

        if paper is None:
            res = VerificationResult(
                citation_id=cit.citation_id, marker=cit.marker,
                claim_sentence=cit.claim_sentence, context_text=cit.context_text,
                ref_id=None, verdict="HAYALET_ATIF", confidence_score=1.0,
                semantic_similarity=0.0, best_evidence="",
                explanation="Metinde atıf var, kaynakçada yok.",
                used_llm_model="Biçimsel Denetçi", decision_source="kural",
            )
        else:
            status, value = run_with_timeout(
                lambda: verifier.verify_claim_against_paper(
                    cit.claim_sentence, paper, cit.citation_id, cit.marker, cit.context_text
                ),
                args.timeout,
            )
            if status == "ok":
                res = value
            elif status == "timeout":
                timeouts += 1
                res = VerificationResult(
                    citation_id=cit.citation_id, marker=cit.marker,
                    claim_sentence=cit.claim_sentence, context_text=cit.context_text,
                    ref_id=cit.matched_ref_id, verdict="DOGRULANAMADI",
                    confidence_score=0.0, semantic_similarity=0.0, best_evidence="",
                    explanation=f"Değerlendirme {args.timeout:.0f} sn içinde tamamlanmadı (zaman aşımı).",
                    used_llm_model="Zaman Aşımı", decision_source="kural",
                )
            else:
                res = VerificationResult(
                    citation_id=cit.citation_id, marker=cit.marker,
                    claim_sentence=cit.claim_sentence, context_text=cit.context_text,
                    ref_id=cit.matched_ref_id, verdict="DOGRULANAMADI",
                    confidence_score=0.0, semantic_similarity=0.0, best_evidence="",
                    explanation=f"Değerlendirme hatası: {type(value).__name__}: {value}"[:300],
                    used_llm_model="Hata", decision_source="kural",
                )

        counts[res.verdict] += 1
        rows.append((cit, paper, res))
        writer.writerow([
            cit.citation_id, cit.marker, cit.matched_ref_id or "-", res.verdict,
            f"{res.confidence_score:.2f}", f"{res.match_score:.2f}", res.decision_source or "-",
            res.claim_sentence[:500], res.best_evidence[:300], res.explanation[:500],
            res.suggested_revision[:300],
        ])
        csv_fh.flush()

        if i % 5 == 0 or i == total_cits:
            rate = (time.time() - t_ver) / i
            print(f"      doğrulama [{i}/{total_cits}] {time.time()-t_ver:.0f}s "
                  f"(ort {rate:.1f}s/atıf, son: {res.verdict})", flush=True)

    csv_fh.close()
    elapsed = time.time() - t_start
    print(f"[3/3] Bitti. Süre: {elapsed:.0f}s, zaman aşımı: {timeouts}", flush=True)
    print("DAĞILIM:", dict(counts), flush=True)

    lines = [
        "# Tam Ölçekli Atıf Denetim Raporu",
        f"**Belge:** {doc_path.name}",
        f"**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"**Süre:** {elapsed:.0f} sn | **Zaman aşımı:** {timeouts}",
        "",
        f"- Kaynak: {len(parsed.references)} | Atıf: {len(parsed.citations)} | "
        f"Hayalet: {len(parsed.ghost_citations)} | Yetim: {len(parsed.orphan_references)}",
        f"- Bulunan kaynak: {sum(1 for p in fetched.values() if p.exists_online)}/{len(fetched)}",
        "",
        "## Karar Dağılımı",
        "",
        "| Karar | Sayı |",
        "|---|---|",
    ]
    for k, v in counts.most_common():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Kontrol Edilmesi Gerekenler", ""]
    for cit, paper, res in rows:
        if res.verdict in ("ÇELİŞİYOR", "SAHTE_KAYNAK", "DOGRULANAMADI", "KANIT_YETERSİZ"):
            lines.append(f"### {cit.marker} — {res.verdict}")
            lines.append(f"- **İddia:** {res.claim_sentence[:300]}")
            lines.append(f"- **Açıklama:** {res.explanation[:300]}")
            lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"RAPOR: {md_path}", flush=True)
    print(f"CSV  : {csv_path}", flush=True)
    os._exit(0)  # takılı daemon thread'ler çıkışı bloklamasın


if __name__ == "__main__":
    raise SystemExit(main())
