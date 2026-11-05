"""ADIM 21 — Künye eşleşme güveni testleri.

Bağımsız çalışır:
    .venv\\Scripts\\python.exe tests\\test_match_confidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from academic_fetcher import FetchedPaper  # noqa: E402
from rag_verifier import SemanticRAGVerifier  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


def paper(**kwargs) -> FetchedPaper:
    base = dict(ref_id="1", query_title="Test", query_doi=None, exists_online=True)
    base.update(kwargs)
    return FetchedPaper(**base)


V = SemanticRAGVerifier.__new__(SemanticRAGVerifier)
V.ollama_model = "test"

# --- _match_confidence birim testleri --------------------------------------

ok, score, reason = V._match_confidence(paper(title_match_score=0.50))
check("zayıf skor güvenilmez", ok is False and abs(score - 0.50) < 1e-6, reason)

ok, score, reason = V._match_confidence(paper(title_match_score=0.90))
check("güçlü skor güvenilir", ok is True, reason)

ok, score, reason = V._match_confidence(paper(title_match_score=0.70))
check("sınır 0.70 güvenilir", ok is True, reason)

ok, score, reason = V._match_confidence(paper(title_match_score=0.699))
check("sınır 0.699 güvenilmez", ok is False, reason)

ok, score, reason = V._match_confidence(paper(title_match_score=0.0))
check("skor yok -> güvenilmez", ok is False and "bilinmiyor" in reason, reason)

ok, score, reason = V._match_confidence(
    paper(title_match_score=0.20, query_doi="10.1000/x", verified_doi="10.1000/x")
)
check("DOI birebir -> güvenilir", ok is True and "DOI" in reason, reason)

ok, score, reason = V._match_confidence(
    paper(title_match_score=0.95, query_doi="10.1000/x", verified_doi="10.1000/y")
)
check("DOI uyuşmazsa skor belirler", ok is True and abs(score - 0.95) < 1e-6, reason)

# --- Uçtan uca erken dallar ------------------------------------------------

res = V.verify_claim_against_paper("Test iddiası [1].", paper(title_match_score=0.50, ref_id="1"))
check(
    "zayıf eşleşme -> DOGRULANAMADI",
    res.verdict == "DOGRULANAMADI" and abs(res.match_score - 0.50) < 1e-6 and "zayıf" in res.explanation.lower(),
    f"{res.verdict} / {res.match_score}",
)

res = V.verify_claim_against_paper("Test iddiası [1].", paper(title_match_score=0.90, ref_id="1"))
check("güçlü eşleşme akışa devam eder", res.verdict == "KANIT_YETERSİZ", res.verdict)

res = V.verify_claim_against_paper(
    "Test iddiası [1].",
    paper(title_match_score=0.20, query_doi="10.1000/x", verified_doi="10.1000/x", ref_id="1"),
)
check("DOI doğrulanmış kayıt akışa devam eder", res.verdict == "KANIT_YETERSİZ", res.verdict)

res = V.verify_claim_against_paper("Test iddiası [1].", paper(title_match_score=0.0, ref_id="1"))
check(
    "skor yok -> DOGRULANAMADI (temkinli)",
    res.verdict == "DOGRULANAMADI" and "bilinmiyor" in res.explanation,
    res.verdict,
)

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
