"""ADIM 17 — Kaynak arama durumu ve 'DOGRULANAMADI' ayrımı testleri.

Ağ hatası / bulunamadı / kanıtlı sahte üçlüsünü doğrular.
Bağımsız çalışır:
    .venv\\Scripts\\python.exe tests\\test_lookup_status.py
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
    base = dict(ref_id="1", query_title="Test", query_doi=None, exists_online=False)
    base.update(kwargs)
    return FetchedPaper(**base)


# 1) Ağ hatası -> DOGRULANAMADI
p = paper(lookup_status="error", lookup_error="CrossRef: Timeout: bağlantı zaman aşımı", sources_checked=["CrossRef", "OpenAlex"])
verdict, exp = SemanticRAGVerifier._classify_missing_source(p)
check("1 ağ hatası -> DOGRULANAMADI", verdict == "DOGRULANAMADI" and "ağ/API" in exp, verdict)

# 2) Bulunamadı -> DOGRULANAMADI (sahte değil)
p = paper(lookup_status="not_found", sources_checked=["CrossRef", "OpenAlex", "PubMed"])
verdict, exp = SemanticRAGVerifier._classify_missing_source(p)
check("2 bulunamadı -> DOGRULANAMADI", verdict == "DOGRULANAMADI" and "GELMEZ" in exp, verdict)

# 3) DOI başka makaleye ait -> SAHTE_KAYNAK
p = paper(
    lookup_status="not_found",
    metadata_warnings=["⚠️ Başlık–DOI Uyuşmazlığı: Kaynakçada yazılan DOI başka bir makaleye aittir!"],
)
verdict, exp = SemanticRAGVerifier._classify_missing_source(p)
check("3 DOI uyuşmazlığı -> SAHTE_KAYNAK", verdict == "SAHTE_KAYNAK" and "DOI" in exp, verdict)

# 4) Uçtan uca erken dal: verify_claim_against_paper ağ hatasında DOGRULANAMADI döner
v = SemanticRAGVerifier.__new__(SemanticRAGVerifier)
v.ollama_model = "test"
p = paper(lookup_status="error", lookup_error="Timeout", ref_id="7")
res = v.verify_claim_against_paper("Test iddiası [7].", p, citation_id="CIT_7", marker="[7]")
check("4 verify: ağ hatası -> DOGRULANAMADI", res.verdict == "DOGRULANAMADI" and res.ref_id == "7", res.verdict)

# 5) Uçtan uca: bulunamadı ama kanıt yok -> DOGRULANAMADI, alternatif öneri yok
p = paper(lookup_status="not_found", ref_id="8", sources_checked=["CrossRef"])
res = v.verify_claim_against_paper("Test iddiası [8].", p, citation_id="CIT_8", marker="[8]")
check("5 verify: bulunamadı -> DOGRULANAMADI", res.verdict == "DOGRULANAMADI" and res.alternative_papers == [], res.verdict)

# 6) Kanıtlı sahte -> alternatif öneri çağrılır
class FakeFinder:
    called = False

    def suggest_real_papers(self, claim, top_k=3):
        FakeFinder.called = True
        return []

p = paper(
    lookup_status="not_found",
    ref_id="9",
    metadata_warnings=["⚠️ Başlık–DOI Uyuşmazlığı: farklı makale"],
)
res = v.verify_claim_against_paper("Test iddiası [9].", p, citation_id="CIT_9", marker="[9]", alternative_finder=FakeFinder())
check("6 kanıtlı sahte -> SAHTE_KAYNAK + alternatif", res.verdict == "SAHTE_KAYNAK" and FakeFinder.called, res.verdict)

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
