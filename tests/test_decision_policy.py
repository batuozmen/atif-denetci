"""ADIM 16 — Karar politikası ve negasyon düzeltmesi için regresyon testleri.

Bağımsız çalışır (pytest gerekmez):
    .venv\\Scripts\\python.exe tests\\test_decision_policy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import rag_verifier  # noqa: E402
from rag_verifier import EvidenceChunk, SemanticRAGVerifier  # noqa: E402


class FakePaper:
    full_text = "x" * 5000
    abstract = ""
    tldr = ""


def make_verifier() -> SemanticRAGVerifier:
    v = SemanticRAGVerifier.__new__(SemanticRAGVerifier)  # model yüklemeden
    v.ollama_model = "test"
    return v


V = make_verifier()
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


def heuristic(claim: str, evidence: str, sim: float, full_text: bool = True):
    paper = FakePaper()
    paper.full_text = "x" * 5000 if full_text else ""
    chunk = EvidenceChunk(chunk_id=1, text=evidence, semantic_score=sim)
    return V._heuristic_semantic_nli(claim, [chunk], paper)


# ---------------------------------------------------------------------------
# A. Kural katmanı
# ---------------------------------------------------------------------------

r = heuristic("İlaç depresyon belirtilerini azaltır.", "İlaç depresyon belirtilerini azaltmaz.", 0.90)
check("A1 TR negasyon", r["verdict"] == "ÇELİŞİYOR" and r["signal"] == "negation_conflict", r["verdict"])

r = heuristic("The drug reduces depressive symptoms.", "The drug does not reduce depressive symptoms.", 0.90)
check("A2 EN negasyon", r["verdict"] == "ÇELİŞİYOR" and r["signal"] == "negation_conflict", r["verdict"])

r = heuristic("Dünyanın yüzde 90'ı sudan oluşur.", "Dünya yüzeyinin yaklaşık %71'i su ile kaplıdır.", 0.80)
check("A3 sayısal çelişki", r["verdict"] == "ÇELİŞİYOR" and r["signal"] == "numeric_conflict", r["verdict"])

r = heuristic("Dünyanın yüzde 90'ı sudan oluşur.", "Yüzde doksanı su olan dünyamız eşsizdir.", 0.80)
check("A4 sayısal eşleşme", r["verdict"] == "DESTEKLENİYOR" and r["signal"] == "numeric_match", r["verdict"])

r = heuristic("Düzenli egzersiz kalp sağlığını iyileştirir.", "Regular exercise improves cardiovascular health.", 0.80)
check("A5 paraphrase", r["verdict"] == "DESTEKLENİYOR", r["verdict"])

r = heuristic("Dünyanın yüzde 90'ı sudan oluşur.", "Kemoterapi hastalarında nötrofil sayısı takip edilmiştir.", 0.30)
check("A6 alakasız", r["verdict"] == "KANIT_YETERSİZ", r["verdict"])

r = heuristic("İlaç belirtileri azaltmaz.", "The drug does not reduce symptoms.", 0.90)
check("A7 iki taraf da olumsuz", r["verdict"] != "ÇELİŞİYOR", r["verdict"])


# ---------------------------------------------------------------------------
# B. Birleştirme politikası
# ---------------------------------------------------------------------------

def llm(verdict: str, conf: float = 0.9) -> dict:
    return {
        "verdict": verdict,
        "confidence": conf,
        "explanation": "llm açıklaması",
        "claim_meaning": "c",
        "evidence_meaning": "e",
        "suggested_revision": "",
        "model_used": "Ollama (test)",
    }


h_support = heuristic("Düzenli egzersiz kalp sağlığını iyileştirir.", "Regular exercise improves cardiovascular health.", 0.80)
out = V._combine_decisions(h_support, llm("ÇELİŞİYOR"))
check("B1 LLM çelişkisi kural desteğini ezer", out["verdict"] == "ÇELİŞİYOR" and out["decision_source"] == "llm", f"{out['verdict']}/{out['decision_source']}")

h_num = heuristic("Dünyanın yüzde 90'ı sudan oluşur.", "Dünya yüzeyinin %71'i su ile kaplıdır.", 0.80)
out = V._combine_decisions(h_num, llm("DESTEKLENİYOR"))
check("B2 nesnel sayı çelişkisi LLM'i ezer", out["verdict"] == "ÇELİŞİYOR" and out["decision_source"] == "kural", f"{out['verdict']}/{out['decision_source']}")

h_match = heuristic("Dünyanın yüzde 90'ı sudan oluşur.", "Yüzde doksanı su olan dünyamız eşsizdir.", 0.80)
out = V._combine_decisions(h_match, llm("KANIT_YETERSİZ", 0.5))
check("B3 sayı eşleşmesi 'yetersiz' ısrarını kırar", out["verdict"] == "DESTEKLENİYOR" and out["decision_source"] == "kural+llm", f"{out['verdict']}/{out['decision_source']}")

out = V._combine_decisions(h_support, None)
check("B4 LLM yoksa kural kararı", out["verdict"] == "DESTEKLENİYOR" and out["decision_source"] == "kural", f"{out['verdict']}/{out['decision_source']}")

h_neg = heuristic("İlaç belirtileri azaltır.", "İlaç belirtileri azaltmaz.", 0.90)
out = V._combine_decisions(h_neg, llm("DESTEKLENİYOR"))
check("B5 negasyon çelişkisi LLM desteğini ezer", out["verdict"] == "ÇELİŞİYOR" and out["decision_source"] == "kural", f"{out['verdict']}/{out['decision_source']}")


# ---------------------------------------------------------------------------
# C. Uçtan uca (Ollama taklit edilerek)
# ---------------------------------------------------------------------------

original_chat = rag_verifier.ollama.chat

try:
    def fake_chat_contradiction(**kwargs):
        return {"message": {"content": json.dumps({
            "claim_meaning": "egzersiz iyileştirir",
            "evidence_meaning": "egzersiz iyileştirmez",
            "verdict": "ÇELİŞİYOR",
            "confidence": 0.93,
            "explanation": "Kanıt, iddianın tersini söylüyor.",
            "suggested_revision": "",
        })}}

    rag_verifier.ollama.chat = fake_chat_contradiction
    paper = FakePaper()
    chunk = EvidenceChunk(1, "Regular exercise does not improve cardiovascular health.", 0.85)
    res = V._evaluate_entailment("Düzenli egzersiz kalp sağlığını iyileştirir.", [chunk], paper)
    check("C1 LLM çelişkisi nihai karar", res["verdict"] == "ÇELİŞİYOR" and res["decision_source"] == "llm", f"{res['verdict']}/{res['decision_source']}")

    def fake_chat_broken(**kwargs):
        raise RuntimeError("ollama kapalı")

    rag_verifier.ollama.chat = fake_chat_broken
    res = V._evaluate_entailment("Düzenli egzersiz kalp sağlığını iyileştirir.", [chunk], paper)
    check("C2 Ollama yoksa kural devreye girer", res["decision_source"] == "kural", f"{res['verdict']}/{res['decision_source']}")
finally:
    rag_verifier.ollama.chat = original_chat


# ---------------------------------------------------------------------------
# Rapor
# ---------------------------------------------------------------------------

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
