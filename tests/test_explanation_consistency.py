"""ADIM 24 — Karar/açıklama tutarlılığı testleri (Ollama taklit edilerek)."""

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

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


class FakePaper:
    full_text = ""
    abstract = "x" * 500
    tldr = ""


V = SemanticRAGVerifier.__new__(SemanticRAGVerifier)
V.ollama_model = "test"

CLAIM = "Düzenli egzersiz kalp hastalığı riskini azaltır."
EVIDENCE = "Regular exercise lowers the risk of cardiovascular disease."
chunk = EvidenceChunk(chunk_id=1, text=EVIDENCE, semantic_score=0.85)

original_chat = rag_verifier.ollama.chat
captured: dict = {}


def make_fake(verdict: str, explanation: str = ""):
    def fake(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        captured["user"] = kwargs["messages"][1]["content"]
        return {"message": {"content": json.dumps({
            "claim_meaning": "c",
            "evidence_meaning": "e",
            "verdict": verdict,
            "confidence": 0.9,
            "explanation": explanation,
            "suggested_revision": "",
        })}}
    return fake


try:
    # 1) KISMEN + boş açıklama -> nötr özet şablonu
    rag_verifier.ollama.chat = make_fake("KISMEN_DESTEKLENİYOR", "")
    res = V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    check(
        "KISMEN + boş açıklama -> özet şablonu",
        res["verdict"] == "KISMEN_DESTEKLENİYOR"
        and "özet" in res["explanation"].lower()
        and "çıkarılabilir" not in res["explanation"].lower(),
        res["explanation"][:80],
    )

    # 2) KISMEN + kısa açıklama -> önek eklenir
    rag_verifier.ollama.chat = make_fake("KISMEN_DESTEKLENİYOR", "Kanıt kısmen uyumlu.")
    res = V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    check(
        "KISMEN + açıklama -> özet öneki",
        res["verdict"] == "KISMEN_DESTEKLENİYOR"
        and res["explanation"].lower().startswith("☑️")
        and "kısmen uyumlu" in res["explanation"].lower(),
        res["explanation"][:80],
    )

    # 3) DESTEKLENİYOR + boş açıklama -> heuristic metni (tutarlı)
    rag_verifier.ollama.chat = make_fake("DESTEKLENİYOR", "")
    res = V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    check(
        "DESTEKLENİYOR + boş açıklama tutarlı",
        res["verdict"] == "DESTEKLENİYOR" and len(res["explanation"]) > 10,
        res["explanation"][:80],
    )

    # 4) ÇELİŞİYOR + boş açıklama -> çelişki şablonu (destek metni değil)
    rag_verifier.ollama.chat = make_fake("ÇELİŞİYOR", "")
    res = V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    check(
        "ÇELİŞİYOR + boş açıklama -> çelişki şablonu",
        res["verdict"] == "ÇELİŞİYOR"
        and "çelişen" in res["explanation"].lower()
        and "çıkarılabilir" not in res["explanation"].lower(),
        res["explanation"][:80],
    )

    # 5) Prompt kuralı güncellendi mi?
    rag_verifier.ollama.chat = make_fake("DESTEKLENİYOR", "ok")
    V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    check(
        "prompt: özet doğrudan söylüyorsa DESTEKLENİYOR kuralı",
        "DOĞRUDAN söylüyorsa" in captured.get("system", ""),
        "kural 2a",
    )
finally:
    rag_verifier.ollama.chat = original_chat

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
