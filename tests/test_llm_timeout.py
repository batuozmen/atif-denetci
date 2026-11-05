"""ADIM 26 — Ollama zaman aşımı testi (arayüz donmasını önleme)."""

from __future__ import annotations

import json
import sys
import time
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
chunk = EvidenceChunk(1, "Regular exercise is associated with cardiovascular health.", 0.80)
CLAIM = "Düzenli egzersiz kalp sağlığıyla ilişkilidir."

original_chat = rag_verifier.ollama.chat
original_timeout = rag_verifier.LLM_TIMEOUT

try:
    # 1) Yavaş model: 3 sn uyuyor, zaman aşımı 0.3 sn -> hızlı dönmeli
    def slow_chat(**kwargs):
        time.sleep(3)
        return {"message": {"content": json.dumps({"verdict": "DESTEKLENİYOR", "confidence": 0.9})}}

    rag_verifier.ollama.chat = slow_chat
    rag_verifier.LLM_TIMEOUT = 0.3

    t0 = time.time()
    res = V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    elapsed = time.time() - t0

    check("zaman aşımı hızlı döndü", elapsed < 2.0, f"{elapsed:.2f}s")
    check("karar üretildi (kural katmanı)", res["verdict"] in (
        "DESTEKLENİYOR", "KISMEN_DESTEKLENİYOR", "KANIT_YETERSİZ", "ÇELİŞİYOR"), res["verdict"])
    check("açıklama zaman aşımını bildiriyor", "zaman aşımı" in res["explanation"].lower(),
          res["explanation"][:90])
    check("model etiketi güncel", "zaman aşımı" in res["model_used"].lower(), res["model_used"])

    # 2) Normal (hızlı) model çalışmaya devam ediyor
    def fast_chat(**kwargs):
        return {"message": {"content": json.dumps({
            "verdict": "ÇELİŞİYOR", "confidence": 0.91,
            "explanation": "Kanıt iddiayla çelişiyor.", "suggested_revision": "",
        })}}

    rag_verifier.ollama.chat = fast_chat
    rag_verifier.LLM_TIMEOUT = 5.0
    res2 = V._evaluate_entailment(CLAIM, [chunk], FakePaper())
    check("normal yanıt çalışıyor", res2["verdict"] == "ÇELİŞİYOR", res2["verdict"])
finally:
    rag_verifier.ollama.chat = original_chat
    rag_verifier.LLM_TIMEOUT = original_timeout

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
