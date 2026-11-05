"""Heuristic (deterministik) NLI katmanını LLM'e gitmeden test eder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rag_verifier import SemanticRAGVerifier, EvidenceChunk  # noqa: E402

v = SemanticRAGVerifier.__new__(SemanticRAGVerifier)  # model yüklemeden çağır
v.ollama_model = "test"

cases = [
    (
        "Sayısal çelişki",
        "Dünyanın yüzde 90'ı sudan oluşur.",
        "Dünya yüzeyinin yaklaşık %71'i su ile kaplıdır.",
        0.80,
    ),
    (
        "Sayısal eşleşme (paraphrase)",
        "Dünyanın yüzde 90'ı sudan oluşur.",
        "Yüzde doksanı su olan dünyamız biyosfer açısından eşsizdir.",
        0.80,
    ),
    (
        "Yön çelişkisi",
        "İlaç depresyon belirtilerini artırmıştır.",
        "Treatment with the drug significantly decreased depressive symptoms in patients.",
        0.75,
    ),
    (
        "Yön çelişkisi (ters)",
        "İlaç depresyon belirtilerini azaltmıştır.",
        "The drug increased depressive symptoms in the trial cohort.",
        0.75,
    ),
    (
        "Alakasız",
        "Dünyanın yüzde 90'ı sudan oluşur.",
        "Kemoterapi hastalarında nötrofil sayısı takip edilmiştir.",
        0.30,
    ),
]

class P:
    full_text = "x" * 5000
    abstract = ""
    tldr = ""

for name, claim, evidence, sim in cases:
    chunk = EvidenceChunk(chunk_id=1, text=evidence, semantic_score=sim)
    r = v._heuristic_semantic_nli(claim, [chunk], P())
    print(f"{name:32} -> {r['verdict']:22} conf={r['confidence']} override={r['strong_override']}")
