"""Heuristic NLI katmanı: negasyon ve yön testleri."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from rag_verifier import SemanticRAGVerifier, EvidenceChunk  # noqa: E402

v = SemanticRAGVerifier.__new__(SemanticRAGVerifier)
v.ollama_model = "test"

cases = [
    ("TR negasyon", "İlaç depresyon belirtilerini azaltır.", "İlaç depresyon belirtilerini azaltmaz.", 0.90),
    ("EN negasyon", "The drug reduces depressive symptoms.", "The drug does not reduce depressive symptoms.", 0.90),
    ("Yön aynı (TR/EN)", "İlaç depresyon belirtilerini azaltmıştır.", "Treatment decreased depressive symptoms.", 0.75),
    ("Yön aynı (artış)", "İlaç yan etkileri artırmıştır.", "The drug increased adverse effects.", 0.75),
    ("Farklı bulgu, yüksek benzerlik",
     "Bu çalışmada depresyon tedavisinde ilaç X'in etkili olduğu gösterilmiştir.",
     "Bu çalışmada depresyon tedavisinde ilaç X'in etkili olmadığı gösterilmiştir.",
     0.93),
]

class P:
    full_text = "x" * 5000
    abstract = ""
    tldr = ""

for name, claim, evidence, sim in cases:
    chunk = EvidenceChunk(chunk_id=1, text=evidence, semantic_score=sim)
    r = v._heuristic_semantic_nli(claim, [chunk], P())
    print(f"{name:34} -> {r['verdict']:22} conf={r['confidence']} override={r['strong_override']}")
