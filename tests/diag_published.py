import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from academic_fetcher import AcademicFetcher
from document_parser import DocumentParser
from rag_verifier import SemanticRAGVerifier

PDF = Path(r"C:\Users\btuoz\Desktop\YAYINLANMIŞ MAKALE.pdf")
p = DocumentParser().parse_file(PDF, PDF.name)

print("YETİM KAYNAKLAR:")
for o in p.orphan_references:
    print(f"  [{o.ref_id}] {o.raw_text[:140]}")

target = "APA_43"
cit = next(c for c in p.citations if c.matched_ref_id == target)
print("\nİDDİA:", cit.claim_sentence[:400])

f = AcademicFetcher(enable_scihub=False)
paper = f.verify_and_fetch(cit.matched_reference, None)
print("\nKAYNAK:", paper.verified_title, "| özet:", len(paper.abstract))

v = SemanticRAGVerifier(embedding_device="cpu")
chunks = v.retrieve_top_evidence(cit.claim_sentence, paper.combined_evidence_text, top_k=4)
for i, ch in enumerate(chunks, 1):
    print(f"\nKANIT {i} ({ch.semantic_score:.3f}): {ch.text[:400]}")

res = v.verify_claim_against_paper(cit.claim_sentence, paper, cit.citation_id, cit.marker, cit.context_text)
print("\nKARAR:", res.verdict, "| kaynak:", res.decision_source)
print("AÇIKLAMA:", res.explanation[:500])
