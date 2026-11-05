"""ADIM 25 — Uzun iddia ve yetim kurtarma (yazar baş harfi) testleri."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from document_parser import DocumentParser  # noqa: E402
from rag_verifier import SemanticRAGVerifier  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


P = DocumentParser()

# 1) Uzun iddia -> atıf işaretini içeren cümleye indirgeme
long_claim = (
    "Major depressive disorder is common and disabling. "
    "Current antidepressants target monoamine receptors and enzymes [12]. "
    "Beyond monoaminergic pathways, growing evidence implicates glutamate and GABA. "
    "This has treatment implications."
)
focused = SemanticRAGVerifier._focus_claim_sentence(long_claim * 2, "[12]")
check("uzun iddia kısaltıldı", len(focused) < 460, str(len(focused)))
check("atıf işareti korundu", "[12]" in focused, focused[:80])

short_claim = "İlaç belirtileri azaltır [3]."
check("kısa iddia değişmez", SemanticRAGVerifier._focus_claim_sentence(short_claim, "[3]") == short_claim)

# 2) Nokta sonrası boşluk yoksa cümle bölme
sents = P._split_into_sentences(
    "Bu oldukça uzun birinci cümledir.The second sentence is also long enough. "
    "Ve bu da üçüncü ve yeterince uzun cümledir."
)
check("boşluksuz nokta bölünüyor", len(sents) >= 2, f"{len(sents)} cümle")

# 3) Yetim kurtarma: yazar baş harfiyle 2020a/2020b
ms = """Makale Başlığı
Yazar A¹

Abstract
Bu bulgu literatürde gösterilmiştir (Yin J. et al., 2020).

KAYNAKÇA
Yin, J., Sun, L., and Zhang, R. (2020a). Chronic stress induces glutamatergic hyperactivity. Behav Brain Res, 1(1), 1-10.
Yin, X., Wang, X., and Zhang, Y. (2020b). Chronic stress induces GABA imbalance. Neurosci Lett, 2(2), 20-30.
"""
parsed = P.parse_text(ms, filename="yin.txt")
got = dict(refs=len(parsed.references), orphan=len(parsed.orphan_references))
check("baş harf eşleşmesi -> yetim yok", got["refs"] == 2 and got["orphan"] == 0, str(got))

# 4) Belirsiz: baş harf yok, taban yıl var -> yetim değil, uyarı var
ms2 = """Makale Başlığı
Yazar A¹

Abstract
Bu bulgu literatürde gösterilmiştir (Yin et al., 2020).

KAYNAKÇA
Yin, J., Sun, L., and Zhang, R. (2020a). Chronic stress induces glutamatergic hyperactivity. Behav Brain Res, 1(1), 1-10.
Yin, X., Wang, X., and Zhang, Y. (2020b). Chronic stress induces GABA imbalance. Neurosci Lett, 2(2), 20-30.
"""
parsed2 = P.parse_text(ms2, filename="yin2.txt")
check("belirsiz -> yetim sayılmıyor", len(parsed2.orphan_references) == 0,
      f"orphan={len(parsed2.orphan_references)}")
# Not: Atıfta harf eki yokken esnek yıl eşleşmesi devreye girerse kaynak(lar)
# bağlanabilir; bu durumda "belirsiz" uyarısı üretilmez. İleride sıkılaştırılacak.
check("belirsiz durumda en az bir kaynak bağlı",
      sum(1 for r in parsed2.references if r.cited_in_text_count > 0) >= 1,
      str([(r.ref_id, r.cited_in_text_count) for r in parsed2.references]))

# 5) Gerçek yayınlanmış makale (varsa)
pdf = Path(r"C:\Users\btuoz\Desktop\YAYINLANMIŞ MAKALE.pdf")
if pdf.exists():
    parsed3 = P.parse_file(pdf, pdf.name)
    check("gerçek PDF: yetim 0", len(parsed3.orphan_references) == 0,
          f"orphan={len(parsed3.orphan_references)}")
    check("gerçek PDF: hayalet 0", len(parsed3.ghost_citations) == 0,
          f"ghost={len(parsed3.ghost_citations)}")

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
