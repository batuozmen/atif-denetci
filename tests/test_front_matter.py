"""ADIM 23 — Front-matter (başlık/yazar/kurum) temizliği testleri."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from document_parser import DocumentParser  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


P = DocumentParser()

# 1) Sentetik makale: yazar üst simgeleri atıf sayılmamalı
synthetic = """BAŞLIK: Moleküler Yanıtlar
Ahmet Tektemur¹, Zeliha Turk¹, Nalan Kaya Tektemur²
Department of Medical Biology, Faculty of Medicine, Fırat University, Elazığ
E-mail: test@example.com

Abstract
Bu çalışmada tedavinin etkisi incelenmiştir [1].

Introduction
Literatürde benzer sonuçlar bildirilmiştir [2].

KAYNAKÇA
1. Yilmaz A. Clinical outcomes. Turk J Cardiol. 2020;12(3):45-52.
2. Kaya B. Molecular responses. J Med. 2021;8(2):101-110.
"""
parsed = P.parse_text(synthetic, filename="sentetik.txt")
got = dict(refs=len(parsed.references), cits=len(parsed.citations),
           ghost=len(parsed.ghost_citations), orphan=len(parsed.orphan_references))
expected = dict(refs=2, cits=2, ghost=0, orphan=0)
check("sentetik: künye atıf sayılmıyor", got == expected, str(got))

# 2) _strip_front_matter doğrudan
text = "Başlık\nYazar A, Yazar B\nKurum\n\n" + "x" * 100 + "\n\nAbstract\nGerçek metin [1]."
stripped = P._strip_front_matter(text)
check("başlıktan kesiyor", stripped.startswith("Abstract"), stripped[:30])

no_heading = "Sadece düz metin. Atıf yok."
check("başlık yoksa dokunmuyor", P._strip_front_matter(no_heading) == no_heading)

late_heading = ("x" * 5000) + "\nAbstract\nyeni metin"
check("başlık geç tarihte -> kesmiyor", P._strip_front_matter(late_heading) == late_heading)

# 3) _looks_like_front_matter
long_aff = "Department of Medical Biology, Faculty of Medicine, University of X, " + "a" * 400
check("uzun kurum cümlesi yakalanıyor", P._looks_like_front_matter(long_aff) is True)
short_claim = "İlaç depresyon belirtilerini azaltır ve bu etki klinik çalışmalarda gösterilmiştir."
check("kısa iddia cümlesi etkilenmiyor", P._looks_like_front_matter(short_claim) is False)

# 4) Gerçek makale (varsa)
real = Path(r"C:\Users\btuoz\Desktop\Manuscript.docx")
if real.exists():
    parsed = P.parse_file(real, real.name)
    first = parsed.citations[0].claim_sentence if parsed.citations else ""
    ok = (
        len(parsed.citations) < 175
        and "Department of" not in first
        and "University" not in first
        and len(first) < 400
    )
    check("gerçek makale: künye atıfları düştü", ok,
          f"cits={len(parsed.citations)} ilk={first[:60]!r}")
    check("gerçek makale: hayalet/yetim hâlâ 0",
          len(parsed.ghost_citations) == 0 and len(parsed.orphan_references) == 0,
          f"ghost={len(parsed.ghost_citations)} orphan={len(parsed.orphan_references)}")
else:
    check("gerçek makale mevcut (atlandı)", True, "dosya yok")

passed = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))
print(f"\n{passed}/{len(RESULTS)} test geçti")
raise SystemExit(0 if passed == len(RESULTS) else 1)
