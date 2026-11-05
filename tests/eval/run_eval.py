"""ADIM 19 — Altın set değerlendirme koşucusu.

Kullanım:
    .venv\\Scripts\\python.exe tests\\eval\\run_eval.py
    .venv\\Scripts\\python.exe tests\\eval\\run_eval.py --strict   (hatalarda çıkış kodu 1)

İki bölüm ölçer:
  1) Parser: sentetik ama gerçekçi makalelerde kaynak/atıf sayısı, hayalet/yetim.
  2) Karar motoru (deterministik katman): 30 iddia–kanıt çifti, gerçek embedding.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from document_parser import DocumentParser  # noqa: E402
from rag_verifier import SemanticRAGVerifier  # noqa: E402


class EvalPaper:
    full_text = ""
    abstract = ""
    tldr = ""


# ---------------------------------------------------------------------------
# 1) PARSER DEĞERLENDİRMESİ
# ---------------------------------------------------------------------------

def numbered_refs(n: int) -> list[str]:
    out = []
    for i in range(1, n + 1):
        out.append(
            f"{i}. Yilmaz A, Kaya B. Clinical outcomes of treatment modality {i}. "
            f"Turk J Cardiol. 2020;12(3):{40 + i}-{50 + i}. doi:10.1000/tjc.2020.{i:03d}"
        )
    return out


def manuscript(body: str, refs: list[str]) -> str:
    return body.strip() + "\n\nKAYNAKÇA\n" + "\n".join(refs)


PARSER_CASES = []

# 1) Temiz numaralı makale
PARSER_CASES.append(("temiz", manuscript(
    "Birinci bulgu literatürle uyumludur [1]. İkinci bulgu da desteklenmektedir [2]. "
    "Üçüncü bulgu ise yeni bir katkı sunmaktadır [3].",
    numbered_refs(3),
), dict(refs=3, cits=3, ghost=0, orphan=0)))

# 2) Hayalet atıf (kaynakçada olmayan numara)
PARSER_CASES.append(("hayalet", manuscript(
    "Birinci bulgu [1]. İkinci bulgu [2]. Üçüncü bulgu ise kaynakçada olmayan bir numaraya atıf yapar [9].",
    numbered_refs(2),
), dict(refs=2, cits=3, ghost=1, orphan=0)))

# 3) Yetim kaynak (metinde hiç kullanılmayan)
PARSER_CASES.append(("yetim", manuscript(
    "Yalnızca birinci kaynağa atıf yapılmıştır [1].",
    numbered_refs(3),
), dict(refs=3, cits=1, ghost=0, orphan=2)))

# 4) Aralık atıf [1-4]
PARSER_CASES.append(("aralik", manuscript(
    "Literatürde bu konu kapsamlı biçimde incelenmiştir [1-4].",
    numbered_refs(6),
), dict(refs=6, cits=4, ghost=0, orphan=2)))

# 5) APA yazar-yıl
PARSER_CASES.append(("apa", manuscript(
    "Bu konu daha önce ayrıntılı olarak ele alınmıştır (Yılmaz, 2022). "
    "Benzer bir sonuç başka bir çalışmada da bildirilmiştir (Demir ve Kaya, 2021).",
    [
        "Yılmaz, A. (2022). Cardiovascular outcomes in adults. Turk J Cardiol, 12(3), 45-52.",
        "Demir, B., & Kaya, C. (2021). Long-term follow-up of patients. Clin Res J, 8(2), 101-110.",
    ],
), dict(refs=2, cits=2, ghost=0, orphan=0)))

# 6) Aynı yazar, 2022a / 2022b
PARSER_CASES.append(("apa_ab", manuscript(
    "İlk çalışma klinik yanıtı göstermiştir (Iosifescu, 2022a).",
    [
        "Iosifescu, D. V. (2022a). Efficacy of treatment in major depression. J Clin Psychiatry, 83(1), 1-9.",
        "Iosifescu, D. V. (2022b). Safety profile of treatment in major depression. J Clin Psychiatry, 83(2), 10-18.",
    ],
), dict(refs=2, cits=1, ghost=0, orphan=1)))

# 7) DOI'siz Türkçe makale
PARSER_CASES.append(("doisiz", manuscript(
    "Ulusal literatürde bu konu sınırlı sayıda çalışmada ele alınmıştır [1].",
    ["1. Öztürk T. Türkçe akademik makale başlığı. Türk Dergisi. 2019;5(1):10-20."],
), dict(refs=1, cits=1, ghost=0, orphan=0)))

# 8) Kitap kaynağı
PARSER_CASES.append(("kitap", manuscript(
    "Kuramsal çerçeve bu kaynağa dayanmaktadır [1].",
    ["1. Kitap Y. Kitap Adı. Ankara: Yayınevi; 2018."],
), dict(refs=1, cits=1, ghost=0, orphan=0)))


def run_parser_eval() -> tuple[int, int]:
    parser = DocumentParser()
    passed = 0
    print("=" * 78)
    print("PARSER DEĞERLENDİRMESİ (8 senaryo)")
    print("=" * 78)
    for name, text, exp in PARSER_CASES:
        p = parser.parse_text(text, filename=f"{name}.txt")
        got = dict(refs=len(p.references), cits=len(p.citations),
                   ghost=len(p.ghost_citations), orphan=len(p.orphan_references))
        ok = got == exp
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name:10} beklenen={exp}  bulunan={got}")
        if not ok:
            print(f"         uyarılar: {p.warnings}")
    print(f"\nParser: {passed}/{len(PARSER_CASES)} senaryo geçti\n")
    return passed, len(PARSER_CASES)


# ---------------------------------------------------------------------------
# 2) KARAR MOTORU DEĞERLENDİRMESİ
# ---------------------------------------------------------------------------

def run_claim_eval(gold_path: Path) -> tuple[int, int, Counter]:
    cases = json.loads(gold_path.read_text(encoding="utf-8"))
    verifier = SemanticRAGVerifier(embedding_device="cpu")

    strict_ok = 0
    lenient_ok = 0
    confusion: Counter = Counter()
    per_cat = defaultdict(lambda: [0, 0])

    print("=" * 78)
    print(f"KARAR MOTORU DEĞERLENDİRMESİ ({len(cases)} iddia, deterministik katman)")
    print("=" * 78)

    for c in cases:
        chunks = verifier.retrieve_top_evidence(c["claim"], c["evidence"], top_k=3)
        h = verifier._heuristic_semantic_nli(c["claim"], chunks, EvalPaper())
        pred = h["verdict"]
        exp = c["expected"]

        strict = pred == exp
        lenient = strict or (exp == "DESTEKLENİYOR" and pred == "KISMEN_DESTEKLENİYOR")
        strict_ok += strict
        lenient_ok += lenient
        per_cat[c["category"]][0] += lenient
        per_cat[c["category"]][1] += 1
        confusion[(exp, pred)] += 1

        mark = "PASS" if strict else ("~" if lenient else "FAIL")
        print(f"[{mark:4}] {c['id']:3} {c['category']:14} beklenen={exp:17} bulunan={pred:22} sim={chunks[0].semantic_score if chunks else 0:.2f}")

    n = len(cases)
    print("\nKategori bazında (esnek):")
    for cat, (ok, total) in sorted(per_cat.items()):
        print(f"  {cat:14}: {ok}/{total}")
    print(f"\nKatı doğruluk  : {strict_ok}/{n} ({strict_ok / n:.0%})")
    print(f"Esnek doğruluk : {lenient_ok}/{n} ({lenient_ok / n:.0%})  (KISMEN, DESTEKLENİYOR yerine kabul)")

    print("\nKarışıklık matrisi (beklenen -> bulunan):")
    for (exp, pred), cnt in sorted(confusion.items()):
        flag = "" if exp == pred else "  <-- HATA"
        print(f"  {exp:17} -> {pred:22} : {cnt}{flag}")

    return strict_ok, n, confusion


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Hata varsa çıkış kodu 1")
    args = ap.parse_args()

    gold = Path(__file__).with_name("gold_claims.json")
    p_ok, p_total = run_parser_eval()
    c_ok, c_total, _ = run_claim_eval(gold)

    print("\n" + "=" * 78)
    print(f"ÖZET: parser {p_ok}/{p_total} | karar (katı) {c_ok}/{c_total}")
    print("=" * 78)

    if args.strict and (p_ok < p_total or c_ok < c_total):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
