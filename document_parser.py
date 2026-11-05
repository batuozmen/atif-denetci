"""
MODÜL 1: Akıllı Makale ve Atıf Ayrıştırıcı (document_parser.py)
---------------------------------------------------------------
Görevleri:
1. Word (.docx), PDF (.pdf) veya düz metin (.txt) formatındaki akademik makaleleri okur.
2. Makaleyi "Ana Metin (Body)" ve "Kaynakça (References)" olarak otomatik ayırır.
3. Kaynakça listesindeki her bir makaleyi (Numara, Yazar, Yıl, Başlık, DOI, URL) ayrıştırır.
4. Ana metin içindeki hem Numaralı ([1], [2-5], [1, 4]) hem de APA/Yazar-Yıl
   ((Yılmaz, 2022), (Castrén and Monteggia, 2021), Smith et al. (2021)) atıf cümlelerini
   bağlam penceresiyle (önceki + mevcut + sonraki cümle) yakalar.
5. Metin içi atıfları Kaynakça ile eşleştirir (2022a/2022b, aksanlı harfler é/ä/ö,
   çoklu yıllar 2011, 2012 ve sayfa geçişleri dahil) ve TÜBİTAK Biçimsel Denetimini yapar:
   - Hayalet Atıf (Ghost Citation): Metinde atıf yapılmış ama kaynakçada yok.
   - Yetim Kaynak (Orphan Reference): Kaynakçada var ama metinde hiç atıf yapılmamış.
"""

from __future__ import annotations

import io
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import pymupdf as fitz  # PyMuPDF
import docx  # python-docx


# ---------------------------------------------------------------------------
# Veri Yapıları (Data Structures)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceItem:
    """Kaynakça bölümündeki tek bir referans makaleyi temsil eder."""
    ref_id: str                       # Örn: "1", "2" veya APA için "APA_1"
    number: Optional[int]             # Numaralı kaynak ise sayısal değeri (1, 2...)
    raw_text: str                     # Kaynakçadaki tam metin satırı
    title: str = ""                   # Ayıklanan makale başlığı
    authors: str = ""                 # Ayıklanan yazarlar
    first_author_surname: str = ""    # APA eşleştirmesi için ilk yazar soyadı
    year: Optional[str] = None        # Yayın yılı (Örn: "2021" veya "2022b")
    doi: Optional[str] = None         # Varsa DOI (Örn: "10.1016/j.xxx...")
    url: Optional[str] = None         # Varsa açık link
    cited_in_text_count: int = 0      # Ana metinde kaç kez atıf yapıldı?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InTextCitation:
    """Ana metin içinde yakalanan bir atıfı ve iddia cümlesini temsil eder."""
    citation_id: str                  # Tekil kimlik: "CIT_1", "CIT_2"...
    marker: str                       # Metinde geçen atıf ifadesi: "[1]", "(Yılmaz, 2022)"
    citation_style: str               # "numeric" veya "author_year"
    claim_sentence: str               # Atıfın geçtiği ana iddia cümlesi
    context_text: str                 # Önceki + Ana + Sonraki cümle (anlam bütünlüğü için)
    matched_ref_id: Optional[str]     # Eşleşen ReferenceItem.ref_id (eşleşmediyse None)
    matched_reference: Optional[ReferenceItem] = None
    expected_key: str = ""            # Aranan numara ("[1]") veya yazar-yıl ("Yılmaz, 2022")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedManuscript:
    """Ayrıştırılmış makalenin tüm sonuçlarını ve biçimsel denetim raporunu tutar."""
    filename: str
    total_chars: int
    body_text: str
    references_text: str
    references: List[ReferenceItem] = field(default_factory=list)
    citations: List[InTextCitation] = field(default_factory=list)
    ghost_citations: List[InTextCitation] = field(default_factory=list)   # Metinde var, kaynakçada yok
    orphan_references: List[ReferenceItem] = field(default_factory=list)  # Kaynakçada var, metinde yok
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "total_chars": self.total_chars,
            "references_count": len(self.references),
            "citations_count": len(self.citations),
            "ghost_citations_count": len(self.ghost_citations),
            "orphan_references_count": len(self.orphan_references),
            "warnings": self.warnings,
            "references": [r.to_dict() for r in self.references],
            "citations": [c.to_dict() for c in self.citations],
            "ghost_citations": [g.to_dict() for g in self.ghost_citations],
            "orphan_references": [o.to_dict() for o in self.orphan_references],
        }


# ---------------------------------------------------------------------------
# Yardımcı Regex ve Unicode Harf Tanımları (Tüm Dünya Harflerini Destekler)
# ---------------------------------------------------------------------------

# Tüm büyük ve küçük harfler (Türkçe + Avrupa aksanlı harfler: é, á, ó, ñ, ä, ü, ß, ć, š, ł vb.)
UPPER_LETTER = r"[A-ZÀ-ÖØ-ÞĀĂĄĆĈĊČĎĐĒĔĖĘĚĜĞĠĢĤĦĨĪĬĮİĲĴĶĹĻĽĿŁŃŅŇŊŌŎŐŒŔŖŘŚŜŞŠŢŤŦŨŪŬŮŰŲŴŶŸŹŻŽÇĞİÖŞÜ]"
ANY_LETTER = r"[^\W\d_]"

# Kaynakça başlığını yakalayan desenler (Türkçe & İngilizce)
REF_HEADING_PATTERNS = [
    r"^\s*(?:\d+\.?\s*|[IVX]+\.?\s*)?(?:KAYNAKLAR|KAYNAKÇA|REFERANSLAR|REFERENCES|BIBLIOGRAPHY|LITERATURE\s+CITED|WORKS\s+CITED)\s*:?\s*$"
]

# DOI yakalama deseni
DOI_REGEX = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", re.IGNORECASE)

# URL yakalama deseni
URL_REGEX = re.compile(r"(https?://[^\s\)\]>\"']+)", re.IGNORECASE)

# Yıl yakalama deseni (1950 - 2030 arası, sonuna a/b/c gelebilir: 2021a, 2022b)
YEAR_REGEX = re.compile(r"\b((?:19[5-9]\d|20[0-3]\d)[a-c]?)\b")

# Tire / kısa çizgi / uzun çizgi / matematiksel eksi karakterlerinin tamamı (Word ve PDF uyumlu)
DASH_CHARS = r"\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

# Numaralı atıf deseni (Köşeli parantez): [1], [1, 2], [1-3], [2, 5–8]
NUMERIC_CITATION_REGEX = re.compile(
    rf"\[(\s*\d{{1,3}}\s*(?:[{DASH_CHARS},;\s]\s*\d{{1,3}}\s*)*)\]"
)

# Numaralı atıf deseni (Yuvarlak parantez - Vancouver/Tıp formatı): (1), (1, 2), (1-3), (2, 5–8)
PAREN_NUMERIC_CITATION_REGEX = re.compile(
    rf"\((\s*\d{{1,3}}\s*(?:[{DASH_CHARS},;]\s*\d{{1,3}}\s*)*)\)"
)

# APA Parantez içi atıf deseni: (Yılmaz, 2021), (Castrén and Monteggia, 2021; Smith et al., 2020)
APA_PAREN_REGEX = re.compile(
    r"\(([^()]{3,600}?(?:19[5-9]\d|20[0-3]\d)[a-c]?[^()]*)\)"
)

# APA Metin içi (Narrative) atıf deseni (En az 2 harfli soyadı zorunlu - 'H (2026)' gibi tek harfli baş harfleri eler):
APA_INLINE_REGEX = re.compile(
    rf"\b({UPPER_LETTER}{ANY_LETTER}+(?:[\-']{ANY_LETTER}+)*"
    rf"(?:\s+(?:ve\s+ark\.?|vd\.?|et\s+al\.?|(?:ve|and|&)\s+{UPPER_LETTER}{ANY_LETTER}+(?:[\-']{ANY_LETTER}+)*))?)"
    rf",?\s*\(\s*((?:19[5-9]\d|20[0-3]\d)[a-c]?(?:\s*,\s*(?:19[5-9]\d|20[0-3]\d|[a-c]))*)\s*\)"
)


def normalize_surname_token(text: str) -> str:
    """
    Soyadlarındaki şapkalı/aksanlı harfleri (Castrén -> castren) ve tireleri
    (Yankelevitch-Yahav -> yankelevitchyahav) standartlaştırır.
    Türkçe ı/ğ/ş/ç/ö/ü karakterlerini de koruyarak veya ASCII karşılığıyla eşleştirir.
    """
    if not text:
        return ""
    s = text.lower().strip()
    # Türkçe özel dönüşümler
    tr_map = str.maketrans({"ı": "i", "ğ": "g", "ş": "s", "ç": "c", "ö": "o", "ü": "u"})
    s = s.translate(tr_map)
    # Avrupa aksanlı harfleri (é -> e, á -> a, ñ -> n) taban harfe indirge
    s = "".join(
        ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z]", "", s)


# ---------------------------------------------------------------------------
# Ana Ayrıştırıcı Sınıfı (DocumentParser)
# ---------------------------------------------------------------------------

class DocumentParser:
    """Word (.docx), PDF (.pdf) ve düz metin makalelerini ayrıştırır."""

    def parse_file(self, file_path_or_bytes: Any, filename: str = "makale.docx") -> ParsedManuscript:
        """Dosya yolundan veya bellek içi byte akışından makaleyi okur ve ayrıştırır."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            raw_text = self._read_pdf(file_path_or_bytes)
        elif ext in (".docx", ".doc"):
            raw_text = self._read_docx(file_path_or_bytes)
        else:
            if isinstance(file_path_or_bytes, (str, Path)) and Path(file_path_or_bytes).exists():
                raw_text = Path(file_path_or_bytes).read_text(encoding="utf-8", errors="ignore")
            elif isinstance(file_path_or_bytes, bytes):
                raw_text = file_path_or_bytes.decode("utf-8", errors="ignore")
            else:
                raw_text = str(file_path_or_bytes)

        return self.parse_text(raw_text, filename=filename)

    def _read_pdf(self, source: Any) -> str:
        """
        PyMuPDF ile PDF metnini okur; sayfa üstbilgisi/altbilgisi gürültülerini
        temizler ki sayfa geçişlerinde bölünen parantez içi atıflar kopmasın!
        """
        if isinstance(source, (str, Path)):
            doc = fitz.open(str(source))
        elif isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            doc = fitz.open(stream=source.read(), filetype="pdf")

        pages_text = []
        doc_self_doi: Optional[str] = None
        if len(doc) > 0:
            first_page_txt = doc[0].get_text("text")
            m_self_doi = DOI_REGEX.search(first_page_txt)
            if m_self_doi:
                doc_self_doi = m_self_doi.group(1).rstrip(".,;)").lower()

        for page in doc:
            text = page.get_text("text")
            clean_lines = []
            for ln in text.split("\n"):
                s = ln.strip()
                if not s:
                    clean_lines.append("")
                    continue
                # Tek başına sayfa numaralarını atla ("05", "11" vb.)
                if re.match(r"^\d{1,3}$", s):
                    continue
                # Dergi sayfa altbilgisi / üstbilgisi satırlarını atla
                if re.search(r"\bfrontiersin\.org\b|^Frontiers in \w+|Volume \d+\s*\|\s*Article", s, re.IGNORECASE):
                    continue
                # Çok kelimeli yazar soyadlarını da kapsayan üstbilgi (Örn: "Buran Kavuran et al. 10.3389/...")
                if re.match(r"^.{2,55}\bet\s+al\.\s*(?:10\.\d{4,9}/\S+)?$", s, re.IGNORECASE):
                    continue
                # Makalenin kendi DOI'si sayfa üst/alt bilgisinde tek başına duruyorsa atla
                if doc_self_doi and doc_self_doi in s.lower() and len(s) < len(doc_self_doi) + 25:
                    continue
                clean_lines.append(ln)

            page_clean = "\n".join(clean_lines)
            # Tireli özel isimleri/soyadlarını koru: "Yankelevitch-\nYahav" -> "Yankelevitch-Yahav"
            page_clean = re.sub(
                rf"({UPPER_LETTER}{ANY_LETTER}+)-\s*\n\s*({UPPER_LETTER}{ANY_LETTER}+)",
                r"\1-\2",
                page_clean,
            )
            # Satır sonundaki küçük harfli kelime bölünmelerini birleştir: "algo-\nritma" -> "algoritma"
            page_clean = re.sub(rf"({ANY_LETTER})-\s*\n\s*([a-zß-öø-ÿçğıöşü])", r"\1\2", page_clean)
            pages_text.append(page_clean)

        doc.close()
        return "\n".join(pages_text)

    def _read_docx(self, source: Any) -> str:
        """
        python-docx ile Word belgesindeki paragrafları, tabloları,
        EndNote/Mendeley/Zotero alan kutularını (w:sdt / w:fldSimple / w:hyperlink)
        ve üst simge (superscript) numaralı atıfları eksiksiz okur.
        """
        if isinstance(source, (str, Path)):
            document = docx.Document(str(source))
        elif isinstance(source, bytes):
            document = docx.Document(io.BytesIO(source))
        else:
            document = docx.Document(source)

        paragraphs: List[str] = []
        for p in document.paragraphs:
            p_txt = self._extract_docx_paragraph_text(p._p).strip()
            if p_txt:
                paragraphs.append(p_txt)

        # Tablo hücrelerindeki metinleri de ekle (tablo içi atıfları kaçırmamak için)
        for table in document.tables:
            for row in table.rows:
                cell_texts: List[str] = []
                for cell in row.cells:
                    c_parts = [
                        self._extract_docx_paragraph_text(cp._p).strip()
                        for cp in cell.paragraphs
                    ]
                    c_txt = " ".join(x for x in c_parts if x)
                    if c_txt and (not cell_texts or c_txt != cell_texts[-1]):
                        cell_texts.append(c_txt)
                row_txt = " ".join(cell_texts).strip()
                if row_txt:
                    paragraphs.append(row_txt)
        return "\n".join(paragraphs)

    def _extract_docx_paragraph_text(self, p_elem: Any) -> str:
        """
        Bir Word paragrafının (<w:p>) XML ağacındaki tüm <w:r> düğümlerini tarar:
        - Mendeley / Zotero / EndNote alan kutuları (<w:sdt>, <w:fldSimple>, <w:hyperlink>, <w:ins>) içini okur.
        - Silinmiş değişiklikleri (<w:del>) ve ham alan kodlarını (<w:instrText>) atlar.
        - Üst simge (<w:vertAlign w:val="superscript"/>) olarak yazılmış atıf rakamlarını
          (örn: ¹‚² veya 1-4) kimyasal formül ve birimlerle (cm², Ca²⁺, 10⁶) karıştırmadan [1, 2] formatına çevirir.
        """
        w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        segments: List[Tuple[bool, str]] = []

        for r in p_elem.iter(f"{w_ns}r"):
            # Silinmiş (Track Changes <w:del>) veya alıntı alanı tanımı içindeyse atla
            parent = r.getparent()
            in_del = False
            while parent is not None and parent != p_elem:
                if parent.tag == f"{w_ns}del":
                    in_del = True
                    break
                parent = parent.getparent()
            if in_del:
                continue

            # Sadece görünür <w:t> ve <w:tab>/<w:br> metinlerini al (<w:instrText> alan kodlarını alma)
            t_chunks: List[str] = []
            for child in r:
                if child.tag == f"{w_ns}t" and child.text:
                    t_chunks.append(child.text)
                elif child.tag == f"{w_ns}tab":
                    t_chunks.append(" ")
                elif child.tag == f"{w_ns}br":
                    t_chunks.append("\n")
            run_text = "".join(t_chunks)
            if not run_text:
                continue

            is_sup = False
            rPr = r.find(f"{w_ns}rPr")
            if rPr is not None:
                va = rPr.find(f"{w_ns}vertAlign")
                if va is not None and va.get(f"{w_ns}val") == "superscript":
                    is_sup = True

            if segments and segments[-1][0] == is_sup:
                segments[-1] = (is_sup, segments[-1][1] + run_text)
            else:
                segments.append((is_sup, run_text))

        non_citation_units = {
            "cm", "mm", "m", "km", "dm", "nm", "um", "µm", "g", "kg", "mg", "ug", "µg",
            "l", "ml", "ul", "µl", "dl", "mol", "mmol", "umol", "µmol", "s", "sec", "min",
            "h", "hr", "d", "wk", "yr", "ca", "fe", "cu", "zn", "mg", "mn", "se", "na",
            "k", "cl", "o", "h", "n", "c", "s", "p", "o2", "h2o", "co2", "h2o2", "so4",
            "po4", "nh4", "nad", "nadp", "atp", "adp", "x", "y", "z", "r", "t", "f", "chi", "χ"
        }

        out_parts: List[str] = []
        for is_sup, seg_txt in segments:
            if not is_sup:
                out_parts.append(seg_txt)
                continue

            sup_clean = seg_txt.strip()
            sup_match = re.match(
                rf"^\[?\(?(\d{{1,3}}(?:\s*[{DASH_CHARS},;]\s*\d{{1,3}})*)\)?\]?$",
                sup_clean,
            )
            if not sup_match:
                out_parts.append(seg_txt)
                continue

            inner_nums = sup_match.group(1).strip()
            prev_text = "".join(out_parts)
            prev_stripped = prev_text.rstrip()

            # Öncesinde rakam varsa (örn: 10^6) bu üslü sayıdır, atıf değildir
            if prev_stripped and prev_stripped[-1].isdigit():
                out_parts.append(seg_txt)
                continue

            # Tek bir rakamsa ve hemen önünde birim/kimyasal sembol varsa (örn: cm², Ca²) atla
            is_multi_or_range = bool(re.search(rf"[{DASH_CHARS},;]", inner_nums))
            last_word_m = re.search(rf"({ANY_LETTER}+)$", prev_stripped)
            last_word = last_word_m.group(1).lower() if last_word_m else ""

            if not is_multi_or_range:
                if last_word in non_citation_units or (len(last_word) == 1 and last_word not in {"a", "i"}):
                    out_parts.append(seg_txt)
                    continue

            # Geçerli üst simge atıfı -> [1, 2] formatına dönüştür
            out_parts.append(f" [{inner_nums}] ")

        full_para = "".join(out_parts)
        return self._normalize_unicode_superscripts(full_para)

    def _normalize_unicode_superscripts(self, text: str) -> str:
        """Metin içinde doğrudan Unicode üst simge (¹²³⁴⁵⁶⁷⁸⁹⁰) ile yazılmış atıfları [1, 2] haline getirir."""
        sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻−", "0123456789--")
        sup_chars = r"⁰¹²³⁴⁵⁶⁷⁸⁹"
        pattern = re.compile(
            rf"({ANY_LETTER}{{3,}}|[\.,;\)\”\"])\s*([{sup_chars}]{{1,3}}(?:\s*[,;⁻−–\-]\s*[{sup_chars}]{{1,3}})*)"
        )

        def _repl(m: re.Match) -> str:
            converted = m.group(2).translate(sup_map)
            return f"{m.group(1)} [{converted}] "

        return pattern.sub(_repl, text)

    def parse_text(self, full_text: str, filename: str = "makale.txt") -> ParsedManuscript:
        """Tam metni Ana Metin ve Kaynakça olarak ayırır, atıfları ve referansları eşleştirir."""
        cleaned_text = full_text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_text = self._normalize_unicode_superscripts(cleaned_text)
        body_text, references_text = self._split_body_and_references(cleaned_text)

        warnings: List[str] = []
        if not references_text.strip():
            warnings.append(
                "Makalede 'Kaynaklar / Kaynakça / References' başlığı otomatik bulunamadı. "
                "Son bölüm kaynakça olarak taranmaya çalışıldı."
            )

        references = self._parse_references_section(references_text)
        citations = self._extract_in_text_citations(body_text, references, references_text)

        # İkinci Güvenlik Taraması (Yetim Kaynak Doğrulaması):
        # Eğer bir kaynak ilk taramada eşleşmediyse, yazar soyadı + yıl ana metinde geçiyor mu kontrol et
        self._second_pass_orphan_recovery(body_text, references, citations)

        # Kesin Tutarlılık Kuralı: Her kaynağın kullanım sayısı (cited_in_text_count),
        # kendisine bağlanan gerçek kanıt cümlelerinin (citations) sayısına birebir eşitlenir.
        for r in references:
            r.cited_in_text_count = sum(1 for c in citations if c.matched_ref_id == r.ref_id)

        ghost_citations = [c for c in citations if c.matched_ref_id is None]
        orphan_references = [r for r in references if r.cited_in_text_count == 0]

        if ghost_citations:
            warnings.append(
                f"Metin içinde atıf yapılan {len(ghost_citations)} kaynak, Kaynakça listesinde bulunamadı (Hayalet Atıf)!"
            )
        if orphan_references:
            warnings.append(
                f"Kaynakça listesindeki {len(orphan_references)} makaleye ana metin içinde hiç atıf yapılmamış (Yetim Kaynak)!"
            )

        return ParsedManuscript(
            filename=filename,
            total_chars=len(cleaned_text),
            body_text=body_text,
            references_text=references_text,
            references=references,
            citations=citations,
            ghost_citations=ghost_citations,
            orphan_references=orphan_references,
            warnings=warnings,
        )

    # -----------------------------------------------------------------------
    # 1. Ana Metin ve Kaynakça Ayırma
    # -----------------------------------------------------------------------

    def _split_body_and_references(self, text: str) -> Tuple[str, str]:
        """Metindeki son 'Kaynaklar / Kaynakça / References' başlığını bulup metni ikiye böler."""
        lines = text.split("\n")
        split_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 45:
                continue
            for pat in REF_HEADING_PATTERNS:
                if re.match(pat, stripped, flags=re.IGNORECASE):
                    split_idx = i

        if split_idx != -1:
            body = "\n".join(lines[:split_idx]).strip()
            refs = "\n".join(lines[split_idx + 1 :]).strip()
            return body, refs

        # Eğer başlık yoksa fakat sonda [1], [2] diye sıralı liste varsa oradan böl
        for i in range(len(lines) - 1, max(0, len(lines) - 150), -1):
            if re.match(rf"^\s*(?:\[1\]|1\.)\s+{UPPER_LETTER}", lines[i]):
                body = "\n".join(lines[:i]).strip()
                refs = "\n".join(lines[i:]).strip()
                return body, refs

        return text, ""

    # -----------------------------------------------------------------------
    # 2. Kaynakça Maddelerini Ayrıştırma (Aksanlı Harf + Birleşik Kaynak Ayırıcı)
    # -----------------------------------------------------------------------

    def _parse_references_section(self, refs_text: str) -> List[ReferenceItem]:
        """Kaynakça metnini tekil ReferenceItem nesnelerine dönüştürür."""
        if not refs_text.strip():
            return []

        lines = [ln.strip() for ln in refs_text.split("\n") if ln.strip()]
        raw_entries: List[str] = []

        # Numaralı format mı ([1], 1., (1)) yoksa APA paragraf formatı mı kontrol et
        numbered_start_regex = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})[\.\)]\s+)(.+)$")
        numbered_hits = sum(1 for ln in lines if numbered_start_regex.match(ln))

        if numbered_hits >= max(1, len(lines) // 4):
            current_entry = ""
            for ln in lines:
                if numbered_start_regex.match(ln):
                    if current_entry:
                        raw_entries.append(current_entry.strip())
                    current_entry = ln
                else:
                    current_entry = (current_entry + " " + ln).strip() if current_entry else ln
            if current_entry:
                raw_entries.append(current_entry.strip())
        else:
            # APA / Yazar-Yıl formatı (Castrén, E., Hayashi, T., van der Kolk, B. dahil)
            apa_start_regex = re.compile(
                rf"^(?:(?:van|von|de|del|della|di|le|la|el|al|st\.?)\s+)*"
                rf"{UPPER_LETTER}{ANY_LETTER}*(?:[\-'\s]{ANY_LETTER}+)?,\s*{UPPER_LETTER}"
            )
            pub_year_regex = re.compile(
                r"\(\s*(?:19[5-9]\d|20[0-3]\d)[a-c]?\s*\)|\.\s*(?:19[5-9]\d|20[0-3]\d)[a-c]?\."
            )
            current_entry = ""
            for ln in lines:
                if len(ln) < 4 or re.match(r"^\d{1,3}$", ln):
                    continue
                if "frontiersin.org" in ln.lower() or re.match(r"^frontiers in ", ln, re.IGNORECASE):
                    continue

                is_new_author_start = bool(apa_start_regex.match(ln))
                has_completed_prev = bool(
                    current_entry
                    and (
                        pub_year_regex.search(current_entry)
                        or DOI_REGEX.search(current_entry)
                    )
                )

                # Eğer yeni satır "Yazar, A." ile başlıyor VE önceki kaynakta zaten (Yıl) varsa,
                # ANCAK yeni satır aslında aynı kaynağın devam eden yazar listesi değilse (yani önceki kaynakta (Yıl) geçildiyse) ayır!
                if is_new_author_start and has_completed_prev:
                    raw_entries.append(current_entry.strip())
                    current_entry = ln
                else:
                    if not current_entry:
                        current_entry = ln
                    elif re.search(
                        r"(?:doi\s*:?\s*|doi\s*:?\s*10\.|10\.\d{4,9}/\S*[/\.\-\(\)]|https?://\S*[/\.\-])$",
                        current_entry,
                        re.IGNORECASE,
                    ):
                        current_entry = current_entry + ln
                    else:
                        current_entry = (current_entry + " " + ln).strip()
            if current_entry:
                raw_entries.append(current_entry.strip())

            # İkincil Ayırıcı (Post-Splitter):
            # Çift sütunlu PDF'lerde iki kaynak aynı satırda/blokta birbirine yapıştıysa
            # (Örn: "... doi: 10.1016/xxx Hayashi, T. (2015). Title..."), ikinci "(YYYY)" kaynağından ayır!
            split_entries: List[str] = []
            glued_split_pattern = re.compile(
                rf"(?<=\S)\s+(?=(?:(?:van|von|de)\s+)*{UPPER_LETTER}{ANY_LETTER}+,\s*{UPPER_LETTER}\.[^()]{{0,220}}\(\s*(?:19[5-9]\d|20[0-3]\d)[a-c]?\s*\))"
            )
            for entry in raw_entries:
                year_parens = re.findall(r"\(\s*(?:19[5-9]\d|20[0-3]\d)[a-c]?\s*\)", entry)
                if len(year_parens) >= 2:
                    sub_parts = [sp.strip() for sp in glued_split_pattern.split(entry) if len(sp.strip()) > 15]
                    split_entries.extend(sub_parts)
                else:
                    split_entries.append(entry)
            raw_entries = split_entries

        parsed_refs: List[ReferenceItem] = []
        for idx, entry in enumerate(raw_entries, start=1):
            parsed = self._parse_single_reference(entry, fallback_idx=idx)
            if parsed:
                parsed_refs.append(parsed)

        return parsed_refs

    def _parse_single_reference(self, raw_entry: str, fallback_idx: int) -> Optional[ReferenceItem]:
        """Tek bir kaynakça satırından numara, yazar, yıl, başlık, DOI ve URL bilgilerini çıkarır."""
        text = re.sub(r"\s+", " ", raw_entry).strip()
        if len(text) < 12:
            return None

        # Satır sonunda bölündüğü için arasına boşluk girmiş DOI numaralarını onar:
        # Örn: "doi: 10. 1016/j.biopsych. 2021.05.008" -> "doi: 10.1016/j.biopsych.2021.05.008"
        text = re.sub(r"\b(doi\s*:?\s*10\.)\s+(\d{4,9}/\S+)", r"\1\2", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(10\.)\s+(\d{4,9}/\S+)", r"\1\2", text)
        for _ in range(2):
            text = re.sub(
                r"\b(10\.\d{4,9}/[^\s,;]+?[/\.\-\(\)])\s+([a-z0-9\(\)\-\._/]{2,})",
                r"\1\2",
                text,
                flags=re.IGNORECASE,
            )

        # 1. Numara kontrolü ([1], 1., (1)) — DOI ("10.1016/...") ile karışmaması için boşluk zorunlu
        num_match = re.match(r"^\s*(?:\[(\d{1,3})\]\s*|(\d{1,3})[\.\)]\s+)(.+)$", text)
        number: Optional[int] = None
        clean_body = text
        if num_match:
            num_str = num_match.group(1) or num_match.group(2)
            number = int(num_str)
            ref_id = str(number)
            clean_body = num_match.group(3).strip()
        else:
            ref_id = f"APA_{fallback_idx}"

        # 2. DOI ve URL ayıklama
        doi_match = DOI_REGEX.search(clean_body)
        doi = doi_match.group(1).rstrip(".,;)") if doi_match else None

        url_match = URL_REGEX.search(clean_body)
        url = url_match.group(1).rstrip(".,;)") if url_match else None
        if not doi and url and "doi.org/" in url.lower():
            doi_part = url.lower().split("doi.org/")[-1].strip("/")
            if doi_part.startswith("10."):
                doi = doi_part

        # 3. Yıl ayıklama (Önce parantez içi yılı (2021) veya (2022b) ara, yoksa DOI hariç metinde ara)
        body_no_doi = URL_REGEX.sub("", DOI_REGEX.sub("", clean_body))
        paren_year_m = re.search(r"\(\s*((?:19[5-9]\d|20[0-3]\d)[a-c]?)\s*\)", body_no_doi)
        if paren_year_m:
            year = paren_year_m.group(1)
        else:
            year_match = YEAR_REGEX.search(body_no_doi)
            year = year_match.group(1) if year_match else None

        # 4. Yazar ve Başlık ayıklama
        authors, title, first_surname = self._extract_authors_and_title(clean_body, year)

        return ReferenceItem(
            ref_id=ref_id,
            number=number,
            raw_text=text,
            title=title,
            authors=authors,
            first_author_surname=first_surname,
            year=year,
            doi=doi,
            url=url,
        )

    def _extract_authors_and_title(self, clean_body: str, year: Optional[str]) -> Tuple[str, str, str]:
        """Kaynak satırından yazarları, makale başlığını ve ilk yazarın soyadını ayırır."""
        work_text = DOI_REGEX.sub("", clean_body)
        work_text = URL_REGEX.sub("", work_text)
        work_text = re.sub(r"\bdoi\s*:?\s*", "", work_text, flags=re.IGNORECASE).strip()

        authors = ""
        title = ""

        quote_match = re.search(r"[\"“”]([^\"“”]{12,250})[\"“”]", work_text)
        if quote_match:
            title = quote_match.group(1).strip(" .,;")
            authors = work_text[: quote_match.start()].strip(" .,;()")
        else:
            apa_split = re.search(r"\(\s*(?:19[5-9]\d|20[0-3]\d)[a-c]?\s*\)\s*[\.\,:]?\s*", work_text)
            if apa_split:
                authors = work_text[: apa_split.start()].strip(" .,;")
                rest = work_text[apa_split.end() :].strip()
                parts = [p.strip() for p in re.split(r"\.\s+", rest) if len(p.strip()) > 5]
                title = parts[0].strip(" .") if parts else rest
            else:
                parts = [p.strip() for p in re.split(r"\.\s+", work_text) if len(p.strip()) > 3]
                if len(parts) >= 2:
                    authors = parts[0]
                    title = parts[1].strip(" .")
                else:
                    comma_parts = [p.strip() for p in work_text.split(",") if len(p.strip()) > 4]
                    if len(comma_parts) >= 3:
                        authors = ", ".join(comma_parts[:2])
                        title = comma_parts[2]
                    else:
                        title = work_text[:180]

        first_surname = ""
        if authors:
            first_author_chunk = re.split(r"\s+(?:ve|and|&)\s+|;", authors)[0].strip()
            if "," in first_author_chunk:
                first_surname = first_author_chunk.split(",")[0].strip()
            else:
                first_surname = first_author_chunk.strip()

            # Vancouver formatındaki sondaki büyük harfli isim kısaltmalarını temizle
            # Örn: "Fauser BCJM" -> "Fauser", "Eisenberg ML" -> "Eisenberg", "Vander Borght M" -> "Vander Borght"
            first_surname = re.sub(r"\s+[A-ZÇĞİÖŞÜ]{1,5}(?:-[A-ZÇĞİÖŞÜ]{1,3})?\.?$", "", first_surname).strip()
            if " " in first_surname:
                # "Vander Borght" gibi soyadlarında son parçayı veya birleşik hali koru
                tokens = [t.strip(".") for t in first_surname.split() if len(t.strip(".")) > 1]
                if tokens and tokens[0].lower() not in {"van", "von", "de", "del", "della", "di", "le", "la", "el", "al", "st", "vander"}:
                    first_surname = tokens[-1]

        first_surname = re.sub(rf"[^a-zA-ZÀ-ÖØ-öø-ÿĀ-žçğıöşüÇĞİÖŞÜ\-']", "", first_surname)
        return authors, title, first_surname

    # -----------------------------------------------------------------------
    # 3. Ana Metin İçindeki Atıfları Yakalama ve Kaynakçayla Eşleştirme
    # -----------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """Metni akademik kısaltmaları (vb., ark., vd., et al., Dr., Şek., Fig.) bozmadan cümlelere böler."""
        # Bölüm başlıklarını (1 Introduction, 2 Materials and methods, KEYWORDS vb.) paragraf sonu (\n\n) ile ayır
        text = re.sub(
            r"(?m)^\s*(?:KEYWORDS|Keywords\s*:?|\d+(?:\.\d+)*\.?\s+(?:Introduction|Materials\s+and\s+methods|Methods|Results|Discussion|Conclusion|Conclusions|Giriş|Yöntem|Bulgular|Tartışma|Sonuç))\s*$",
            r"\n\n\g<0>\n\n",
            text,
            flags=re.IGNORECASE,
        )
        joined_lines = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        joined_lines = re.sub(r"[ \t]+", " ", joined_lines)

        # Noktadan hemen sonra gelen atıfları (örn: "...infertility. (1, 2)" veya "...infertility. [1, 2]")
        # noktanın önüne çek ki cümle bölünürken atıf kendi cümlesinden kopmasın!
        joined_lines = re.sub(
            rf"([.!?])\s*(\[\s*\d{{1,3}}(?:\s*[{DASH_CHARS},;]\s*\d{{1,3}})*\s*\]|\(\s*\d{{1,3}}(?:\s*[{DASH_CHARS},;]\s*\d{{1,3}})*\s*\))",
            r" \2\1",
            joined_lines,
        )

        protected = re.sub(
            r"\b(ve\s+ark|ark|vd|vb|et\s+al|Dr|Prof|Doç|Şek|Tab|Fig|Eq|vs|yy|No|Vol|pp|Ed|al)\.",
            r"\1<DOT>",
            joined_lines,
            flags=re.IGNORECASE,
        )
        protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

        raw_sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", protected)
        sentences: List[str] = []
        for s in raw_sentences:
            restored = s.replace("<DOT>", ".").strip()
            if not restored:
                continue
            # Cümle başına yapışmış KEYWORDS ... 1 Introduction veya bölüm başlıklarını temizle
            restored = re.sub(
                r"^.*?\b(?:\d+\s+Introduction|1\.\s*Introduction|1\s+GİRİŞ|1\.\s*Giriş)\s+",
                "",
                restored,
                flags=re.IGNORECASE,
            ).strip()
            restored = re.sub(
                r"^\d+(?:\.\d+)*\.?\s+(?:Materials\s+and\s+methods|Methods|Results|Discussion|Conclusion|Conclusions|Animals|Experimental\s+design|Statistical\s+analysis)\s+",
                "",
                restored,
                flags=re.IGNORECASE,
            ).strip()
            if len(restored) > 15:
                sentences.append(restored)
            elif re.search(rf"[\[\(]\s*\d{{1,3}}\s*(?:[{DASH_CHARS},;]\s*\d{{1,3}}\s*)*[\]\)]", restored):
                if sentences:
                    sentences[-1] = f"{sentences[-1]} {restored}"
                else:
                    sentences.append(restored)
        return sentences

    def _extract_in_text_citations(
        self, body_text: str, references: List[ReferenceItem], references_text: str = ""
    ) -> List[InTextCitation]:
        """Ana metindeki tüm cümleleri tarar, atıfları yakalar ve kaynakçayla eşleştirir."""
        sentences = self._split_into_sentences(body_text)
        citations: List[InTextCitation] = []

        num_map: Dict[int, ReferenceItem] = {
            r.number: r for r in references if r.number is not None
        }
        max_ref_num = max(num_map.keys()) if num_map else 0

        non_citation_paren_labels = {
            "fig", "figure", "figures", "table", "tables", "tablo", "şekil", "sekil",
            "eq", "equation", "denklem", "model", "step", "stage", "group", "groups",
            "phase", "type", "grade", "class", "score", "day", "days", "week", "weeks",
            "month", "months", "year", "years", "age", "n", "p", "sd", "se", "sem",
            "mean", "median", "iqr", "range", "scale", "level", "no", "number", "version"
        }

        cit_counter = 1
        for idx, sent in enumerate(sentences):
            prev_sent = sentences[idx - 1] if idx > 0 else ""
            next_sent = sentences[idx + 1] if idx + 1 < len(sentences) else ""
            context_window = " ".join(p for p in [prev_sent, sent, next_sent] if p).strip()

            # A1) Köşeli parantezli numaralı atıfları yakala: [1], [2, 5], [3-6]
            for match in NUMERIC_CITATION_REGEX.finditer(sent):
                marker = match.group(0)
                inner = match.group(1)
                expanded_nums = self._expand_numeric_citation(inner)
                for num in expanded_nums:
                    if num > 500 or num <= 0:
                        continue
                    matched_ref = num_map.get(num)
                    if matched_ref:
                        matched_ref.cited_in_text_count += 1

                    citations.append(
                        InTextCitation(
                            citation_id=f"CIT_{cit_counter}",
                            marker=marker,
                            citation_style="numeric",
                            claim_sentence=sent,
                            context_text=context_window,
                            matched_ref_id=matched_ref.ref_id if matched_ref else None,
                            matched_reference=matched_ref,
                            expected_key=f"[{num}]",
                        )
                    )
                    cit_counter += 1

            # A2) Yuvarlak parantezli numaralı atıfları yakala (Vancouver formatı): (1), (2, 5), (3–6)
            # Sadece kaynakçası numaralı olan makalelerde (örn. Manuscript.docx) devrededir.
            if num_map:
                upper_limit = max(max_ref_num + 20, 50)
                for match in PAREN_NUMERIC_CITATION_REGEX.finditer(sent):
                    marker = match.group(0)
                    inner = match.group(1).strip()

                    # Hemen önünde rakam varsa (örn: 30(2) cilt/sayı) veya hemen arkasında :rakam varsa atla
                    if match.start() > 0 and sent[match.start() - 1].isdigit():
                        continue
                    after_str = sent[match.end() :]
                    if re.match(r"^\s*:\s*\d", after_str):
                        continue

                    prefix = sent[: match.start()].rstrip()
                    is_multi_or_range = bool(re.search(rf"[{DASH_CHARS},;]", inner))

                    # Tek bir sayıysa (örn: "(1)"), madde sıralaması veya Fig. (1) / n = (10) olup olmadığını denetle
                    if not is_multi_or_range:
                        if not prefix or prefix[-1] in ":=-+<>±/*%$":
                            continue
                        prev_word_m = re.search(rf"({ANY_LETTER}+)\.?\s*$", prefix)
                        prev_word = prev_word_m.group(1).lower() if prev_word_m else ""
                        if prev_word in non_citation_paren_labels:
                            continue

                    expanded_nums = self._expand_numeric_citation(inner)
                    if not expanded_nums or any(n <= 0 or n > upper_limit for n in expanded_nums):
                        continue

                    for num in expanded_nums:
                        matched_ref = num_map.get(num)
                        if matched_ref:
                            matched_ref.cited_in_text_count += 1

                        citations.append(
                            InTextCitation(
                                citation_id=f"CIT_{cit_counter}",
                                marker=marker,
                                citation_style="numeric",
                                claim_sentence=sent,
                                context_text=context_window,
                                matched_ref_id=matched_ref.ref_id if matched_ref else None,
                                matched_reference=matched_ref,
                                expected_key=f"[{num}]",
                            )
                        )
                        cit_counter += 1

            # B) APA Parantez içi atıfları yakala: (Yılmaz, 2021; Castrén and Monteggia, 2021)
            for match in APA_PAREN_REGEX.finditer(sent):
                inner = match.group(1)
                # Noktalı virgülle ayrılmış çoklu atıfları parçala
                sub_cits = [sc.strip() for sc in inner.split(";") if YEAR_REGEX.search(sc)]
                for sc in sub_cits:
                    # Eğer sadece "(2021)" gibi tek başına yıl ise C adımında yakalanır
                    if re.match(r"^\s*(?:19[5-9]\d|20[0-3]\d)[a-c]?(?:\s*,\s*(?:19[5-9]\d|20[0-3]\d|[a-c]))*\s*$", sc):
                        continue
                    # Protokol, etik kurul veya tarih numaralarını atla: örn. "2024/07-12", "2022/14"
                    if re.search(r"\b(?:19|20)\d{2}\s*[/\-–]\s*\d+", sc) or re.search(r"\d+\s*/\s*(?:19|20)\d{2}\b", sc):
                        continue
                    if re.search(
                        r"\b(şekil|tablo|figure|table|denklem|eq|university|üniversitesi|ethics|committee|approval|protocol|grant|no\.|number)\b",
                        sc,
                        re.IGNORECASE,
                    ):
                        continue

                    year_m_sc = YEAR_REGEX.search(sc)
                    before_year = sc[: year_m_sc.start()].strip() if year_m_sc else ""
                    if not re.search(rf"{UPPER_LETTER}{ANY_LETTER}+", before_year):
                        continue

                    # Aynı yazar için birden fazla yıl yazılmış olabilir: örn. "Reus et al., 2011, 2012" veya "Iosifescu, 2022a, 2022b"
                    all_years_in_sc = YEAR_REGEX.findall(sc)
                    all_4digit_years = {y[:4] for y in all_years_in_sc}
                    for y_val in all_years_in_sc:
                        single_query = f"{before_year}, {y_val}"
                        other_years = all_4digit_years - {y_val[:4]}
                        matched_refs = self._match_apa_to_references(
                            single_query,
                            references,
                            references_text,
                            claim_sentence=sent,
                            exclude_flex_years=other_years,
                        )
                        if matched_refs:
                            for mr in matched_refs:
                                mr.cited_in_text_count += 1
                                citations.append(
                                    InTextCitation(
                                        citation_id=f"CIT_{cit_counter}",
                                        marker=f"({sc})",
                                        citation_style="author_year",
                                        claim_sentence=sent,
                                        context_text=context_window,
                                        matched_ref_id=mr.ref_id,
                                        matched_reference=mr,
                                        expected_key=single_query,
                                    )
                                )
                                cit_counter += 1
                        else:
                            citations.append(
                                InTextCitation(
                                    citation_id=f"CIT_{cit_counter}",
                                    marker=f"({sc})",
                                    citation_style="author_year",
                                    claim_sentence=sent,
                                    context_text=context_window,
                                    matched_ref_id=None,
                                    matched_reference=None,
                                    expected_key=single_query,
                                )
                            )
                            cit_counter += 1

            # C) APA Cümle içi (Narrative) atıfları yakala: "Yılmaz ve ark. (2022)", "Castrén and Monteggia (2021)"
            for match in APA_INLINE_REGEX.finditer(sent):
                # Derginin kendi 1. sayfa künye kutusunu ("CITATION Buran Kavuran...") atla
                if "CITATION " in sent[: match.start()]:
                    continue

                author_part = match.group(1).strip()
                years_part = match.group(2).strip()
                marker = match.group(0)

                ignore_words = {
                    "şekil", "tablo", "bölüm", "denklem", "çizelge", "figure",
                    "table", "section", "equation", "model", "yöntem", "ancak",
                    "örneğin", "ayrıca", "sonuç", "bulgular", "veriler", "see", "in",
                    "citation", "received", "accepted", "published", "copyright"
                }
                first_word = author_part.split()[0].lower()
                if first_word in ignore_words:
                    continue

                all_inline_years = YEAR_REGEX.findall(years_part)
                all_inline_4digit = {y[:4] for y in all_inline_years}
                for y_val in all_inline_years:
                    query_str = f"{author_part}, {y_val}"
                    other_years = all_inline_4digit - {y_val[:4]}
                    matched_refs = self._match_apa_to_references(
                        query_str,
                        references,
                        references_text,
                        claim_sentence=sent,
                        exclude_flex_years=other_years,
                    )
                    if matched_refs:
                        for mr in matched_refs:
                            mr.cited_in_text_count += 1
                            citations.append(
                                InTextCitation(
                                    citation_id=f"CIT_{cit_counter}",
                                    marker=marker,
                                    citation_style="author_year",
                                    claim_sentence=sent,
                                    context_text=context_window,
                                    matched_ref_id=mr.ref_id,
                                    matched_reference=mr,
                                    expected_key=query_str,
                                )
                            )
                            cit_counter += 1
                    else:
                        citations.append(
                            InTextCitation(
                                citation_id=f"CIT_{cit_counter}",
                                marker=marker,
                                citation_style="author_year",
                                claim_sentence=sent,
                                context_text=context_window,
                                matched_ref_id=None,
                                matched_reference=None,
                                expected_key=query_str,
                            )
                        )
                        cit_counter += 1

        return citations

    def _expand_numeric_citation(self, inner_str: str) -> List[int]:
        """'1, 3-5, 8' veya '1–4, 7' gibi ifadeleri [1, 3, 4, 5, 8] listesine genişletir."""
        nums: List[int] = []
        parts = re.split(r"[,;]", inner_str)
        for p in parts:
            p = p.strip()
            range_match = re.match(rf"^(\d{{1,3}})\s*[{DASH_CHARS}]\s*(\d{{1,3}})$", p)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start <= end and (end - start) <= 60:
                    nums.extend(range(start, end + 1))
            elif p.isdigit():
                nums.append(int(p))
        return nums

    def _match_apa_to_references(
        self,
        citation_str: str,
        references: List[ReferenceItem],
        references_text: str = "",
        claim_sentence: str = "",
        exclude_flex_years: Optional[set] = None,
    ) -> List[ReferenceItem]:
        """
        APA atıf ifadesini (Örn: 'Castrén and Monteggia, 2021', 'Iosifescu, 2022b')
        kaynakçadaki makale(ler) ile eşleştirir.
        - Sadece ilk yazarın soyadını (birleşik soyadları dahil) baz alır; ortak yazarlarla karışmaz.
        - Aynı yazar ve aynı yıla ait birden fazla kaynak varsa (örn. iki 'Stahl, 2013' veya iki 'Reus, 2012'):
          1) Tek yazar ('Stahl, 2013') ile çok yazar ('Stahl et al., 2013') ayrımı yapar.
          2) Yazar sayısı da aynıysa hem her iki kaynağı bağlar hem de cümle/başlık uyumuna göre sıralar.
        """
        year_match = YEAR_REGEX.search(citation_str)
        if not year_match:
            return []
        cit_year_full = year_match.group(1).lower()      # Örn: "2022b" veya "2021"
        cit_year_4 = cit_year_full[:4]                   # Örn: "2022"
        has_letter_suffix = len(cit_year_full) > 4       # 'a', 'b', 'c' var mı?

        raw_author_chunk = citation_str[: year_match.start()]
        cit_has_etal = bool(re.search(r"\b(ve\s+ark\.?|vd\.?|et\s+al\.?)\b", raw_author_chunk, re.IGNORECASE))

        author_chunk = re.sub(
            r"\b(ve\s+ark\.?|vd\.?|et\s+al\.?|ve|and|&)\b", " ", raw_author_chunk, flags=re.IGNORECASE
        )
        surnames = [
            w for w in re.findall(rf"{ANY_LETTER}{{2,}}(?:[\-']{ANY_LETTER}+)*", author_chunk)
        ]
        if not surnames:
            return []

        norm_primary = normalize_surname_token(surnames[0])
        norm_second = normalize_surname_token(surnames[1]) if len(surnames) > 1 else ""
        if not norm_primary:
            return []

        cit_is_single = (not cit_has_etal) and (not norm_second)

        def _is_ref_single_author(r: ReferenceItem) -> bool:
            auth_str = r.authors or ""
            has_conj = bool(re.search(r"\b(and|ve|&)\b|;", auth_str, re.IGNORECASE))
            return (not has_conj) and (auth_str.count(",") <= 1)

        exact_matches: List[ReferenceItem] = []

        # 1. Aşama: İlk yazar soyadı (aksan/tire normalize) + Yıl eşleşmesi
        for ref in references:
            ref_year_full = (ref.year or "").lower()
            ref_year_4 = ref_year_full[:4]
            norm_ref_first = normalize_surname_token(ref.first_author_surname)
            first_author_part_norm = normalize_surname_token((ref.authors or "").split(",")[0])

            surname_ok = (norm_ref_first == norm_primary) or (
                len(norm_primary) >= 4 and norm_primary == first_author_part_norm
            )
            if not surname_ok:
                continue

            # Eğer atıfta 2. yazar varsa (örn: "Castrén and Monteggia, 2021"), 2. yazar da kaynakta geçmeli
            if norm_second and len(norm_second) >= 3:
                if norm_second not in normalize_surname_token(ref.authors):
                    continue

            if has_letter_suffix:
                if ref_year_full == cit_year_full:
                    exact_matches.append(ref)
            else:
                if ref_year_4 == cit_year_4:
                    exact_matches.append(ref)

        if exact_matches:
            # Eğer aynı yazar ve aynı yıla ait birden fazla kaynak varsa (örn: Stahl, 2013 vs Stahl et al., 2013),
            # önce tek yazar / çok yazar (et al.) ayrımıyla doğru olanı filtrele:
            if len(exact_matches) > 1:
                if cit_has_etal:
                    multi_only = [r for r in exact_matches if not _is_ref_single_author(r)]
                    if multi_only:
                        exact_matches = multi_only
                elif cit_is_single:
                    single_only = [r for r in exact_matches if _is_ref_single_author(r)]
                    if single_only:
                        exact_matches = single_only

            # Hâlâ birden fazla eşleşme varsa ve cümle metni mevcutsa, başlık kelime örtüşmesine göre sırala
            if len(exact_matches) > 1 and claim_sentence:
                sent_words = set(re.findall(r"[a-z]{4,}", claim_sentence.lower()))
                exact_matches.sort(
                    key=lambda r: len(sent_words.intersection(re.findall(r"[a-z]{4,}", (r.title or "").lower()))),
                    reverse=True,
                )
            return exact_matches

        # 2. Aşama: Kaynakçada iki kaynak birbirine yapışmışsa (örn. Hayashi, 2015 veya Liu and Sun, 2018)
        # her bir ReferenceItem.raw_text içinde soyadı + yıl yan yana geçiyor mu kontrol et!
        for ref in references:
            raw_norm = normalize_surname_token(ref.raw_text)
            if norm_primary in raw_norm and cit_year_4 in ref.raw_text:
                if not norm_second or norm_second in raw_norm:
                    return [ref]

        # 3. Aşama: ±1 Yıl Esnek Eşleşme (Online basım 2010 / Dergi atıfı 2011 farkları için)
        if len(norm_primary) >= 2:
            try:
                cy = int(cit_year_4)
                flex_matches: List[ReferenceItem] = []
                for ref in references:
                    norm_ref_first = normalize_surname_token(ref.first_author_surname)
                    raw_norm = normalize_surname_token(ref.raw_text[:120])
                    if norm_ref_first == norm_primary and ref.year and ref.year[:4].isdigit():
                        if norm_second and norm_second not in raw_norm:
                            continue
                        if not norm_second and len(norm_primary) < 4:
                            continue
                        ry_str = ref.year[:4]
                        if exclude_flex_years and ry_str in exclude_flex_years:
                            continue
                        ry = int(ry_str)
                        if abs(cy - ry) <= 1:
                            flex_matches.append(ref)
                if flex_matches:
                    unmatched_flex = [r for r in flex_matches if r.cited_in_text_count == 0]
                    return unmatched_flex if unmatched_flex else flex_matches
            except ValueError:
                pass

        return []

    def _second_pass_orphan_recovery(
        self,
        body_text: str,
        references: List[ReferenceItem],
        citations: List[InTextCitation],
    ) -> None:
        """
        İkinci Güvenlik Taraması:
        İlk taramada `cited_in_text_count == 0` kalan (Yetim görünen) her bir kaynak için,
        ilk yazarın soyadı ve yılı ana metin (body_text) içinde birbirine yakın (120 karakter)
        geçiyor mu diye kontrol eder. Tablo içi, şekil altı veya karmaşık parantez içi
        atıfları böylece %100 yakalar!
        """
        joined_body = re.sub(r"\s+", " ", body_text)
        for ref in references:
            if ref.cited_in_text_count > 0:
                continue
            if not ref.first_author_surname or not ref.year:
                continue

            surname_esc = re.escape(ref.first_author_surname)
            year_4 = ref.year[:4]
            # Soyadı ... (en fazla 100 karakter) ... Yıl VEYA aksansız soyadı araması
            pattern = re.compile(
                rf"\b{surname_esc}\b.{{0,100}}?\b{year_4}[a-c]?\b",
                re.IGNORECASE,
            )
            m = pattern.search(joined_body)
            if not m:
                # Aksanlı harf farkı varsa (örn: Castrén vs Castren) normalize arama yap
                norm_s = normalize_surname_token(ref.first_author_surname)
                if len(norm_s) >= 4:
                    for y_m in re.finditer(rf"\b{year_4}[a-c]?\b", joined_body):
                        start_pos = max(0, y_m.start() - 100)
                        window = joined_body[start_pos : y_m.end() + 20]
                        if norm_s in normalize_surname_token(window):
                            m = y_m
                            break

            if m:
                ref.cited_in_text_count += 1
                start_s = max(0, m.start() - 120)
                end_s = min(len(joined_body), m.end() + 120)
                snippet = joined_body[start_s:end_s].strip()
                citations.append(
                    InTextCitation(
                        citation_id=f"CIT_{len(citations) + 1}",
                        marker=f"({ref.first_author_surname}, {ref.year})",
                        citation_style="author_year",
                        claim_sentence=snippet,
                        context_text=snippet,
                        matched_ref_id=ref.ref_id,
                        matched_reference=ref,
                        expected_key=f"{ref.first_author_surname}, {ref.year}",
                    )
                )


# ---------------------------------------------------------------------------
# Hızlı Modül Testi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sample_manuscript = """
    1. GİRİŞ
    Yapay zeka tabanlı görüntü işleme yöntemleri son yıllarda tıbbi teşhis süreçlerinde büyük bir devrim yaratmıştır [1].
    Özellikle derin evrişimli sinir ağları (CNN), akciğer grafilerinde %94 oranında doğruluk sağlamaktadır (Yılmaz ve ark., 2022).
    Öte yandan nörobiyoloji çalışmalarında (Castrén and Monteggia, 2021; Iosifescu, 2022a, 2022b) önemli bulgular elde edilmiştir.
    Literatürde Smith et al. (2021) tarafından önerilen dikkat mekanizması (attention) bu sorunu büyük ölçüde çözmüştür.
    Ayrıca Demir (2023) çalışmasında kuantum tabanlı yeni bir optimizasyon önerilmiştir.

    KAYNAKÇA
    [1] Kaya, M., & Öztürk, B. (2020). Tıbbi Görüntülemede Derin Öğrenme Uygulamaları. Türkiye Bilişim Dergisi, 12(3), 45-58. doi:10.1234/tbd.2020.001
    [2] Yılmaz, A., Demir, C., & Şahin, K. (2022). Deep Convolutional Networks for Chest X-Ray Classification. IEEE Access, 10, 11200-11215. https://doi.org/10.1109/ACCESS.2022.123456
    [3] Smith, J., & Doe, A. (2021). Attention Is All You Need in Medical Vision. Nature Medicine, 27(4), 500-512.
    [4] Castrén, E., and Monteggia, L. M. (2021). Brain-Derived Neurotrophic Factor Signaling in Depression. Biol. Psychiatry, 90(2), 128-136.
    [5] Iosifescu, D. V. (2022a). Glutamatergic Modulation in MDD. J. Clin. Psychiatry, 83(1).
    [6] Iosifescu, D. V. (2022b). Effect of AXS-05 in Major Depressive Disorder. J. Clin. Psychiatry, 83(2).
    [7] Çelik, T. (2019). Klasik Makine Öğrenmesi Yöntemlerinin Karşılaştırılması. Mühendislik Bilimleri Dergisi, 5(1), 10-20.
    """

    parser = DocumentParser()
    result = parser.parse_text(sample_manuscript, filename="ornek_tubitak_makalesi.txt")

    print(f"=== MAKALE AYRIŞTIRMA TEST SONUCU ({result.filename}) ===")
    print(f"Toplam Kaynakça Sayısı   : {len(result.references)}")
    print(f"Yakalanan Metin İçi Atıf : {len(result.citations)}")
    print(f"Hayalet Atıf (Kaynakçada Yok): {len(result.ghost_citations)}")
    print(f"Yetim Kaynak (Metinde Yok)   : {len(result.orphan_references)}")
    print("\n--- Yakalanan Atıflar ---")
    for c in result.citations:
        durum = f"Eşleşti -> Kaynak [{c.matched_ref_id}] ({c.matched_reference.title[:40]}...)" if c.matched_reference else "🚨 HAYALET ATIF (Kaynakçada Yok!)"
        print(f"  * {c.marker} ({c.expected_key}): {durum}")
    print("\n--- Yetim Kaynaklar (Metinde Hiç Kullanılmayanlar) ---")
    for o in result.orphan_references:
        print(f"  * [{o.ref_id}] {o.raw_text[:80]}...")
