"""
MODÜL 2: 8 Kanallı Akademik İnternet Avcısı (academic_fetcher.py)
-----------------------------------------------------------------
Görevleri:
1. Kaynakçadaki bir makalenin dünyada gerçekten var olup olmadığını (Halüsinasyon /
   Sahte Makale Tespiti) ve geri çekilip çekilmediğini (Retracted Paper) denetler.
2. 8 farklı akademik kanaldan şelale (waterfall) mimarisiyle makalenin
   künyesini, özetini (Abstract + TL;DR) ve TAM METNİNİ (PDF / HTML) çeker:
   - Kanal 1: CrossRef API (Resmi DOI, yazar, yıl, dergi ve Retraction kontrolü)
   - Kanal 2: OpenAlex API (250M+ makale, inverted-index özet, açık erişim PDF ve Retraction)
   - Kanal 3: PubMed & PubMed Central (PMC) Entrez API (Tıp/Fen özet ve PMC tam metin)
   - Kanal 4: Semantic Scholar & arXiv API (TL;DR ana bulgu özeti ve açık PDF)
   - Kanal 5: Google Scholar & DergiPark / TR Dizin (Özellikle Türkçe makaleler ve [PDF] linkleri)
   - Kanal 6: Sci-Hub Otomatik Tam Metin Kilit Açıcı (Ücretli/kilitli makalelerin DOI ile PDF'ini indirir)
   - Kanal 7: Agent-Reach / Jina Reader (r.jina.ai ile web makalelerini engelsiz Markdown olarak okur)
   - Kanal 8: Alternatif Gerçek Makale Önerici (Sahte kaynak yerine iddiayı destekleyen 3 gerçek makale bulur)
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import sys
import time
import ipaddress
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import requests
from bs4 import BeautifulSoup
import pymupdf as fitz  # PyMuPDF

from document_parser import ReferenceItem


# ---------------------------------------------------------------------------
# Sabitler ve Önbellek (Cache) Ayarları
# ---------------------------------------------------------------------------

CACHE_DIR = Path(".cache")
PDF_CACHE_DIR = CACHE_DIR / "pdfs"
META_CACHE_DIR = CACHE_DIR / "metadata"

for d in (CACHE_DIR, PDF_CACHE_DIR, META_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}

API_HEADERS = {
    "User-Agent": "TubitakAcademicVerifier/1.0 (mailto:academic-verifier@tubitak-rag.local)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# URL ve Dosya Adı Güvenliği (ADIM 18)
# ---------------------------------------------------------------------------

BLOCKED_HOSTNAMES = {
    "localhost", "localhost.localdomain", "metadata", "metadata.google.internal",
    "instance-data", "169.254.169.254",
}


def is_safe_public_url(url: str) -> bool:
    """Yalnızca http/https ve genel (özel olmayan) adreslere izin verir.

    Kaynakçadan veya dış API yanıtından gelen URL'ler doğrulanmadan
    indirilmemelidir (SSRF koruması).
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host or host in BLOCKED_HOSTNAMES or host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False

    for info in infos or []:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        ):
            return False
    return bool(infos)


def safe_ref_token(value: Any, fallback: str = "ref") -> str:
    """ref_id'yi dosya adında güvenle kullanılabilecek bir belirtece çevirir."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or ""))
    token = re.sub(r"\.{2,}", ".", token)          # ".." dizilerini etkisizleştir
    token = token.strip("._-")[:40].strip("._-")
    return token or fallback

SCIHUB_MIRRORS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
    "https://sci-hub.wf",
    "https://sci-hub.mksa.top",
]


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class FetchedPaper:
    """İnternetten doğrulanan ve içeriği çekilen referans makale."""
    ref_id: str
    query_title: str
    query_doi: Optional[str]
    exists_online: bool = False                 # Makale internette gerçek mi?
    title_match_score: float = 0.0              # Başlık benzerlik oranı (0.0 - 1.0)
    verified_title: str = ""
    verified_authors: List[str] = field(default_factory=list)
    verified_year: Optional[str] = None
    verified_venue: str = ""
    verified_doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    is_retracted: bool = False                  # Geri çekilmiş (sahte/hatalı) makale mi?
    retraction_details: str = ""
    citation_count: int = 0
    abstract: str = ""                          # Makale özeti
    tldr: str = ""                              # Semantic Scholar TL;DR özeti
    full_text: str = ""                         # PDF veya Web'den çekilen tam metin
    content_source: str = "Bulunamadı"          # Örn: "Sci-Hub PDF + CrossRef + OpenAlex"
    sources_checked: List[str] = field(default_factory=list)
    sources_found: List[str] = field(default_factory=list)
    pdf_url: Optional[str] = None
    local_pdf_path: Optional[str] = None
    metadata_warnings: List[str] = field(default_factory=list)
    lookup_status: str = "not_found"            # "found" | "not_found" | "error"
    lookup_error: str = ""                      # Ağ/API hatası varsa kısa açıklama

    @property
    def combined_evidence_text(self) -> str:
        """RAG motorunun tarayacağı birleştirilmiş tüm makale metni (Özet + TL;DR + Tam Metin)."""
        parts = []
        if self.verified_title:
            parts.append(f"MAKALE BAŞLIĞI: {self.verified_title}")
        if self.tldr:
            parts.append(f"ANA BULGU (TL;DR): {self.tldr}")
        if self.abstract:
            parts.append(f"MAKALE ÖZETİ (ABSTRACT):\n{self.abstract}")
        if self.full_text:
            parts.append(f"MAKALE TAM METNİ (FULL TEXT):\n{self.full_text}")
        return "\n\n".join(parts).strip()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["has_full_text"] = bool(self.full_text and len(self.full_text) > 400)
        d["has_abstract"] = bool(self.abstract and len(self.abstract) > 50)
        d["full_text_length"] = len(self.full_text)
        return d


@dataclass
class AlternativePaper:
    """Sahte veya hatalı kaynak yerine önerilen gerçek akademik makale."""
    title: str
    authors: str
    year: str
    venue: str
    doi: str
    citation_count: int
    abstract_snippet: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Ana İnternet Avcısı Sınıfı (AcademicFetcher)
# ---------------------------------------------------------------------------

class AcademicFetcher:
    """8 kanallı akademik doğrulama, özet ve tam metin (PDF/Web) çekme motoru."""

    def __init__(self, timeout: int = 12, enable_scihub: bool = False, enable_jina: bool = True):
        self.timeout = timeout
        self.enable_scihub = enable_scihub
        self.enable_jina = enable_jina
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    @staticmethod
    def _record_lookup_error(paper: FetchedPaper, channel: str, exc: Exception) -> None:
        """Kanal hatasını kaydeder. Yalnızca ilk hata tutulur (kısa ve okunur kalsın)."""
        if not paper.lookup_error:
            paper.lookup_error = f"{channel}: {type(exc).__name__}: {exc}"[:300]

    # -----------------------------------------------------------------------
    # Ana Giriş Noktası: Tek Bir Kaynağı 8 Kanaldan Doğrula ve İçeriğini Çek
    # -----------------------------------------------------------------------

    def verify_and_fetch(
        self, ref: ReferenceItem, local_pdfs_dir: Optional[Path] = None
    ) -> FetchedPaper:
        """
        Verilen ReferenceItem için:
        1. Önce önbelleğe (cache) bakar.
        2. CrossRef, OpenAlex, PubMed, Semantic Scholar, arXiv, DergiPark ve Google Scholar'ı tarar.
        3. Açık erişim PDF veya Sci-Hub üzerinden tam metin PDF indirir; Jina Reader ile web metnini okur.
        """
        cache_key = self._make_cache_key(ref.raw_text)
        cache_file = META_CACHE_DIR / f"{cache_key}.json"

        # Önbellekte doğrulanmış sonuç varsa diskten anında (0.01 sn) yükle
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached_data.get("exists_online") and (cached_data.get("abstract") or cached_data.get("full_text")):
                    cached_data.pop("has_full_text", None)
                    cached_data.pop("has_abstract", None)
                    cached_data.pop("full_text_length", None)
                    cached_data["ref_id"] = ref.ref_id
                    cached_data["lookup_status"] = "found" if cached_data.get("exists_online") else "not_found"
                    return FetchedPaper(**cached_data)
            except Exception:
                pass

        search_title = ref.title if len(ref.title) >= 8 else ref.raw_text
        paper = FetchedPaper(
            ref_id=ref.ref_id,
            query_title=search_title,
            query_doi=ref.doi,
            verified_doi=ref.doi,
        )

        # 0. Kullanıcının yüklediği yerel PDF klasörü varsa önce orada eşleşen dosya var mı bak
        if local_pdfs_dir and local_pdfs_dir.exists():
            self._check_user_local_pdfs(ref, paper, local_pdfs_dir)

        # 1. Kanal: CrossRef API (DOI veya Başlık ile Resmi Künye & Retraction Kontrolü)
        self._query_crossref(ref, paper)

        # 2. Kanal: OpenAlex API (Künye, Inverted-Index Özet, Retraction ve Açık Erişim PDF)
        self._query_openalex(ref, paper)

        # 3. Kanal: PubMed & PubMed Central (PMC) (Tam metin veya özet eksikse tıp/fen makaleleri için)
        if not paper.full_text or len(paper.abstract) < 800:
            self._query_pubmed_and_pmc(ref, paper)

        # 4. Kanal: Semantic Scholar & arXiv API (Eğer hala zengin özet/TL;DR veya doğrulama bulunamadıysa)
        if not paper.exists_online or len(paper.abstract) < 800 or not paper.tldr:
            self._query_semantic_scholar(ref, paper)
            if not paper.exists_online or len(paper.abstract) < 200:
                self._query_arxiv(ref, paper)

        # 5. Kanal: DergiPark / TR Dizin & Google Scholar (Türkçe Makaleler ve Ek PDF Keşfi)
        if not paper.exists_online or not (paper.abstract or paper.full_text):
            self._query_dergipark(ref, paper)
            self._query_google_scholar(ref, paper)

        # 6. Tam Metin İndirme: Açık Erişim PDF -> Sci-Hub Mirrorları -> Agent-Reach (Jina Reader)
        if not paper.full_text:
            # 6a. Açık erişim PDF URL bulunduysa indir ve oku
            if paper.pdf_url:
                pdf_text = self._download_and_extract_pdf(paper.pdf_url, paper, source_label="Açık Erişim PDF")
                if pdf_text:
                    paper.full_text = pdf_text

            # 6b. Hala tam metin yoksa ve Sci-Hub aktifse: DOI veya başlık ile Sci-Hub'dan indir
            if not paper.full_text and self.enable_scihub and (paper.verified_doi or paper.exists_online):
                scihub_text = self._fetch_from_scihub(paper)
                if scihub_text:
                    paper.full_text = scihub_text

            # 6c. Hala tam metin yoksa Agent-Reach / Jina Reader (r.jina.ai) ile makale web sayfasını oku
            if not paper.full_text and self.enable_jina:
                target_url = ref.url or (f"https://doi.org/{paper.verified_doi}" if paper.verified_doi else None)
                if target_url:
                    jina_text = self._fetch_via_jina_reader(target_url, paper)
                    if jina_text and len(jina_text) > 500:
                        paper.full_text = jina_text

        # 7. Künye Uyuşmazlığı Kontrolleri (Yıl / Yazar Farkı Uyarısı)
        self._validate_metadata_consistency(ref, paper)

        # İçerik kaynağını özetle
        content_tags = list(dict.fromkeys(paper.sources_found))
        if paper.full_text:
            content_tags.append("Tam Metin (Full-Text)")
        elif paper.abstract:
            content_tags.append("Özet (Abstract)")
        paper.content_source = " + ".join(content_tags) if content_tags else "Bulunamadı"

        # Arama durumu: bulundu / bulunamadı / ağ hatası (DOGRULANAMADI kararı için)
        if paper.exists_online:
            paper.lookup_status = "found"
        elif paper.lookup_error:
            paper.lookup_status = "error"
        else:
            paper.lookup_status = "not_found"

        # Sonucu önbelleğe yaz
        try:
            cache_file.write_text(json.dumps(paper.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        return paper

    # -----------------------------------------------------------------------
    # KANAL 1: CrossRef API (Resmi DOI Kaydı & Retraction Denetimi)
    # -----------------------------------------------------------------------

    def _query_crossref(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        paper.sources_checked.append("CrossRef")
        try:
            item = None
            doi_to_check = paper.verified_doi or ref.doi
            if doi_to_check:
                url = f"https://api.crossref.org/works/{urllib.parse.quote(doi_to_check)}"
                resp = requests.get(url, headers=API_HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    cand_item = resp.json().get("message", {})
                    cand_titles = cand_item.get("title", [])
                    cand_title = re.sub(r"<[^>]+>", "", cand_titles[0]).strip() if cand_titles else ""
                    if len(ref.title) >= 18 and cand_title:
                        doi_sim = self._title_similarity(ref.title, cand_title, ref.raw_text)
                        if doi_sim >= 0.45:
                            item = cand_item
                            paper.title_match_score = max(0.95, doi_sim)
                        else:
                            # DOI numarası kaynakçadaki başlıkla uyuşmuyor (başka makalenin DOI'si)!
                            paper.verified_doi = None
                            paper.metadata_warnings.append(
                                f"⚠️ Başlık–DOI Uyuşmazlığı: Kaynakçada yazılan DOI ({doi_to_check}) "
                                f"'{cand_title[:95]}' başlıklı farklı bir makaleye aittir!"
                            )
                    else:
                        item = cand_item
                        paper.title_match_score = 1.0

            if not item:
                query_str = ref.raw_text if len(ref.raw_text) >= 15 else ref.title
                url = "https://api.crossref.org/works"
                params = {"query.bibliographic": query_str[:300], "rows": 6}
                resp = requests.get(url, params=params, headers=API_HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    items = resp.json().get("message", {}).get("items", [])
                    best_item, best_score = self._find_best_crossref_match(ref, items)
                    if best_item and best_score >= 0.68:
                        item = best_item
                        paper.title_match_score = max(paper.title_match_score, best_score)

            if not item:
                return

            paper.exists_online = True
            paper.sources_found.append("CrossRef")

            titles = item.get("title", [])
            if titles and not paper.verified_title:
                paper.verified_title = re.sub(r"<[^>]+>", "", titles[0]).strip()

            if item.get("DOI") and not paper.verified_doi:
                paper.verified_doi = item["DOI"]

            # Yazarlar
            authors_list = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                full_a = f"{given} {family}".strip()
                if full_a:
                    authors_list.append(full_a)
            if authors_list and not paper.verified_authors:
                paper.verified_authors = authors_list

            # Yıl
            for date_Field in ("published-print", "published-online", "issued", "created"):
                date_parts = item.get(date_Field, {}).get("date-parts", [])
                if date_parts and date_parts[0]:
                    paper.verified_year = str(date_parts[0][0])
                    break

            # Dergi / Venue
            containers = item.get("container-title", [])
            if containers and not paper.verified_venue:
                paper.verified_venue = containers[0]

            # Atıf sayısı
            paper.citation_count = max(paper.citation_count, int(item.get("is-referenced-by-count", 0)))

            # Özet (CrossRef JATS XML etiketi içerebilir, temizle)
            abstract_raw = item.get("abstract", "")
            if abstract_raw and len(abstract_raw) > len(paper.abstract):
                clean_abs = re.sub(r"<[^>]+>", " ", abstract_raw)
                clean_abs = re.sub(r"\s+", " ", clean_abs).strip()
                paper.abstract = clean_abs

            # Retraction (Geri çekilmiş makale) kontrolü
            update_to = item.get("update-to", [])
            for upd in update_to:
                upd_type = str(upd.get("type", "")).lower()
                if "retract" in upd_type:
                    paper.is_retracted = True
                    paper.retraction_details = f"CrossRef kaydına göre bu makale geri çekilmiştir ({upd.get('label', 'Retraction')})."
            if paper.verified_title and re.search(r"^retracted\s*:", paper.verified_title, re.IGNORECASE):
                paper.is_retracted = True
                paper.retraction_details = "Makale başlığında 'RETRACTED' (Geri Çekilmiş) ibaresi yer alıyor."

            # PDF Linki varsa al
            for link in item.get("link", []):
                if "pdf" in str(link.get("content-type", "")).lower() and link.get("URL"):
                    paper.pdf_url = paper.pdf_url or link["URL"]
        except Exception as exc:
            self._record_lookup_error(paper, "CrossRef", exc)

    def _find_best_crossref_match(
        self, ref: ReferenceItem, items: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        best_item = None
        best_score = 0.0
        target_title = (ref.title if len(ref.title) >= 8 else ref.raw_text).lower()
        ref_surname = self._strip_accents((ref.first_author_surname or "").lower())

        for it in items:
            cand_titles = it.get("title", [])
            if not cand_titles:
                continue
            cand_title = re.sub(r"<[^>]+>", "", cand_titles[0]).lower()
            score = self._title_similarity(target_title, cand_title, ref.raw_text.lower())

            # Yazar soyadı eşleşmesi kontrolü (Aksan duyarsız: Réus == Reus, Castrén == Castren)
            cand_authors = self._strip_accents(
                " ".join((a.get("family") or "") for a in it.get("author", [])).lower()
            )
            if ref_surname and cand_authors:
                if ref_surname in cand_authors:
                    score = min(1.0, score + 0.08)
                else:
                    score *= 0.85

            if score > best_score:
                best_score = score
                best_item = it
        return best_item, best_score

    # -----------------------------------------------------------------------
    # KANAL 2: OpenAlex API (Özet, Açık Erişim PDF & Retraction)
    # -----------------------------------------------------------------------

    def _query_openalex(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        paper.sources_checked.append("OpenAlex")
        try:
            work = None
            if paper.verified_doi:
                url = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(paper.verified_doi)}"
                resp = requests.get(url, headers=API_HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    cand_work = resp.json()
                    cand_title = (cand_work.get("title") or "").strip()
                    if len(ref.title) >= 18 and cand_title:
                        doi_sim = self._title_similarity(ref.title, cand_title, ref.raw_text)
                        if doi_sim >= 0.45:
                            work = cand_work
                            paper.title_match_score = max(0.95, doi_sim)
                        else:
                            paper.verified_doi = None
                    else:
                        work = cand_work
                        paper.title_match_score = 1.0

            if not work:
                query_str = ref.title if len(ref.title) >= 8 else ref.raw_text
                clean_q = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", " ", query_str)
                clean_q = re.sub(r"\s+", " ", clean_q).strip()[:200]
                url = "https://api.openalex.org/works"
                params = {"search": clean_q, "per-page": 6}
                resp = requests.get(url, params=params, headers=API_HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    ref_surname = self._strip_accents((ref.first_author_surname or "").lower())
                    best_oa_score = 0.0
                    for cand in results:
                        cand_title = (cand.get("title") or "").lower()
                        score = self._title_similarity(query_str.lower(), cand_title, ref.raw_text.lower())
                        cand_auths = self._strip_accents(
                            " ".join(
                                (a.get("author", {}).get("display_name") or "")
                                for a in cand.get("authorships", [])
                            ).lower()
                        )
                        if ref_surname and cand_auths:
                            if ref_surname in cand_auths:
                                score = min(1.0, score + 0.08)
                            else:
                                score *= 0.85
                        if score >= 0.68 and score > best_oa_score:
                            best_oa_score = score
                            work = cand

                    if work and best_oa_score > paper.title_match_score:
                        paper.title_match_score = best_oa_score
                        # Daha yüksek eşleşmeli makale bulunduysa önceki zayıf künyeyi güncelle
                        paper.verified_title = work.get("title") or paper.verified_title
                        if work.get("publication_year"):
                            paper.verified_year = str(work["publication_year"])
                        oa_auths = [
                            a.get("author", {}).get("display_name")
                            for a in work.get("authorships", [])
                            if a.get("author", {}).get("display_name")
                        ]
                        if oa_auths:
                            paper.verified_authors = oa_auths

            if not work:
                return

            paper.exists_online = True
            paper.sources_found.append("OpenAlex")

            if work.get("title") and not paper.verified_title:
                paper.verified_title = work["title"]

            if work.get("doi") and not paper.verified_doi:
                paper.verified_doi = work["doi"].replace("https://doi.org/", "").strip()

            if work.get("publication_year") and not paper.verified_year:
                paper.verified_year = str(work["publication_year"])

            # Yazarlar
            if not paper.verified_authors:
                auths = []
                for a in work.get("authorships", []):
                    name = a.get("author", {}).get("display_name")
                    if name:
                        auths.append(name)
                paper.verified_authors = auths

            # Atıf sayısı
            paper.citation_count = max(paper.citation_count, int(work.get("cited_by_count", 0)))

            # Retraction (Geri çekilme) kontrolü
            if work.get("is_retracted"):
                paper.is_retracted = True
                paper.retraction_details = "OpenAlex veritabanına göre bu makale GERİ ÇEKİLMİŞTİR (Retracted Paper)!"

            # Inverted Index formatındaki özeti düz metne çevir
            inv_abs = work.get("abstract_inverted_index")
            if inv_abs and len(paper.abstract) < 600:
                reconstructed = self._reconstruct_openalex_abstract(inv_abs)
                if len(reconstructed) > len(paper.abstract):
                    paper.abstract = reconstructed

            # Açık erişim PDF linki
            oa_info = work.get("open_access", {})
            best_oa = work.get("best_oa_location") or {}
            pdf_candidate = best_oa.get("pdf_url") or oa_info.get("oa_url")
            if pdf_candidate and not paper.pdf_url:
                paper.pdf_url = pdf_candidate
        except Exception as exc:
            self._record_lookup_error(paper, "OpenAlex", exc)

    @staticmethod
    def _reconstruct_openalex_abstract(inverted_index: Dict[str, List[int]]) -> str:
        """OpenAlex'in kelime->pozisyon sözlüğünü tam paragrafa dönüştürür."""
        word_positions: List[Tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    # -----------------------------------------------------------------------
    # KANAL 3: PubMed & PubMed Central (PMC) API (Özet ve Tam Metin)
    # -----------------------------------------------------------------------

    def _query_pubmed_and_pmc(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        paper.sources_checked.append("PubMed/PMC")
        try:
            term = paper.verified_doi if paper.verified_doi else (paper.verified_title or ref.title)
            if not term or len(term) < 6:
                return

            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": 3}
            resp = requests.get(esearch_url, params=params, headers=API_HEADERS, timeout=self.timeout)
            if resp.status_code != 200:
                return

            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                # İngiliz/Amerikan yazım farkını normalize ederek (behavioural -> behavioral) tekrar dene
                norm_t = self._normalize_academic_text(paper.verified_title or ref.title)
                if norm_t and len(norm_t) >= 10:
                    params["term"] = norm_t
                    resp = requests.get(esearch_url, params=params, headers=API_HEADERS, timeout=self.timeout)
                    if resp.status_code == 200:
                        id_list = resp.json().get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return

            pmid = id_list[0]
            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_resp = requests.get(
                efetch_url,
                params={"db": "pubmed", "id": pmid, "retmode": "xml"},
                headers=API_HEADERS,
                timeout=self.timeout,
            )
            if fetch_resp.status_code != 200:
                return

            root = ET.fromstring(fetch_resp.text)
            article_title_el = root.find(".//ArticleTitle")
            if article_title_el is not None and article_title_el.text:
                cand_title = "".join(article_title_el.itertext()).strip()
                target_t = (paper.verified_title or ref.title or ref.raw_text).lower()
                sim = self._title_similarity(target_t, cand_title.lower(), ref.raw_text.lower())
                if not paper.verified_doi and sim < 0.65:
                    return
                if not paper.verified_title:
                    paper.verified_title = cand_title
                paper.title_match_score = max(paper.title_match_score, sim)

            paper.pmid = pmid
            paper.exists_online = True
            paper.sources_found.append("PubMed")

            # PubMed Özeti (Bölümlü özetleri birleştir)
            abs_texts = []
            for abs_el in root.findall(".//AbstractText"):
                label = abs_el.attrib.get("Label", "")
                txt = "".join(abs_el.itertext()).strip()
                if txt:
                    abs_texts.append(f"{label}: {txt}" if label else txt)
            if abs_texts:
                full_abs = "\n".join(abs_texts)
                if len(full_abs) > len(paper.abstract):
                    paper.abstract = full_abs

            # Retraction kontrolü (PubMed PublicationType)
            for pt in root.findall(".//PublicationType"):
                if pt.text and "retracted" in pt.text.lower():
                    paper.is_retracted = True
                    paper.retraction_details = "PubMed kaydına göre bu yayın geri çekilmiştir (Retracted Publication)."

            # PMC ID varsa PubMed Central'dan tam metin çek!
            for art_id in root.findall(".//ArticleId"):
                id_type = art_id.attrib.get("IdType", "")
                if id_type == "pmc" and art_id.text:
                    paper.pmcid = art_id.text.strip()
                elif id_type == "doi" and art_id.text and not paper.verified_doi:
                    paper.verified_doi = art_id.text.strip()

            if paper.pmcid:
                self._fetch_pmc_fulltext(paper)
        except Exception as exc:
            self._record_lookup_error(paper, "PubMed", exc)

    def _fetch_pmc_fulltext(self, paper: FetchedPaper) -> None:
        """PubMed Central (PMC) açık erişim makalesinin tam metnini XML üzerinden çeker."""
        try:
            pmc_num = paper.pmcid.replace("PMC", "").strip()
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            resp = requests.get(
                url,
                params={"db": "pmc", "id": pmc_num, "retmode": "xml"},
                headers=API_HEADERS,
                timeout=self.timeout,
            )
            if resp.status_code == 200 and "<body" in resp.text:
                root = ET.fromstring(resp.text)
                body_el = root.find(".//body")
                if body_el is not None:
                    paragraphs = [
                        " ".join(p.itertext()).strip()
                        for p in body_el.findall(".//p")
                        if p is not None
                    ]
                    text = "\n\n".join(p for p in paragraphs if len(p) > 30)
                    if len(text) > 500:
                        paper.full_text = text
                        paper.sources_found.append("PMC Tam Metin")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # KANAL 4: Semantic Scholar & arXiv API (TL;DR, Özet ve PDF)
    # -----------------------------------------------------------------------

    def _query_semantic_scholar(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        paper.sources_checked.append("Semantic Scholar")
        try:
            fields = "title,authors,year,abstract,tldr,openAccessPdf,externalIds,citationCount,venue"
            s2_data = None

            if paper.verified_doi:
                url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(paper.verified_doi)}"
                resp = requests.get(url, params={"fields": fields}, headers=API_HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    s2_data = resp.json()

            if not s2_data:
                query_str = paper.verified_title or (ref.title if len(ref.title) >= 10 else ref.raw_text)
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                resp = requests.get(
                    url,
                    params={"query": query_str[:200], "limit": 2, "fields": fields},
                    headers=API_HEADERS,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    for cand in resp.json().get("data", []):
                        cand_t = (cand.get("title") or "").lower()
                        sim = self._title_similarity(query_str.lower(), cand_t, ref.raw_text.lower())
                        if sim >= 0.68:
                            s2_data = cand
                            paper.title_match_score = max(paper.title_match_score, sim)
                            break

            if not s2_data:
                return

            paper.exists_online = True
            paper.sources_found.append("Semantic Scholar")

            if s2_data.get("title") and not paper.verified_title:
                paper.verified_title = s2_data["title"]
            if s2_data.get("year") and not paper.verified_year:
                paper.verified_year = str(s2_data["year"])
            if s2_data.get("venue") and not paper.verified_venue:
                paper.verified_venue = s2_data["venue"]
            if s2_data.get("abstract") and len(s2_data["abstract"]) > len(paper.abstract):
                paper.abstract = s2_data["abstract"]

            tldr_obj = s2_data.get("tldr")
            if isinstance(tldr_obj, dict) and tldr_obj.get("text"):
                paper.tldr = tldr_obj["text"]

            oa_pdf = s2_data.get("openAccessPdf")
            if isinstance(oa_pdf, dict) and oa_pdf.get("url") and not paper.pdf_url:
                paper.pdf_url = oa_pdf["url"]

            ext_ids = s2_data.get("externalIds") or {}
            if ext_ids.get("DOI") and not paper.verified_doi:
                paper.verified_doi = ext_ids["DOI"]

            paper.citation_count = max(paper.citation_count, int(s2_data.get("citationCount") or 0))
        except Exception as exc:
            self._record_lookup_error(paper, "SemanticScholar", exc)

    def _query_arxiv(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        paper.sources_checked.append("arXiv")
        try:
            # Kaynak metninde doğrudan arXiv ID var mı? (Örn: arXiv:2104.12345)
            arxiv_match = re.search(r"arxiv\s*[:/]\s*(\d{4}\.\d{4,5})", ref.raw_text, re.IGNORECASE)
            query_title = paper.verified_title or ref.title
            if not arxiv_match and (not query_title or len(query_title) < 12):
                return

            url = "https://export.arxiv.org/api/query"
            if arxiv_match:
                params = {"id_list": arxiv_match.group(1), "max_results": 1}
            else:
                clean_t = re.sub(r"[^\w\s]", " ", query_title).strip()
                params = {"search_query": f'ti:"{clean_t[:120]}"', "max_results": 2}

            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return

            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                id_el = entry.find("atom:id", ns)
                if title_el is None or not title_el.text:
                    continue

                cand_title = re.sub(r"\s+", " ", title_el.text).strip()
                sim = self._title_similarity((query_title or "").lower(), cand_title.lower(), ref.raw_text.lower())
                if arxiv_match or sim >= 0.72:
                    paper.exists_online = True
                    paper.sources_found.append("arXiv")
                    paper.title_match_score = max(paper.title_match_score, sim)
                    if not paper.verified_title:
                        paper.verified_title = cand_title
                    if summary_el is not None and summary_el.text:
                        abs_txt = re.sub(r"\s+", " ", summary_el.text).strip()
                        if len(abs_txt) > len(paper.abstract):
                            paper.abstract = abs_txt
                    if id_el is not None and id_el.text and not paper.pdf_url:
                        paper.pdf_url = id_el.text.replace("/abs/", "/pdf/") + ".pdf"
                    break
        except Exception as exc:
            self._record_lookup_error(paper, "arXiv", exc)

    # -----------------------------------------------------------------------
    # KANAL 5: DergiPark / TR Dizin & Google Scholar (Türkçe ve Genel Tarama)
    # -----------------------------------------------------------------------

    def _query_dergipark(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        """DergiPark üzerinde Türkçe/yerel akademik makale araması ve PDF tespiti yapar."""
        paper.sources_checked.append("DergiPark")
        try:
            query_str = paper.verified_title or (ref.title if len(ref.title) >= 8 else ref.raw_text[:120])
            url = "https://dergipark.org.tr/tr/search"
            resp = self.session.get(url, params={"q": query_str, "section": "articles"}, timeout=self.timeout)
            if resp.status_code != 200:
                return

            soup = BeautifulSoup(resp.text, "lxml")
            cards = soup.select(".card.article-card, .search-results .card")
            for card in cards[:3]:
                title_el = card.select_one("h5.card-title a, a.card-title")
                if not title_el:
                    continue
                cand_title = re.sub(r"\s+", " ", title_el.get_text()).strip()
                sim = self._title_similarity(query_str.lower(), cand_title.lower(), ref.raw_text.lower())
                if sim >= 0.65:
                    paper.exists_online = True
                    paper.sources_found.append("DergiPark")
                    paper.title_match_score = max(paper.title_match_score, sim)
                    if not paper.verified_title:
                        paper.verified_title = cand_title

                    article_href = title_el.get("href", "")
                    if article_href and article_href.startswith("http"):
                        # Makale detay sayfasına gidip özet ve PDF linkini al
                        art_resp = self.session.get(article_href, timeout=self.timeout)
                        if art_resp.status_code == 200:
                            art_soup = BeautifulSoup(art_resp.text, "lxml")
                            abs_div = art_soup.select_one(".article-abstract, #article-main-portlet")
                            if abs_div and len(paper.abstract) < 100:
                                paper.abstract = re.sub(r"\s+", " ", abs_div.get_text()).strip()[:2500]
                            pdf_a = art_soup.select_one("a[href*='/download/article-file/']")
                            if pdf_a and not paper.pdf_url:
                                pdf_href = pdf_a.get("href", "")
                                paper.pdf_url = (
                                    pdf_href
                                    if pdf_href.startswith("http")
                                    else f"https://dergipark.org.tr{pdf_href}"
                                )
                    break
        except Exception as exc:
            self._record_lookup_error(paper, "DergiPark", exc)

    def _query_google_scholar(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        """Google Scholar üzerinden makale gerçekliğini, özet snippet'ini ve sağdaki [PDF] linkini tarar."""
        paper.sources_checked.append("Google Scholar")
        try:
            query_str = paper.verified_title or (ref.title if len(ref.title) >= 8 else ref.raw_text[:140])
            url = "https://scholar.google.com/scholar"
            resp = self.session.get(url, params={"q": query_str, "hl": "tr"}, timeout=self.timeout)
            if resp.status_code != 200:
                return

            soup = BeautifulSoup(resp.text, "lxml")
            for block in soup.select(".gs_r.gs_or.gs_scl")[:3]:
                title_el = block.select_one("h3.gs_rt")
                if not title_el:
                    continue
                cand_title = re.sub(r"^\[[^\]]+\]\s*", "", title_el.get_text()).strip()
                sim = self._title_similarity(query_str.lower(), cand_title.lower(), ref.raw_text.lower())
                if sim >= 0.65:
                    paper.exists_online = True
                    paper.sources_found.append("Google Scholar")
                    paper.title_match_score = max(paper.title_match_score, sim)
                    if not paper.verified_title:
                        paper.verified_title = cand_title

                    snippet_el = block.select_one(".gs_rs")
                    if snippet_el and not paper.abstract:
                        paper.abstract = re.sub(r"\s+", " ", snippet_el.get_text()).strip()

                    pdf_link_el = block.select_one(".gs_or_ggsm a")
                    if pdf_link_el and pdf_link_el.get("href") and not paper.pdf_url:
                        paper.pdf_url = pdf_link_el["href"]
                    break
        except Exception as exc:
            self._record_lookup_error(paper, "GoogleScholar", exc)

    # -----------------------------------------------------------------------
    # KANAL 6: Sci-Hub Otomatik Tam Metin PDF İndirici
    # -----------------------------------------------------------------------

    def _fetch_from_scihub(self, paper: FetchedPaper) -> str:
        """
        Aktif Sci-Hub mirrorlarını (sci-hub.se, sci-hub.st, sci-hub.ru vb.) sırayla dener;
        sayfadaki iframe/embed PDF linkini yakalayıp PDF'i indirir ve tam metnini çıkarır.
        """
        identifier = paper.verified_doi or paper.verified_title
        if not identifier:
            return ""

        paper.sources_checked.append("Sci-Hub")

        for mirror in SCIHUB_MIRRORS:
            try:
                target_url = f"{mirror}/{identifier.lstrip('/')}"
                resp = self.session.get(target_url, timeout=self.timeout)
                if resp.status_code != 200:
                    continue

                # Eğer doğrudan PDF döndüyse
                if resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
                    text = self._save_and_read_pdf_bytes(resp.content, paper, "Sci-Hub PDF")
                    if text:
                        return text

                soup = BeautifulSoup(resp.text, "lxml")
                pdf_src = None

                # 1. <embed id="pdf" src="..."> veya <iframe id="pdf" src="...">
                embed_el = soup.select_one("embed#pdf, iframe#pdf, embed[type='application/pdf'], iframe[src*='.pdf']")
                if embed_el and embed_el.get("src"):
                    pdf_src = embed_el["src"]

                # 2. <button onclick="location.href='...'">
                if not pdf_src:
                    btn = soup.find("button", attrs={"onclick": re.compile(r"location\.href")})
                    if btn:
                        m = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", btn["onclick"])
                        if m:
                            pdf_src = m.group(1)

                if not pdf_src:
                    continue

                # Göreli URL'leri tam URL'ye tamamla
                pdf_src = pdf_src.split("#")[0].strip()
                if pdf_src.startswith("//"):
                    pdf_src = "https:" + pdf_src
                elif pdf_src.startswith("/"):
                    pdf_src = mirror + pdf_src

                pdf_text = self._download_and_extract_pdf(pdf_src, paper, source_label="Sci-Hub PDF")
                if pdf_text:
                    return pdf_text
            except Exception:
                continue

        return ""

    # -----------------------------------------------------------------------
    # KANAL 7: Agent-Reach / Jina Reader (r.jina.ai) Web Makale Okuyucu
    # -----------------------------------------------------------------------

    def _fetch_via_jina_reader(self, target_url: str, paper: FetchedPaper) -> str:
        """
        Agent-Reach mimarisinde kullanılan Jina Reader (https://r.jina.ai/) üzerinden
        yayıncı web sayfasını bot engeline takılmadan temiz Markdown metni olarak çeker.
        """
        paper.sources_checked.append("Jina Reader (Agent-Reach)")
        if not is_safe_public_url(target_url):
            return ""
        try:
            jina_url = f"https://r.jina.ai/{target_url}"
            headers = {
                "User-Agent": DEFAULT_HEADERS["User-Agent"],
                "X-Return-Format": "markdown",
            }
            resp = requests.get(jina_url, headers=headers, timeout=self.timeout + 4)
            if resp.status_code == 200 and len(resp.text) > 600:
                text = resp.text.strip()
                # Çerez/giriş duvarı (paywall) sayfalarını filtrele
                if "just a moment..." in text.lower()[:400] or "access denied" in text.lower()[:400]:
                    return ""
                paper.sources_found.append("Jina Web Reader")
                return text[:45000]
        except Exception:
            pass
        return ""

    # -----------------------------------------------------------------------
    # KANAL 8: Alternatif Gerçek Makale Önerici & Akıllı Cümle Kaynak Bulucu
    # -----------------------------------------------------------------------

    # Sık geçen Türkçe akademik/tıbbi kavramların uluslararası (İngilizce) arama karşılıkları
    TR_TO_EN_ACADEMIC_MAP: Dict[str, str] = {
        "siklofosfamid": "cyclophosphamide",
        "oksidatif": "oxidative",
        "stres": "stress",
        "stresi": "stress",
        "sperm": "sperm",
        "hareketliliği": "motility",
        "hareketlilik": "motility",
        "motilite": "motility",
        "motilitesi": "motility",
        "morfoloji": "morphology",
        "morfolojik": "morphological",
        "testis": "testis testicular",
        "testiküler": "testicular",
        "hücre": "cell",
        "hücreleri": "cells",
        "hücresi": "cell",
        "apoptoz": "apoptosis",
        "apoptozis": "apoptosis",
        "inflamasyon": "inflammation",
        "inflamatuar": "inflammatory",
        "antioksidan": "antioxidant",
        "hasar": "damage injury",
        "hasarı": "damage injury",
        "toksisite": "toxicity",
        "toksik": "toxic",
        "kanser": "cancer",
        "akciğer": "lung",
        "karaciğer": "liver hepatic",
        "böbrek": "kidney renal",
        "beyin": "brain cerebral",
        "kalp": "heart cardiac",
        "diyabet": "diabetes",
        "hipertansiyon": "hypertension",
        "depresyon": "depression",
        "anksiyete": "anxiety",
        "nöron": "neuron neuronal",
        "nöroprotektif": "neuroprotective",
        "mitokondri": "mitochondria",
        "mitokondriyal": "mitochondrial",
        "peroksidasyon": "peroxidation",
        "peroksidasyonu": "peroxidation",
        "glutatyon": "glutathione",
        "katalaz": "catalase",
        "süperoksit": "superoxide",
        "dismutaz": "dismutase",
        "malondialdehit": "malondialdehyde MDA",
        "testosteron": "testosterone",
        "fertilite": "fertility",
        "infertilite": "infertility",
        "kemoterapi": "chemotherapy",
        "radyoterapi": "radiotherapy",
        "derin": "deep",
        "öğrenme": "learning",
        "yapay": "artificial",
        "zeka": "intelligence",
        "zekâ": "intelligence",
        "algoritma": "algorithm",
        "sınıflandırma": "classification",
        "tahmin": "prediction",
        "doğruluk": "accuracy",
        "tedavi": "treatment therapy",
        "tedavisi": "treatment therapy",
        "hastalık": "disease",
        "hastalığı": "disease",
        "gen": "gene",
        "ekspresyonu": "expression",
        "protein": "protein",
        "enzim": "enzyme",
        "reseptör": "receptor",
    }

    TR_STOPWORDS = {
        "olan", "olarak", "için", "göre", "daha", "çok", "gibi", "kadar", "sonra", "önce",
        "arasında", "üzerine", "tarafından", "birlikte", "ancak", "veya", "olup", "olduğu",
        "nedeniyle", "sebebiyle", "sonucunda", "sayesinde", "rağmen", "karşı", "ilgili",
        "önemli", "büyük", "yüksek", "düşük", "farklı", "çeşitli", "birçok", "bazı",
        "tüm", "bütün", "her", "hiç", "hem", "ise", "ile", "ve", "de", "da", "bu", "şu",
        "bir", "iki", "üç", "oldukça", "özellikle", "genellikle", "sadece", "yalnız",
        "yalnızca", "ayrıca", "böylece", "dolayısıyla", "örneğin", "mesela", "yüzde",
        "oranı", "oranında", "artış", "azalma", "azaltır", "artırır", "neden", "olur",
        "gösterilmiştir", "bildirilmiştir", "bulunmuştur", "saptanmıştır", "incelenmiştir",
    }

    def build_search_queries_for_claim(
        self, claim_sentence: str, ollama_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Kullanıcının yazdığı atıf/iddia cümlesinden:
        1) Orijinal dilde temiz anahtar kelime sorgusu (`native_query`)
        2) Uluslararası veritabanları (PubMed, OpenAlex, Semantic Scholar) için İngilizce
           akademik anahtar kelime sorgusu (`english_query`) üretir.
        """
        clean_claim = re.sub(r"\[[^\]]+\]|\([^)]*\d{4}[^)]*\)", " ", claim_sentence)
        clean_claim = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ\-]", " ", clean_claim)
        raw_words = [
            w for w in clean_claim.split()
            if len(w) > 2 and w.lower() not in self.TR_STOPWORDS and not w.isdigit()
        ]
        native_query = " ".join(raw_words[:12]).strip()

        # Sözlük tabanlı hızlı İngilizce terim çevirisi
        en_terms: List[str] = []
        for w in raw_words:
            wl = w.lower()
            if wl in self.TR_TO_EN_ACADEMIC_MAP:
                en_terms.append(self.TR_TO_EN_ACADEMIC_MAP[wl])
            else:
                # Kök eşleşmesi (örn: "antioksidanların" -> "antioksidan")
                matched_prefix = False
                for tr_k, en_v in self.TR_TO_EN_ACADEMIC_MAP.items():
                    if len(tr_k) >= 5 and wl.startswith(tr_k):
                        en_terms.append(en_v)
                        matched_prefix = True
                        break
                if not matched_prefix and re.match(r"^[a-zA-Z0-9\-]+$", w):
                    # Zaten Latin/İngilizce bilimsel terimse (örn: Sertoli, Leydig, COVID-19, BDNF, TRPM2) koru
                    en_terms.append(w)

        dict_en_query = " ".join(dict.fromkeys(" ".join(en_terms).split()))[:140].strip()

        # Eğer yerel Ollama açıksa çok hızlı (6 sn sınırıyla) İngilizce akademik arama terimlerini iste
        llm_en_query = ""
        if ollama_model:
            try:
                import ollama
                resp = ollama.chat(
                    model=ollama_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Translate the scientific claim into 5 to 8 concise English academic search keywords "
                                "for PubMed/OpenAlex search. Output ONLY the English keywords separated by spaces, no punctuation or explanation."
                            ),
                        },
                        {"role": "user", "content": claim_sentence[:350]},
                    ],
                    options={"temperature": 0.0, "num_predict": 30},
                )
                content = (
                    getattr(getattr(resp, "message", None), "content", None)
                    or resp.get("message", {}).get("content", "")
                )
                cleaned_llm = re.sub(r"[^a-zA-Z0-9\s\-]", " ", content or "")
                cleaned_llm = re.sub(r"\s+", " ", cleaned_llm).strip()
                if len(cleaned_llm) >= 8:
                    llm_en_query = " ".join(cleaned_llm.split()[:10])
            except Exception:
                pass

        english_query = llm_en_query or dict_en_query or native_query
        return native_query, english_query

    def search_candidate_papers_for_claim(
        self,
        claim_sentence: str,
        max_candidates: int = 10,
        year_from: Optional[int] = None,
        only_open_access: bool = False,
        ollama_model: Optional[str] = None,
    ) -> Tuple[List[FetchedPaper], Dict[str, str]]:
        """
        Kullanıcının yazdığı tek bir cümle için OpenAlex, Semantic Scholar, Europe PMC (PubMed)
        ve CrossRef üzerinde yayınlanmış gerçek makaleleri araştırır ve özetleriyle birlikte
        FetchedPaper listesi olarak döndürür.
        """
        native_q, english_q = self.build_search_queries_for_claim(claim_sentence, ollama_model=ollama_model)
        queries_to_try = list(dict.fromkeys(q for q in [english_q, native_q] if q and len(q) >= 4))

        candidates_by_key: Dict[str, FetchedPaper] = {}
        ref_counter = 1

        def _add_candidate(fp: FetchedPaper) -> None:
            nonlocal ref_counter
            if fp.is_retracted or not fp.verified_title:
                return
            if not fp.abstract and not fp.tldr and not fp.full_text:
                return
            if year_from and fp.verified_year:
                try:
                    if int(str(fp.verified_year)[:4]) < int(year_from):
                        return
                except ValueError:
                    pass
            if only_open_access and not (fp.pdf_url or fp.full_text or fp.pmcid):
                return

            dedup_key = (
                (fp.verified_doi or "").strip().lower()
                or re.sub(r"[^\w]", "", fp.verified_title.lower())[:65]
            )
            if not dedup_key:
                return

            if dedup_key in candidates_by_key:
                existing = candidates_by_key[dedup_key]
                if len(fp.abstract) > len(existing.abstract):
                    existing.abstract = fp.abstract
                if fp.tldr and not existing.tldr:
                    existing.tldr = fp.tldr
                if fp.pdf_url and not existing.pdf_url:
                    existing.pdf_url = fp.pdf_url
                existing.citation_count = max(existing.citation_count, fp.citation_count)
                for sf in fp.sources_found:
                    if sf not in existing.sources_found:
                        existing.sources_found.append(sf)
                existing.content_source = " + ".join(existing.sources_found)
            else:
                fp.ref_id = f"ADAY_{ref_counter}"
                ref_counter += 1
                fp.exists_online = True
                fp.title_match_score = 1.0
                fp.lookup_status = "found"
                fp.content_source = " + ".join(dict.fromkeys(fp.sources_found)) or "Akademik Veritabanı"
                candidates_by_key[dedup_key] = fp

        # 1. KANAL: OpenAlex (250M+ makale, hem İngilizce hem Türkçe özet desteği)
        for q in queries_to_try:
            try:
                url = "https://api.openalex.org/works"
                params: Dict[str, Any] = {
                    "search": q,
                    "per-page": max(10, max_candidates),
                }
                if year_from:
                    params["filter"] = f"from_publication_date:{int(year_from)}-01-01"
                resp = requests.get(url, params=params, headers=API_HEADERS, timeout=self.timeout)
                if resp.status_code == 200:
                    for w in resp.json().get("results", []):
                        if w.get("is_retracted"):
                            continue
                        title = (w.get("title") or "").strip()
                        inv_abs = w.get("abstract_inverted_index")
                        abs_text = self._reconstruct_openalex_abstract(inv_abs).strip() if inv_abs else ""
                        if not title or len(abs_text) < 60:
                            continue
                        auths = [
                            a.get("author", {}).get("display_name", "")
                            for a in w.get("authorships", [])
                            if a.get("author", {}).get("display_name")
                        ]
                        year = str(w.get("publication_year") or "")
                        doi = (w.get("doi") or "").replace("https://doi.org/", "").strip() or None
                        venue = (
                            ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
                            or "Akademik Dergi"
                        )
                        oa_info = w.get("open_access") or {}
                        best_oa = w.get("best_oa_location") or {}
                        pdf_url = best_oa.get("pdf_url") or oa_info.get("oa_url")

                        _add_candidate(
                            FetchedPaper(
                                ref_id="TEMP",
                                query_title=title,
                                query_doi=doi,
                                exists_online=True,
                                title_match_score=1.0,
                                verified_title=title,
                                verified_authors=auths,
                                verified_year=year,
                                verified_venue=venue,
                                verified_doi=doi,
                                citation_count=int(w.get("cited_by_count") or 0),
                                abstract=abs_text,
                                pdf_url=pdf_url,
                                sources_checked=["OpenAlex"],
                                sources_found=["OpenAlex"],
                            )
                        )
            except Exception:
                pass

        # 2. KANAL: Semantic Scholar (TL;DR + Zengin Özet + Atıf Sayısı)
        for q in queries_to_try[:1]:
            try:
                fields = "title,authors,year,abstract,tldr,openAccessPdf,externalIds,citationCount,venue"
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                resp = requests.get(
                    url,
                    params={"query": q[:180], "limit": max(8, max_candidates), "fields": fields},
                    headers=API_HEADERS,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    for item in resp.json().get("data", []):
                        title = (item.get("title") or "").strip()
                        abs_text = (item.get("abstract") or "").strip()
                        tldr_obj = item.get("tldr")
                        tldr_text = tldr_obj.get("text", "").strip() if isinstance(tldr_obj, dict) else ""
                        if not title or (len(abs_text) < 50 and not tldr_text):
                            continue
                        auths = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
                        year = str(item.get("year") or "")
                        venue = (item.get("venue") or "").strip() or "Akademik Dergi"
                        ext_ids = item.get("externalIds") or {}
                        doi = ext_ids.get("DOI")
                        oa_pdf = item.get("openAccessPdf")
                        pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None

                        _add_candidate(
                            FetchedPaper(
                                ref_id="TEMP",
                                query_title=title,
                                query_doi=doi,
                                exists_online=True,
                                title_match_score=1.0,
                                verified_title=title,
                                verified_authors=auths,
                                verified_year=year,
                                verified_venue=venue,
                                verified_doi=doi,
                                citation_count=int(item.get("citationCount") or 0),
                                abstract=abs_text,
                                tldr=tldr_text,
                                pdf_url=pdf_url,
                                sources_checked=["Semantic Scholar"],
                                sources_found=["Semantic Scholar"],
                            )
                        )
            except Exception:
                pass

        # 3. KANAL: Europe PMC / PubMed (Biyomedikal, Tıp, Yaşam Bilimleri Makaleleri)
        for q in queries_to_try[:1]:
            try:
                url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                resp = requests.get(
                    url,
                    params={
                        "query": q[:180],
                        "resultType": "core",
                        "pageSize": max(8, max_candidates),
                        "format": "json",
                    },
                    headers=API_HEADERS,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    results = resp.json().get("resultList", {}).get("result", [])
                    for item in results:
                        title = re.sub(r"<[^>]+>", "", (item.get("title") or "")).strip().rstrip(".")
                        abs_raw = item.get("abstractText") or ""
                        abs_text = re.sub(r"<[^>]+>", " ", abs_raw)
                        abs_text = re.sub(r"\s+", " ", abs_text).strip()
                        if not title or len(abs_text) < 60:
                            continue
                        auth_str = item.get("authorString") or ""
                        auths = [a.strip() for a in auth_str.rstrip(".").split(",") if a.strip()]
                        year = str(item.get("pubYear") or "")
                        venue = item.get("journalTitle") or "PubMed / Europe PMC"
                        doi = item.get("doi")
                        pmid = item.get("pmid")
                        pmcid = item.get("pmcid")
                        pdf_url = (
                            f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                            if pmcid
                            else None
                        )

                        _add_candidate(
                            FetchedPaper(
                                ref_id="TEMP",
                                query_title=title,
                                query_doi=doi,
                                exists_online=True,
                                title_match_score=1.0,
                                verified_title=title,
                                verified_authors=auths,
                                verified_year=year,
                                verified_venue=venue,
                                verified_doi=doi,
                                pmid=pmid,
                                pmcid=pmcid,
                                citation_count=int(item.get("citedByCount") or 0),
                                abstract=abs_text,
                                pdf_url=pdf_url,
                                sources_checked=["PubMed / Europe PMC"],
                                sources_found=["PubMed / Europe PMC"],
                            )
                        )
            except Exception:
                pass

        all_candidates = list(candidates_by_key.values())
        return all_candidates[: max_candidates * 2], {
            "native_query": native_q,
            "english_query": english_q,
            "total_raw_found": str(len(all_candidates)),
        }

    def suggest_real_papers(self, claim_sentence: str, top_k: int = 3) -> List[AlternativePaper]:
        """
        Eğer kullanıcının kaynağı sahteyse (halüsinasyon) veya iddiayı desteklemiyorsa,
        kullanıcının iddia cümlesini destekleyen GERÇEK ve yüksek atıflı makaleleri
        OpenAlex ve Semantic Scholar üzerinden bulup önerir.
        """
        # İddia cümlesindeki parantez içi atıfları temizle
        clean_claim = re.sub(r"\[[^\]]+\]|\([^)]*\d{4}[^)]*\)", "", claim_sentence)
        clean_claim = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", " ", clean_claim)
        words = [w for w in clean_claim.split() if len(w) > 3]
        search_query = " ".join(words[:14])
        if not search_query:
            return []

        suggestions: List[AlternativePaper] = []
        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": search_query,
                "per-page": top_k * 2,
                "sort": "cited_by_count:desc",
            }
            resp = requests.get(url, params=params, headers=API_HEADERS, timeout=self.timeout)
            if resp.status_code == 200:
                for w in resp.json().get("results", []):
                    if w.get("is_retracted"):
                        continue
                    title = w.get("title") or ""
                    if not title:
                        continue
                    auths = [
                        a.get("author", {}).get("display_name", "")
                        for a in w.get("authorships", [])[:3]
                    ]
                    auth_str = ", ".join(a for a in auths if a)
                    if len(w.get("authorships", [])) > 3:
                        auth_str += " et al."
                    year = str(w.get("publication_year") or "")
                    doi = (w.get("doi") or "").replace("https://doi.org/", "")
                    venue = (
                        (w.get("primary_location") or {})
                        .get("source", {})
                        .get("display_name", "Akademik Yayın")
                        if (w.get("primary_location") or {}).get("source")
                        else "Akademik Yayın"
                    )
                    inv_abs = w.get("abstract_inverted_index")
                    abs_text = self._reconstruct_openalex_abstract(inv_abs)[:320] + "..." if inv_abs else ""

                    suggestions.append(
                        AlternativePaper(
                            title=title,
                            authors=auth_str or "Bilinmiyor",
                            year=year,
                            venue=venue,
                            doi=doi,
                            citation_count=int(w.get("cited_by_count") or 0),
                            abstract_snippet=abs_text,
                            source="OpenAlex",
                        )
                    )
                    if len(suggestions) >= top_k:
                        break
        except Exception:
            pass

        return suggestions

    # -----------------------------------------------------------------------
    # Yardımcı Fonksiyonlar: PDF İndirme, Metin Çıkarma ve Benzerlik Hesabı
    # -----------------------------------------------------------------------

    def _download_and_extract_pdf(self, pdf_url: str, paper: FetchedPaper, source_label: str) -> str:
        """PDF dosyasını indirir, .cache/pdfs/ altına kaydeder ve tüm sayfalarını okur."""
        if not is_safe_public_url(pdf_url):
            self._record_lookup_error(
                paper, source_label, ValueError("Güvenli olmayan/özel ağ adresi reddedildi")
            )
            return ""
        try:
            resp = self.session.get(pdf_url, timeout=self.timeout + 6, allow_redirects=True)
            if resp.status_code != 200:
                return ""
            # PDF imzası (%PDF) kontrolü
            if not resp.content.startswith(b"%PDF"):
                return ""
            return self._save_and_read_pdf_bytes(resp.content, paper, source_label)
        except Exception:
            return ""

    def _save_and_read_pdf_bytes(self, pdf_bytes: bytes, paper: FetchedPaper, source_label: str) -> str:
        try:
            file_hash = hashlib.md5(pdf_bytes[:4096]).hexdigest()[:12]
            pdf_path = PDF_CACHE_DIR / f"ref_{safe_ref_token(paper.ref_id)}_{file_hash}.pdf"
            pdf_path.write_bytes(pdf_bytes)
            paper.local_pdf_path = str(pdf_path)

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            # Çok uzun kitaplarda sistemi yormamak için ilk 35 sayfayı oku
            for page_idx in range(min(len(doc), 35)):
                txt = doc[page_idx].get_text("text")
                txt = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", txt)
                pages.append(txt)
            doc.close()

            full_txt = "\n\n".join(pages).strip()
            if len(full_txt) > 300:
                paper.sources_found.append(source_label)
                return full_txt
        except Exception:
            pass
        return ""

    def _check_user_local_pdfs(self, ref: ReferenceItem, paper: FetchedPaper, local_dir: Path) -> None:
        """Kullanıcının kendi bilgisayarından yüklediği PDF'ler içinde eşleşen makale varsa okur."""
        for pdf_file in local_dir.glob("*.pdf"):
            try:
                doc = fitz.open(str(pdf_file))
                first_pages = "\n".join(doc[i].get_text("text") for i in range(min(2, len(doc))))
                doc.close()

                # DOI veya başlık ilk 2 sayfada geçiyor mu?
                matched = False
                if ref.doi and ref.doi.lower() in first_pages.lower():
                    matched = True
                elif ref.title and len(ref.title) >= 12:
                    sim = self._title_similarity(ref.title.lower(), first_pages[:1000].lower(), first_pages.lower())
                    if sim >= 0.72 or ref.title.lower()[:35] in first_pages.lower():
                        matched = True

                if matched:
                    pdf_bytes = pdf_file.read_bytes()
                    full_txt = self._save_and_read_pdf_bytes(pdf_bytes, paper, "Yerel PDF")
                    if full_txt:
                        paper.exists_online = True
                        paper.full_text = full_txt
                        paper.title_match_score = 1.0
                        if not paper.verified_title:
                            paper.verified_title = ref.title
                        return
            except Exception:
                continue

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Aksanlı Latin harfleri düzleştirir (Réus -> Reus, Castrén -> Castren) ancak Türkçe harfleri korur."""
        tr_map = {
            "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
            "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u",
        }
        out = []
        for ch in text:
            if ch in tr_map:
                out.append(tr_map[ch])
            else:
                decomposed = unicodedata.normalize("NFKD", ch)
                out.append("".join(c for c in decomposed if not unicodedata.combining(c)))
        return "".join(out)

    @classmethod
    def _normalize_academic_text(cls, text: str) -> str:
        """İngiliz/Amerikan bilimsel yazım farklarını (behavioural -> behavioral vb.) ve aksanları eşitler."""
        s = cls._strip_accents((text or "").lower())
        uk_us_pairs = [
            (r"\bbehaviour", "behavior"),
            (r"\bsignalling\b", "signaling"),
            (r"\bmodelling\b", "modeling"),
            (r"\blabelled\b", "labeled"),
            (r"\btumour", "tumor"),
            (r"\bcharacteris", "characteriz"),
            (r"\brandomis", "randomiz"),
            (r"\banaesthes", "anesthes"),
            (r"\boestrogen", "estrogen"),
            (r"\bhaematol", "hematol"),
            (r"\bleukaem", "leukem"),
            (r"\bpaediatr", "pediatr"),
            (r"\bfoetal\b", "fetal"),
            (r"\boesophag", "esophag"),
            (r"\bischaem", "ischem"),
            (r"\boedema\b", "edema"),
            (r"\bneurone", "neuron"),
        ]
        for pat, repl in uk_us_pairs:
            s = re.sub(pat, repl, s)
        return s

    def _validate_metadata_consistency(self, ref: ReferenceItem, paper: FetchedPaper) -> None:
        """Kullanıcının yazdığı yıl/yazar ile internetteki resmi künye arasında fark varsa uyarı ekler."""
        if not paper.exists_online:
            paper.metadata_warnings.append(
                "🚨 Bu makale CrossRef, OpenAlex, PubMed, Semantic Scholar ve Google Scholar dizinlerinde BULUNAMADI! "
                "Yapay zeka tarafından uydurulmuş (halüsinasyon) veya başlığı yanlış yazılmış olabilir."
            )
            return

        if paper.is_retracted:
            paper.metadata_warnings.append(
                f"🛑 DİKKAT (GERİ ÇEKİLMİŞ MAKALE): {paper.retraction_details}"
            )

        # Yıl kontrolü (1 yıldan fazla fark varsa uyar; online/print farkı 1 yıl olabilir)
        if ref.year and paper.verified_year:
            try:
                y1 = int(ref.year[:4])
                y2 = int(paper.verified_year[:4])
                if abs(y1 - y2) > 1:
                    paper.metadata_warnings.append(
                        f"⚠️ Yıl Uyuşmazlığı: Kaynakçada '{y1}' yazılmış ancak makalenin resmi yayın yılı '{y2}'."
                    )
            except ValueError:
                pass

        # İlk yazar soyadı kontrolü (Aksan duyarsız: Réus == Reus, Castrén == Castren)
        if ref.first_author_surname and paper.verified_authors:
            all_authors_norm = self._strip_accents(" ".join(paper.verified_authors).lower())
            ref_surname_norm = self._strip_accents(ref.first_author_surname.lower())
            if ref_surname_norm not in all_authors_norm:
                paper.metadata_warnings.append(
                    f"⚠️ Yazar Uyuşmazlığı: Kaynakçada '{ref.first_author_surname}' belirtilmiş ancak "
                    f"makalenin resmi yazarları: {', '.join(paper.verified_authors[:4])}."
                )

    @classmethod
    def _title_similarity(cls, query_title: str, candidate_title: str, full_raw_ref: str = "") -> float:
        """
        İki makale başlığı arasındaki normalize benzerlik skorunu hesaplar (0.0 - 1.0).
        Genel akademik dolgu kelimelerini ('mechanism of action of' vb.) filtreleyerek
        spesifik bilimsel terimlerin (ilaç, gen, reseptör, hastalık adı) örtüşmesini denetler.
        """
        q_clean = re.sub(r"[^\w\s]", " ", cls._normalize_academic_text(query_title))
        c_clean = re.sub(r"[^\w\s]", " ", cls._normalize_academic_text(candidate_title))
        q_clean = re.sub(r"\s+", " ", q_clean).strip()
        c_clean = re.sub(r"\s+", " ", c_clean).strip()

        if not q_clean or not c_clean:
            return 0.0
        len_ratio = min(len(q_clean), len(c_clean)) / max(1, max(len(q_clean), len(c_clean)))
        if (c_clean in q_clean or q_clean in c_clean) and len_ratio >= 0.78:
            if min(len(q_clean), len(c_clean)) >= 18:
                return 0.95
        if full_raw_ref and len(c_clean) >= 22:
            raw_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", cls._normalize_academic_text(full_raw_ref))).strip()
            if c_clean in raw_norm and len_ratio >= 0.75:
                return 0.95

        seq_ratio = SequenceMatcher(None, q_clean, c_clean).ratio()

        q_words = {w for w in q_clean.split() if len(w) > 2}
        c_words = {w for w in c_clean.split() if len(w) > 2}
        if not q_words or not c_words:
            return seq_ratio

        # Genel akademik dolgu kelimeleri (bunların eşleşmesi tek başına makaleyi gerçek yapmaz!)
        generic_academic_words = {
            "the", "and", "for", "with", "from", "into", "upon", "between", "among", "through",
            "during", "under", "after", "before", "over", "about", "within", "without", "towards",
            "study", "studies", "review", "analysis", "evaluation", "investigation", "assessment",
            "effect", "effects", "role", "roles", "involvement", "mechanism", "mechanisms",
            "action", "actions", "model", "models", "induced", "associated", "mediated", "related",
            "acute", "chronic", "mild", "repeated", "novel", "new", "potential", "therapeutic",
            "rats", "rat", "mice", "mouse", "rodents", "rodent", "patients", "human", "humans",
            "animal", "animals", "experimental", "clinical", "treatment", "activity", "expression",
            "levels", "function", "functions", "alterations", "changes", "response", "responses",
            "bir", "icin", "uzerine", "etkisi", "incelenmesi", "degerlendirilmesi", "arastirilmasi",
        }
        q_distinct = q_words - generic_academic_words
        c_distinct = c_words - generic_academic_words

        # Eğer sorgu başlığında ayırt edici spesifik bilimsel terimler varsa (örn: "bupropion", "trpm2", "slc6a15"),
        # aday makale başlığında bu spesifik terimlerin en az yarısı geçmek ZORUNDADIR!
        if q_distinct:
            distinct_recall = len(q_distinct & c_distinct) / len(q_distinct)
            if distinct_recall < 0.45:
                return min(0.40, seq_ratio * 0.65)

        # Dice / F1 kelime örtüşmesi (min yerine toplam kelime sayısına oranla ki kısa başlıklar haksız puan almasın)
        dice_overlap = (2.0 * len(q_words & c_words)) / (len(q_words) + len(c_words))
        return max(seq_ratio, dice_overlap * 0.94)

    @staticmethod
    def _make_cache_key(raw_text: str) -> str:
        norm = "v2:" + re.sub(r"\s+", " ", raw_text.lower()).strip()
        return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Hızlı Modül Testi (1 Gerçek Makale + 1 Uydurma/Sahte Makale Testi)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    fetcher = AcademicFetcher()

    # Test 1: Gerçek ve ünlü bir makale (Attention Is All You Need / veya DOI'li tıbbi makale)
    real_ref = ReferenceItem(
        ref_id="1",
        number=1,
        raw_text="[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.",
        title="Attention is all you need",
        authors="Vaswani, A., Shazeer, N.",
        first_author_surname="Vaswani",
        year="2017",
    )

    # Test 2: Yapay zekanın uydurduğu (Halüsinasyon) sahte bir makale
    fake_ref = ReferenceItem(
        ref_id="2",
        number=2,
        raw_text="[2] Korkmaz, Z., & टेली, M. (2023). Quantum Hyper-Dimensional Telepathic Neural Networks for Satellite Photosynthesis. Journal of Imaginary Science, 99(1), 1-15.",
        title="Quantum Hyper-Dimensional Telepathic Neural Networks for Satellite Photosynthesis",
        authors="Korkmaz, Z.",
        first_author_surname="Korkmaz",
        year="2023",
    )

    print("=== ADIM 3: AKADEMİK İNTERNET AVCISI (academic_fetcher.py) TESTİ ===")
    for test_ref in (real_ref, fake_ref):
        print(f"\nSorgulanıyor: [{test_ref.ref_id}] {test_ref.title}")
        res = fetcher.verify_and_fetch(test_ref)
        print(f"  -> İnternette Gerçek mi? : {'✅ EVET' if res.exists_online else '🚨 HAYIR (SAHTE / BULUNAMADI)'}")
        print(f"  -> Bulunan Kaynaklar     : {res.content_source}")
        print(f"  -> Doğrulanan Başlık     : {res.verified_title or '-'}")
        print(f"  -> Yazarlar & Yıl        : {', '.join(res.verified_authors[:3])} ({res.verified_year or '-'})")
        print(f"  -> Özet Uzunluğu         : {len(res.abstract)} karakter")
        print(f"  -> Tam Metin Uzunluğu    : {len(res.full_text)} karakter")
        if res.metadata_warnings:
            for w in res.metadata_warnings:
                print(f"  -> Uyarı                 : {w}")
