"""
MODÜL 3: Çok Dilli Semantik RAG & Anlam Çıkarımı Motoru (rag_verifier.py)
-------------------------------------------------------------------------
Görevleri:
1. RTX 3050 (4 GB VRAM) dostu mimari:
   - Çok dilli vektörleştirme modelini (`paraphrase-multilingual-MiniLM-L12-v2`)
     SADECE CPU üzerinde çalıştırır (ekran kartı belleğini hiç harcamaz).
   - Türkçe bir iddia cümlesi ile İngilizce veya Türkçe referans makale metnini
     aynı matematiksel anlam uzayında eşleştirir.
2. Kelimeye Değil, ANLAMA Bakan Doğrulama (Semantic Entailment / NLI):
   - "Dünyanın yüzde 90'ı sudan oluşur" <-> "Yüzde doksanı su olan dünyamız..."
     gibi farklı kelimelerle yazılmış (paraphrase), etken/edilgen çatısı değişmiş
     veya İngilizce'den Türkçe'ye çevrilmiş cümlelerin aynı anlama geldiğini tespit eder.
   - Türkçe ve İngilizce yazılı sayıları ("yüzde doksan" -> "%90", "ninety percent" -> "90%")
     otomatik normalize ederek sayısal tutarlılığı ve çelişkileri yakalar.
3. Yerel Ollama (Llama 3.2 / Qwen 2.5) Entegrasyonu:
   - Seçilen en güçlü kanıt paragraflarını yerel LLM'e göndererek adım adım
     mantıksal çıkarım (Chain-of-Thought) yapar.
   - İkincil atıf (makalenin kendi bulgusu değil, başkasına yaptığı atıf) uyarısı verir.
   - Çelişki varsa kullanıcının makalesine doğrudan yapıştırabileceği düzeltilmiş
     Türkçe akademik cümle önerisi üretir.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from academic_fetcher import FetchedPaper, AlternativePaper


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

MULTILINGUAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREFERRED_OLLAMA_MODELS = [
    "qwen2.5:3b",
    "llama3.2:latest",
    "llama3.2:3b",
    "llama3.2:1b",
    "qwen2.5:1.5b",
    "gemma2:2b",
    "llama3.1:8b",
]

# Zayıf künye eşleşmesi: bu skorun altındaki kaynaklarla anlam karşılaştırması
# yapılmaz; sonuç "DOGRULANAMADI" olur (yanlış makaleyle hüküm vermemek için).
MIN_TRUSTED_MATCH = 0.70


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class EvidenceChunk:
    """Referans makaleden cımbızla çekilen kanıt paragrafı."""
    chunk_id: int
    text: str
    semantic_score: float              # 0.0 - 1.0 arası çok dilli anlam yakınlığı
    is_secondary_citation: bool = False  # Bu paragraf başka bir makaleye mi atıf yapıyor?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Tek bir atıf/iddia için nihai doğrulama ve anlam çıkarım sonucu."""
    citation_id: str
    marker: str
    claim_sentence: str
    context_text: str
    ref_id: Optional[str]
    verdict: str                       # "DESTEKLENİYOR" | "ÇELİŞİYOR" | "KANIT_YETERSİZ" | "DOGRULANAMADI" | "SAHTE_KAYNAK" | "HAYALET_ATIF"
    confidence_score: float            # 0.0 - 1.0 güven skoru
    semantic_similarity: float         # En iyi kanıt paragrafının anlam benzerliği (0.0 - 1.0)
    best_evidence: str                 # Makaleden bulunan en güçlü kanıt paragrafı
    match_score: float = 0.0           # Kaynak künyesinin eşleşme güveni (0.0 - 1.0)
    top_chunks: List[EvidenceChunk] = field(default_factory=list)
    explanation: str = ""              # Neden destekleniyor / neden çelişiyor açıklaması
    claim_core_meaning: str = ""       # Cümlenin öz anlamı
    evidence_core_meaning: str = ""    # Kanıtın öz anlamı
    suggested_revision: str = ""       # Hata/çelişki varsa önerilen düzeltilmiş akademik cümle
    is_secondary_citation: bool = False
    secondary_citation_note: str = ""
    is_retracted: bool = False
    metadata_warnings: List[str] = field(default_factory=list)
    alternative_papers: List[AlternativePaper] = field(default_factory=list)
    used_llm_model: str = ""
    decision_source: str = ""          # "llm" | "kural" | "kural+llm" — kararı kim verdi?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Türkçe & İngilizce Sayısal İfade Normalleştirici (Yüzde Doksan <-> %90)
# ---------------------------------------------------------------------------

TR_NUM_WORDS = {
    "sıfır": 0, "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5,
    "altı": 6, "yedi": 7, "sekiz": 8, "dokuz": 9, "on": 10,
    "yirmi": 20, "otuz": 30, "kırk": 40, "elli": 50,
    "altmış": 60, "yetmiş": 70, "seksen": 80, "doksan": 90, "yüz": 100,
}

EN_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


def normalize_semantic_text(text: str) -> str:
    """
    Metin içindeki yazılı sayı ve yüzdelik ifadelerin yanına rakamsal karşılığını ekler.
    Örn: "yüzde doksanı su olan" -> "yüzde doksanı (%90) su olan"
         "yüzde 90'ı" -> "yüzde 90'ı (%90)"
         "ninety percent" -> "ninety percent (90%)"
    Böylece hem embedding modeli hem de sayısal karşılaştırıcı tam eşleşmeyi yakalar.
    """
    out = text

    # 1. Türkçe "yüzde <kelime> [<kelime>]" -> (%XX)
    def _tr_pct_repl(match: re.Match) -> str:
        full = match.group(0)
        w1 = match.group(1).lower()
        w2 = (match.group(2) or "").lower()
        # Sondaki çekim eklerini temizle (doksanı -> doksan, ellisi -> elli)
        w1_root = _strip_tr_num_suffix(w1)
        w2_root = _strip_tr_num_suffix(w2) if w2 else ""
        val = 0
        if w1_root in TR_NUM_WORDS:
            val += TR_NUM_WORDS[w1_root]
        if w2_root in TR_NUM_WORDS:
            val += TR_NUM_WORDS[w2_root]
        if val > 0:
            return f"{full} (%{val})"
        return full

    out = re.sub(
        r"\byüzde\s+([a-zçğıöşü]+)(?:\s+([a-zçğıöşü]+))?\b",
        _tr_pct_repl,
        out,
        flags=re.IGNORECASE,
    )

    # 2. Türkçe "yüzde 90'ı" -> "yüzde 90'ı (%90)"
    out = re.sub(
        r"\byüzde\s+(\d+(?:[.,]\d+)?)\s*(?:['’]?[a-zçğıöşü]*)?",
        lambda m: f"{m.group(0)} (%{m.group(1).replace(',', '.')})",
        out,
        flags=re.IGNORECASE,
    )

    # 3. İngilizce "ninety percent" / "eighty five percent"
    def _en_pct_repl(match: re.Match) -> str:
        full = match.group(0)
        w1 = match.group(1).lower()
        w2 = (match.group(2) or "").lower()
        val = EN_NUM_WORDS.get(w1, 0) + EN_NUM_WORDS.get(w2, 0)
        if val > 0:
            return f"{full} ({val}%)"
        return full

    out = re.sub(
        r"\b([a-z]+)(?:[\s\-]+([a-z]+))?\s+per\s*cent\b",
        _en_pct_repl,
        out,
        flags=re.IGNORECASE,
    )

    # 4. Kesirli/Oransal ifadeler: "yarı yarıya" -> "(%50)", "dörtte üçü" -> "(%75)"
    out = re.sub(r"\byarı\s+yarıya\b|\byarısı\b", lambda m: f"{m.group(0)} (%50)", out, flags=re.IGNORECASE)
    out = re.sub(r"\bdörtte\s+üç[üu]?\b|\bthree[\s\-]quarters\b", lambda m: f"{m.group(0)} (%75)", out, flags=re.IGNORECASE)
    out = re.sub(r"\bdörtte\s+bir[i]?\b|\bone[\s\-]quarter\b", lambda m: f"{m.group(0)} (%25)", out, flags=re.IGNORECASE)

    return out


def _strip_tr_num_suffix(word: str) -> str:
    """'doksanı' -> 'doksan', 'ellisi' -> 'elli', 'sekzeni' -> 'seksen' kök bulucu."""
    w = word.lower().strip(" '’")
    for k in sorted(TR_NUM_WORDS.keys(), key=len, reverse=True):
        if w == k or (w.startswith(k) and len(w) - len(k) <= 4):
            return k
    return w


def extract_numbers_and_percentages(text: str) -> Tuple[List[float], List[float]]:
    """
    Normalize edilmiş metindeki yüzdelik değerleri (percentages) ve genel sayıları (numbers)
    ayrı listeler halinde döndürür. (Yılları 1950-2035 filtreler).
    """
    norm = normalize_semantic_text(text)

    # Yüzdeler: %90, 90%, %92.5
    pct_matches = re.findall(r"(?:%\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%)", norm)
    percentages: List[float] = []
    for m1, m2 in pct_matches:
        val_str = (m1 or m2).replace(",", ".")
        try:
            v = float(val_str)
            if v not in percentages:
                percentages.append(v)
        except ValueError:
            pass

    # Genel sayılar (yıllar ve referans numaraları hariç)
    cleaned_for_nums = re.sub(r"\[[^\]]*\]|\([^)]*\d{4}[^)]*\)", " ", norm)
    raw_nums = re.findall(r"\b(\d+(?:[.,]\d+)?)\b", cleaned_for_nums)
    numbers: List[float] = []
    for rn in raw_nums:
        try:
            v = float(rn.replace(",", "."))
            if 1950 <= v <= 2035:
                continue
            if v not in percentages and v not in numbers:
                numbers.append(v)
        except ValueError:
            pass

    return percentages, numbers


# ---------------------------------------------------------------------------
# Olumsuzluk (negasyon) sinyali
# ---------------------------------------------------------------------------
# Amaç: "X azaltır" ile "X azaltmaz" gibi cümleleri embedding benzerliği
# yakalayamaz; bu yüzden ayrı ve dar kapsamlı bir sözlük taraması yapılır.
# Kural yalnızca ÇELİŞKİ yönünde sinyal üretir; asla "destekleniyor" üretmez.

TR_NEGATION_RE = re.compile(
    r"\b(?:değil|değildir|yok|yoktur|olmaz|olmadı|olmayan|etmez|etmedi|etmemiştir|"
    r"göstermez|göstermedi|göstermemiştir|azaltmaz|azaltmadı|azaltmamıştır|"
    r"artmaz|artmadı|artmamıştır|düşürmez|yükseltmez|sağlamaz|sağlamadı|"
    r"bulunmamış|saptanmamış|gözlenmemiş|asla|hiçbir)\w*",
    re.IGNORECASE,
)

EN_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|none|neither|nor|"
    r"does\s+not|doesn't|did\s+not|didn't|is\s+not|isn't|was\s+not|wasn't|"
    r"were\s+not|weren't|cannot|can't|fail(?:s|ed)?\s+to|failed\s+to)\b",
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    """Metinde dar kapsamlı bir olumsuzluk ifadesi var mı?"""
    if not text:
        return False
    return bool(TR_NEGATION_RE.search(text) or EN_NEGATION_RE.search(text))


# ---------------------------------------------------------------------------
# Ana Doğrulama ve Semantik RAG Sınıfı (SemanticRAGVerifier)
# ---------------------------------------------------------------------------

class SemanticRAGVerifier:
    """
    CPU üzerinde Çok Dilli Vektör Arama + RTX 3050 üzerinde Ollama NLI ile
    kelimeye değil ANLAMA bakan akademik atıf doğrulayıcı.
    """

    def __init__(
        self,
        ollama_model: Optional[str] = None,
        embedding_device: str = "cpu",
    ):
        self.embedding_device = embedding_device
        self._embedder: Optional[SentenceTransformer] = None
        self.ollama_model = ollama_model or self._detect_best_ollama_model()

    # -----------------------------------------------------------------------
    # 1. Model Yükleme (Lazy Loading & CPU Sabitleme)
    # -----------------------------------------------------------------------

    @property
    def embedder(self) -> SentenceTransformer:
        """Embedding modelini ilk kullanımda CPU üzerinde yükler (VRAM'i korur)."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(
                MULTILINGUAL_EMBEDDING_MODEL,
                device=self.embedding_device,
            )
        return self._embedder

    def _detect_best_ollama_model(self) -> str:
        """Yerel Ollama'da kurulu modelleri kontrol eder ve RTX 3050'ye en uygun olanı seçer."""
        try:
            models_resp = ollama.list()
            installed_names: List[str] = []
            # ollama.list() sürümüne göre hem nesne hem dict dönebilir
            models_list = getattr(models_resp, "models", None) or models_resp.get("models", [])
            for m in models_list:
                name = getattr(m, "model", None) or getattr(m, "name", None) or m.get("model") or m.get("name")
                if name:
                    installed_names.append(str(name))

            for pref in PREFERRED_OLLAMA_MODELS:
                for inst in installed_names:
                    if pref.lower() == inst.lower() or inst.lower().startswith(pref.split(":")[0].lower()):
                        return inst

            if installed_names:
                return installed_names[0]
        except Exception:
            pass
        return "llama3.2"

    def check_ollama_status(self) -> Tuple[bool, str, List[str]]:
        """Ollama servisinin çalışıp çalışmadığını ve yüklü modelleri döndürür."""
        try:
            models_resp = ollama.list()
            models_list = getattr(models_resp, "models", None) or models_resp.get("models", [])
            names = []
            for m in models_list:
                name = getattr(m, "model", None) or getattr(m, "name", None) or m.get("model") or m.get("name")
                if name:
                    names.append(str(name))
            return True, self.ollama_model, names
        except Exception as e:
            return False, self.ollama_model, [str(e)]

    # -----------------------------------------------------------------------
    # 2. Metin Parçalama (Chunking) ve Çok Dilli Vektör Arama
    # -----------------------------------------------------------------------

    def chunk_paper_text(self, text: str, chunk_size: int = 550, overlap: int = 120) -> List[str]:
        """Makale metnini cümle bütünlüğünü bozmadan örtüşmeli paragraflara böler."""
        if not text or not text.strip():
            return []

        cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(cleaned) <= chunk_size:
            return [cleaned]

        # Önce cümle sonlarından böl
        sentences = re.split(r"(?<=[.!?])\s+|\n\n+", cleaned)
        chunks: List[str] = []
        current = ""

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 <= chunk_size:
                current = f"{current} {sent}".strip() if current else sent
            else:
                if current:
                    chunks.append(current)
                # Örtüşme (overlap): önceki parçanın son kısmını koru
                if overlap > 0 and current:
                    tail = current[-overlap:]
                    # Kelime ortasından bölünmesin
                    space_idx = tail.find(" ")
                    if space_idx != -1:
                        tail = tail[space_idx + 1 :]
                    current = f"{tail} {sent}".strip()
                else:
                    current = sent

        if current:
            chunks.append(current)

        return chunks[:250]  # Çok büyük PDF'lerde en anlamlı ilk 250 parçayı sınırla

    def retrieve_top_evidence(
        self, claim_sentence: str, paper_text: str, top_k: int = 5
    ) -> List[EvidenceChunk]:
        """
        Kullanıcının iddia cümlesiyle (Türkçe veya İngilizce) referans makalenin
        paragraflarını Çok Dilli Semantik Vektör uzayında karşılaştırır ve
        anlamca en yakın kanıt paragraflarını döndürür.
        """
        chunks = self.chunk_paper_text(paper_text)
        if not chunks:
            return []

        # Atıf işaretlerini ([1], (Yılmaz, 2022)) temizleyip sayıları normalize et
        clean_claim = re.sub(r"\[[^\]]+\]|\([^)]*\d{4}[^)]*\)", "", claim_sentence).strip()
        norm_claim = normalize_semantic_text(clean_claim)
        norm_chunks = [normalize_semantic_text(c) for c in chunks]

        # CPU üzerinde vektörleştir (normalize_embeddings=True ile dot product = cosine similarity)
        claim_vec = self.embedder.encode([norm_claim], normalize_embeddings=True)[0]
        chunk_vecs = self.embedder.encode(norm_chunks, normalize_embeddings=True, batch_size=32)

        cosine_scores = np.dot(chunk_vecs, claim_vec)

        # Sayısal / yüzdelik eşleşme bonusu (Örn: %90 == yüzde doksan yakalandığında o paragrafı öne çıkar)
        claim_pcts, claim_nums = extract_numbers_and_percentages(norm_claim)

        scored_chunks: List[EvidenceChunk] = []
        for idx, (raw_chunk, norm_c, cos_sim) in enumerate(zip(chunks, norm_chunks, cosine_scores)):
            score = float(cos_sim)

            # Paragraf içindeki tekil cümlelerle de karşılaştır (Max-Pooling):
            # Uzun bir paragrafın içindeki tek bir cümle iddiayı %100 karşılıyorsa diğer cümleler skoru düşürmesin
            sub_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", norm_c) if len(s.strip()) > 15]
            if len(sub_sents) > 1:
                sent_vecs = self.embedder.encode(sub_sents, normalize_embeddings=True, batch_size=16)
                max_sent_sim = float(np.max(np.dot(sent_vecs, claim_vec)))
                score = max(score, max_sent_sim)

            c_pcts, c_nums = extract_numbers_and_percentages(norm_c)

            # Eğer iddiada bir yüzde varsa ve bu paragrafta da yüzde geçiyorsa kanıt olma ihtimali çok yüksektir
            if claim_pcts and c_pcts:
                score = min(1.0, score + 0.08)
            elif claim_nums and c_nums and set(claim_nums) & set(c_nums):
                score = min(1.0, score + 0.06)

            is_sec = self._detect_secondary_citation_in_chunk(raw_chunk)
            scored_chunks.append(
                EvidenceChunk(
                    chunk_id=idx + 1,
                    text=raw_chunk,
                    semantic_score=round(score, 4),
                    is_secondary_citation=is_sec,
                )
            )

        scored_chunks.sort(key=lambda x: x.semantic_score, reverse=True)
        return scored_chunks[:top_k]

    @staticmethod
    def _detect_secondary_citation_in_chunk(chunk_text: str) -> bool:
        """
        Kanıt paragrafının makalenin kendi bulgusu değil de başka bir çalışmaya yaptığı
        ikincil bir atıf olup olmadığını kontrol eder.
        """
        secondary_patterns = [
            r"\[\d{1,3}(?:,\s*\d{1,3})*\]",
            r"\b(?:et\s+al\.|ve\s+ark\.|vd\.)\s*\(\d{4}\)",
            r"\b(?:as\s+reported\s+by|according\s+to|demonstrated\s+by|tarafından\s+bildirilmiştir|çalışmasında\s+belirtildiği\s+üzere)\b",
        ]
        hits = sum(1 for p in secondary_patterns if re.search(p, chunk_text, re.IGNORECASE))
        return hits >= 1

    # -----------------------------------------------------------------------
    # 3. Ana Doğrulama Fonksiyonu: İddia + Kaynak Makale -> Karar
    # -----------------------------------------------------------------------

    def verify_claim_against_paper(
        self,
        claim_sentence: str,
        paper: FetchedPaper,
        citation_id: str = "CIT_1",
        marker: str = "[1]",
        context_text: str = "",
        alternative_finder: Optional[Any] = None,
    ) -> VerificationResult:
        """
        1. Makale sahteyse (internette yoksa) -> SAHTE_KAYNAK döner ve alternatif gerçek makaleler önerir.
        2. Makale gerçekse -> Kanıt paragraflarını Çok Dilli RAG ile bulur.
        3. Ollama (veya Semantik NLI Analizörü) ile anlam çıkarımı yapar.
        """
        # Durum 1: Kaynak doğrulanamadı (hiç bulunamadı veya ağ hatası)
        if not paper.exists_online:
            verdict, explanation = self._classify_missing_source(paper)
            alt_papers = []
            if verdict == "SAHTE_KAYNAK" and alternative_finder is not None:
                alt_papers = alternative_finder.suggest_real_papers(claim_sentence, top_k=3)

            return VerificationResult(
                citation_id=citation_id,
                marker=marker,
                claim_sentence=claim_sentence,
                context_text=context_text or claim_sentence,
                ref_id=paper.ref_id,
                verdict=verdict,
                confidence_score=0.90 if verdict == "SAHTE_KAYNAK" else 0.50,
                semantic_similarity=0.0,
                best_evidence="",
                explanation=explanation,
                metadata_warnings=paper.metadata_warnings,
                alternative_papers=alt_papers,
                used_llm_model="Kaynak Doğrulama Motoru",
                decision_source="kural",
            )

        # Durum 2: Künye zayıf eşleştiyse yanlış makaleyle hüküm verme
        trusted, match_score, match_reason = self._match_confidence(paper)
        if not trusted:
            return VerificationResult(
                citation_id=citation_id,
                marker=marker,
                claim_sentence=claim_sentence,
                context_text=context_text or claim_sentence,
                ref_id=paper.ref_id,
                verdict="DOGRULANAMADI",
                confidence_score=0.50,
                semantic_similarity=0.0,
                best_evidence="",
                match_score=match_score,
                explanation=(
                    f"Kaynak künyesi zayıf eşleşti ({match_reason}; benzerlik "
                    f"%{match_score * 100:.0f}). Yanlış makaleyle anlam karşılaştırması yapmamak için "
                    "karar verilmedi. Lütfen künyeyi (yazar/yıl/DOI) elle doğrulayın veya makalenin "
                    "PDF'ini yükleyin."
                ),
                metadata_warnings=paper.metadata_warnings,
                used_llm_model="Künye Güven Denetçisi",
                decision_source="kural",
            )

        evidence_corpus = paper.combined_evidence_text
        if not evidence_corpus or len(evidence_corpus) < 30:
            return VerificationResult(
                citation_id=citation_id,
                marker=marker,
                claim_sentence=claim_sentence,
                context_text=context_text or claim_sentence,
                ref_id=paper.ref_id,
                verdict="KANIT_YETERSİZ",
                confidence_score=0.50,
                semantic_similarity=0.0,
                best_evidence="",
                explanation=(
                    "Makalenin künyesi internette doğrulandı (gerçek makale), ancak yayıncı "
                    "tarafından özet veya açık metin paylaşılmadığı için içerik karşılaştırması yapılamadı. "
                    "Makalenin PDF'ini yan menüden yükleyerek tam metin kontrolü yapabilirsiniz."
                ),
                is_retracted=paper.is_retracted,
                metadata_warnings=paper.metadata_warnings,
                used_llm_model="Künye Doğrulayıcı",
                decision_source="kural",
                match_score=match_score,
            )

        # Adım 2: Çok Dilli Semantik RAG ile en yakın kanıt paragraflarını bul
        top_chunks = self.retrieve_top_evidence(claim_sentence, evidence_corpus, top_k=4)
        best_chunk = top_chunks[0] if top_chunks else EvidenceChunk(0, evidence_corpus[:500], 0.0)

        # Adım 3: Yerel Ollama LLM + Semantik NLI ile Anlam Çıkarımı Yap
        llm_res = self._evaluate_entailment(claim_sentence, top_chunks, paper)

        verdict = llm_res["verdict"]
        alt_papers = []
        if verdict in ("ÇELİŞİYOR", "KANIT_YETERSİZ") and alternative_finder is not None:
            alt_papers = alternative_finder.suggest_real_papers(claim_sentence, top_k=2)

        sec_note = ""
        if best_chunk.is_secondary_citation:
            sec_note = (
                "ℹ️ İkincil Atıf Uyarısı: Bulunan kanıt paragrafı incelendiğinde, bu bilginin "
                "referans makalenin kendi deneysel bulgusu olmayıp makale içinde başka bir "
                "kaynağa yapılan atıf olabileceği görülmektedir. Birincil kaynağa atıf yapmanız önerilir."
            )

        return VerificationResult(
            citation_id=citation_id,
            marker=marker,
            claim_sentence=claim_sentence,
            context_text=context_text or claim_sentence,
            ref_id=paper.ref_id,
            verdict=verdict,
            confidence_score=llm_res["confidence"],
            semantic_similarity=best_chunk.semantic_score,
            best_evidence=best_chunk.text,
            top_chunks=top_chunks,
            explanation=llm_res["explanation"],
            claim_core_meaning=llm_res.get("claim_meaning", ""),
            evidence_core_meaning=llm_res.get("evidence_meaning", ""),
            suggested_revision=llm_res.get("suggested_revision", ""),
            is_secondary_citation=best_chunk.is_secondary_citation,
            secondary_citation_note=sec_note,
            is_retracted=paper.is_retracted,
            metadata_warnings=paper.metadata_warnings,
            alternative_papers=alt_papers,
            used_llm_model=llm_res["model_used"],
            decision_source=llm_res.get("decision_source", ""),
            match_score=match_score,
        )

    # -----------------------------------------------------------------------
    # 4. Derin Anlam Çıkarımı (Ollama LLM + Kural Tabanlı Semantik Yedek)
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_shared_scientific_terms(claim_text: str, evidence_text: str) -> List[str]:
        """İddia cümlesi ile referans makale arasındaki ortak spesifik bilimsel/teknik terimleri bulur."""
        stopwords = {
            "the", "and", "for", "with", "from", "into", "upon", "between", "among", "through",
            "during", "under", "after", "before", "over", "about", "within", "without", "towards",
            "that", "this", "these", "those", "which", "were", "have", "been", "also", "such",
            "more", "most", "other", "some", "only", "both", "each", "very", "well", "however",
            "study", "studies", "review", "analysis", "evaluation", "investigation", "assessment",
            "effect", "effects", "role", "roles", "involvement", "mechanism", "mechanisms",
            "action", "actions", "model", "models", "induced", "associated", "mediated", "related",
            "acute", "chronic", "mild", "repeated", "novel", "new", "potential", "therapeutic",
            "rats", "rat", "mice", "mouse", "rodents", "rodent", "patients", "human", "humans",
            "animal", "animals", "experimental", "clinical", "treatment", "activity", "expression",
            "levels", "function", "functions", "alterations", "changes", "response", "responses",
            "condition", "persistent", "common", "highly", "prevalent", "accompanying", "various",
            "olan", "olarak", "için", "göre", "daha", "çok", "গibi", "kadar", "sonra", "önce",
            "arasında", "üzerine", "tarafından", "birlikte", "ancak", "veya", "olup", "olduğu",
        }
        c_words = {
            w for w in re.findall(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ\-]{4,}", claim_text.lower())
            if w not in stopwords and not w.isdigit()
        }
        e_words = {
            w for w in re.findall(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ\-]{4,}", evidence_text.lower())
            if w not in stopwords and not w.isdigit()
        }
        shared_set = set(c_words & e_words)
        # Türemiş bilimsel kelimeleri de (depressive <-> depression, glutamatergic <-> glutamate) yakala
        for cw in c_words:
            if len(cw) >= 6:
                stem = cw[:6]
                for ew in e_words:
                    if len(ew) >= 6 and ew.startswith(stem):
                        shared_set.add(cw)
                        break
        shared = sorted(shared_set, key=len, reverse=True)
        return shared

    def _evaluate_entailment(
        self,
        claim_sentence: str,
        top_chunks: List[EvidenceChunk],
        paper: FetchedPaper,
    ) -> Dict[str, Any]:
        """
        Önce kural tabanlı sayısal/anlamsal ön-analiz yapar, ardından yerel Ollama LLM'e
        göndererek 4 dereceli derin anlam çıkarımı (NLI) gerçekleştirir.
        """
        clean_claim = re.sub(r"\[[^\]]+\]|\([^)]*\d{4}[^)]*\)", "", claim_sentence).strip()
        norm_claim = normalize_semantic_text(clean_claim)
        evidence_joined = "\n---\n".join(
            f"[Kanıt Paragrafı {i+1}]: { normalize_semantic_text(c.text) }"
            for i, c in enumerate(top_chunks[:3])
        )

        has_full_text = bool(paper.full_text and len(paper.full_text) >= 1500)
        access_mode_note = (
            "TAM METİN (Makalenin tam metin PDF'ine erişildi)"
            if has_full_text
            else "SADECE KISA ÖZET / BAŞLIK (Yayıncı tam metni kilitli/ücretli olduğu için yalnızca özet mevcut)"
        )

        # Ön-analiz: Sayısal tutarlılık, konu terimleri ve semantik skor kontrolü
        heuristic = self._heuristic_semantic_nli(clean_claim, top_chunks, paper)

        # Ollama üzerinden derin anlam çıkarımı dene
        system_prompt = (
            "Sen TÜBİTAK hakemi düzeyinde uzman bir Akademik Anlam Çıkarımı (Natural Language Inference) "
            "denetçisisin. Görevin, yazarın makalesindeki İDDİA CÜMLESİ ile referans makaleden alınan "
            "KANIT PARAGRAFLARI arasında ANLAM BÜTÜNLÜĞÜ olup olmadığını denetlemektir.\n\n"
            "ALTIN KURALLAR:\n"
            "1. KELİMELERİN BİREBİR AYNI OLMASINA BAKMA! Farklı kelimelerle yeniden ifade edilmiş (paraphrase), "
            "devrik/kurallı yapısı değişmiş veya İngilizce'den Türkçe'ye çevrilmiş cümleler AYNI ANLAMI "
            "taşıyorsa karar kesinlikle 'DESTEKLENİYOR' olmalıdır.\n"
            "   Örnek: İddia='Dünyanın yüzde 90'ı sudan oluşur' ve Kanıt='Yüzde doksanı su olan dünyamız...' "
            "-> İkisi de %90 su oranını ifade eder -> KARAR: 'DESTEKLENİYOR'.\n"
            "2. KİLİTLİ TAM METİN / SADECE ÖZET DURUMU ('KISMEN_DESTEKLENİYOR'): Eğer makalenin sadece "
            "KISA ÖZETİ/BAŞLIĞI mevcutsa (tam metin kilitliyse), yazarın cümlesindeki ana konu/hastalık/mekanizma "
            "makalenin özetiyle doğrudan uyumluysa ve hiçbir çelişki yoksa, spesifik bir alt detay özette "
            "yazmasa bile karar 'KISMEN_DESTEKLENİYOR' (Özet Düzeyinde Uyumlu) olmalıdır.\n"
            "3. SAYISAL VEYA YÖN ÇELİŞKİSİ: İddiada '%90' denirken kanıtta '%70' deniyorsa veya 'artırmıştır' denirken "
            "'azaltmıştır' deniyorsa -> KARAR: 'ÇELİŞİYOR'.\n"
            "4. KANIT YETERSİZ / ALAKASIZ ATIF: Kanıt paragraflarında iddiadaki konuyla ilgili hiçbir bilimsel "
            "bağ yoksa -> KARAR: 'KANIT_YETERSİZ'.\n\n"
            "Yanıtını SADECE aşağıdaki JSON formatında ver:\n"
            "{\n"
            '  "claim_meaning": "Yazarın cümlesinin tek cümlelik öz anlamı",\n'
            '  "evidence_meaning": "Referans makaledeki kanıtın tek cümlelik öz anlamı",\n'
            '  "verdict": "DESTEKLENİYOR veya KISMEN_DESTEKLENİYOR veya ÇELİŞİYOR veya KANIT_YETERSİZ",\n'
            '  "confidence": 0.95,\n'
            '  "explanation": "Neden bu kararın verildiğinin net Türkçe açıklaması",\n'
            '  "suggested_revision": "Eğer ÇELİŞİYOR ise yazarın cümlesini makaledeki gerçek veriye göre düzelten Türkçe akademik cümle (Destekleniyorsa boş bırak)"\n'
            "}"
        )

        user_prompt = (
            f"ERİŞİM DURUMU: {access_mode_note}\n"
            f"YAZARIN İDDİA CÜMLESİ (Normalize): {norm_claim}\n\n"
            f"REFERANS MAKALEDEKİ KANIT PARAGRAFLARI:\n{evidence_joined}\n\n"
            f"Ön-Analiz İpucu: {heuristic['hint']}\n"
            "Lütfen anlam bütünlüğünü incele ve JSON çıktısını üret."
        )

        llm_result: Optional[Dict[str, Any]] = None
        try:
            resp = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.0, "num_ctx": 3072},
            )
            content = (
                getattr(getattr(resp, "message", None), "content", None)
                or resp.get("message", {}).get("content", "")
            )
            parsed = json.loads(content)
            raw_verdict = str(parsed.get("verdict", "")).upper().strip()

            if "KISMEN" in raw_verdict or "PARTIAL" in raw_verdict:
                verdict = "KISMEN_DESTEKLENİYOR"
            elif "DESTEK" in raw_verdict or "SUPPORT" in raw_verdict:
                verdict = "DESTEKLENİYOR"
            elif "ÇELİŞ" in raw_verdict or "CELIS" in raw_verdict or "CONTRADICT" in raw_verdict:
                verdict = "ÇELİŞİYOR"
            else:
                verdict = "KANIT_YETERSİZ"

            explanation = parsed.get("explanation") or heuristic["explanation"]
            if verdict == "KISMEN_DESTEKLENİYOR" and "kilitli" not in explanation.lower() and "özet" not in explanation.lower():
                explanation = f"{heuristic['explanation']} (Hakem Notu: {explanation})"

            llm_result = {
                "verdict": verdict,
                "confidence": float(parsed.get("confidence", heuristic["confidence"])),
                "explanation": explanation,
                "claim_meaning": parsed.get("claim_meaning") or heuristic["claim_meaning"],
                "evidence_meaning": parsed.get("evidence_meaning") or heuristic["evidence_meaning"],
                "suggested_revision": parsed.get("suggested_revision") or heuristic["suggested_revision"],
                "model_used": f"Ollama ({self.ollama_model}) + Çok Dilli Semantik RAG",
            }
        except Exception:
            # Ollama kapalıysa veya model henüz indirilmemişse bile sistem asla çökmez;
            # Çok dilli vektör + sayısal/kavramsal NLI motoru devreye girer!
            llm_result = None

        return self._combine_decisions(heuristic, llm_result)

    @staticmethod
    def _match_confidence(paper: FetchedPaper) -> Tuple[bool, float, str]:
        """Künye eşleşmesinin güvenilir olup olmadığını belirler.

        - DOI birebir doğrulandıysa güvenilir (başlık skoru düşük olsa bile).
        - Aksi hâlde başlık benzerliği MIN_TRUSTED_MATCH üstünde olmalıdır.
        """
        doi = (paper.verified_doi or "").strip().lower()
        query_doi = (paper.query_doi or "").strip().lower()
        if doi and query_doi and doi == query_doi:
            return True, 1.0, "DOI birebir doğrulandı"

        score = float(paper.title_match_score or 0.0)
        if score >= MIN_TRUSTED_MATCH:
            return True, score, "başlık eşleşmesi güçlü"
        if score <= 0.0:
            return False, 0.0, "eşleşme skoru bilinmiyor"
        return False, score, "başlık eşleşmesi zayıf"

    @staticmethod
    def _classify_missing_source(paper: FetchedPaper) -> Tuple[str, str]:
        """Kaynağı doğrulanamayan kayıtları 'sahte' ile 'doğrulanamadı' arasında ayırır.

        Kural: "bulunamadı" tek başına sahtelik kanıtı değildir. Sahte damgası
        yalnızca somut bir uyumsuzluk (ör. DOI'nin başka makaleye ait olması)
        varsa vurulur; aksi hâlde DOGRULANAMADI döner.
        """
        if paper.lookup_status == "error":
            detail = f" ({paper.lookup_error})" if paper.lookup_error else ""
            return (
                "DOGRULANAMADI",
                "Kaynak doğrulanamadı: İnternet araması sırasında ağ/API hatası oluştu"
                f"{detail}. Bu bir kaynak hatası değil, geçici bir erişim sorunudur; "
                "bağlantı düzelince tekrar deneyin.",
            )

        warnings_text = " ".join(paper.metadata_warnings or [])
        if "Başlık–DOI Uyuşmazlığı" in warnings_text:
            return (
                "SAHTE_KAYNAK",
                "Kaynakçadaki DOI numarası başka bir makaleye aittir (başlık ile DOI uyuşmuyor). "
                "Bu, künyenin hatalı olduğunu gösterir; DOI ve başlığı düzeltin.",
            )

        checked = ", ".join(dict.fromkeys(paper.sources_checked)) or "hiçbir veritabanı"
        return (
            "DOGRULANAMADI",
            f"Kaynak {checked} üzerinde arandı ancak bulunamadı. Bu, kaynağın sahte olduğu "
            "anlamına GELMEZ: DOI'siz dergi makaleleri, kitaplar, tezler ve henüz indekslenmemiş "
            "yayınlar da bu durumda olabilir. Künyeyi elle kontrol edin veya PDF'ini yükleyin.",
        )

    @staticmethod
    def _combine_decisions(
        heuristic: Dict[str, Any], llm_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Kural katmanı ile LLM kararını tek bir nihai karara birleştirir.

        Politika (ADIM 16):
        - LLM kararı esastır: negasyon ve çıkarım gibi anlamsal durumlarda LLM,
          kelime/vektör benzerliğinden daha güvenilirdir. Kural katmanı artık
          "DESTEKLENİYOR" kararını ASLA dayatmaz.
        - Kural katmanı yalnızca NESNEL çelişkilerde (sayı, yön, olumsuzluk)
          LLM'i ezebilir; bu durumda bile yalnızca "ÇELİŞİYOR" yönünde.
        - LLM yoksa/başarısızsa kural kararı döner (çevrimdışı çalışma korunur).
        """
        if llm_result is None:
            out = dict(heuristic)
            out["decision_source"] = "kural"
            return out

        hv = heuristic.get("verdict")
        lv = llm_result.get("verdict")
        objective = bool(heuristic.get("objective"))
        source = "llm"

        if objective and hv == "ÇELİŞİYOR":
            source = "kural+llm" if lv == "ÇELİŞİYOR" else "kural"
            if lv != "ÇELİŞİYOR":
                llm_result = dict(llm_result)
                llm_result["verdict"] = "ÇELİŞİYOR"
                llm_result["explanation"] = (
                    f"{heuristic.get('explanation', '')} "
                    f"(LLM notu: {llm_result.get('explanation', '')})"
                ).strip()
                llm_result["confidence"] = max(
                    float(heuristic.get("confidence", 0.8)),
                    float(llm_result.get("confidence", 0.8)),
                )
                llm_result["suggested_revision"] = (
                    llm_result.get("suggested_revision") or heuristic.get("suggested_revision", "")
                )
        elif lv == "KANIT_YETERSİZ" and heuristic.get("signal") == "numeric_match":
            # Küçük modelin "kanıt yetersiz" ısrarını yalnızca nesnel sayı eşleşmesi kırabilir.
            llm_result = dict(llm_result)
            llm_result["verdict"] = "DESTEKLENİYOR"
            llm_result["explanation"] = heuristic.get("explanation", "")
            llm_result["confidence"] = max(0.90, float(llm_result.get("confidence", 0.0)))
            source = "kural+llm"

        out = dict(llm_result)
        out["decision_source"] = source
        for key in ("claim_meaning", "evidence_meaning", "suggested_revision"):
            if not out.get(key):
                out[key] = heuristic.get(key, "")
        return out

    def _heuristic_semantic_nli(
        self, clean_claim: str, top_chunks: List[EvidenceChunk], paper: Optional[FetchedPaper] = None
    ) -> Dict[str, Any]:
        """
        Çok dilli semantik vektör benzerliği, sayısal/yüzdelik eşdeğerlik (%90 == yüzde doksan),
        spesifik bilimsel terim örtüşmesi ve zıtlık/negasyon analizi yapan deterministik NLI katmanı.
        """
        if not top_chunks:
            return {
                "verdict": "KANIT_YETERSİZ",
                "confidence": 0.60,
                "explanation": "Referans makalede bu iddiayla eşleşen bir paragraf bulunamadı.",
                "claim_meaning": clean_claim,
                "evidence_meaning": "Kanıt bulunamadı.",
                "suggested_revision": "",
                "hint": "Kanıt bulunamadı.",
                "strong_override": False,
                "signal": "no_evidence",
                "objective": False,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }

        best = top_chunks[0]
        sim = best.semantic_score
        has_full_text = bool(paper and paper.full_text and len(paper.full_text) >= 1500)
        shared_terms = self._extract_shared_scientific_terms(clean_claim, best.text)

        claim_pcts, claim_nums = extract_numbers_and_percentages(clean_claim)
        ev_pcts, ev_nums = extract_numbers_and_percentages(best.text)

        norm_claim = normalize_semantic_text(clean_claim)
        norm_ev = normalize_semantic_text(best.text)

        # 0. Olumsuzluk (negasyon) çelişkisi: "azaltır" <-> "azaltmaz"
        # Yalnızca yeterince güçlü anlamsal örtüşme ve en az bir ortak bilimsel
        # terim varsa devreye girer; kural yalnızca ÇELİŞKİ yönünde sinyal üretir.
        claim_neg = has_negation(clean_claim)
        ev_neg = has_negation(best.text)
        if claim_neg != ev_neg and sim >= 0.45 and len(shared_terms) >= 1:
            return {
                "verdict": "ÇELİŞİYOR",
                "confidence": 0.86,
                "explanation": (
                    "Olumsuzluk (negasyon) çelişkisi yakalandı: İddia ile referans makaledeki "
                    "kanıt paragrafı aynı konuyu ele alıyor, ancak biri olumlu diğeri olumsuz "
                    "ifade ediyor (örn. 'azaltır' ↔ 'azaltmaz')."
                ),
                "claim_meaning": norm_claim,
                "evidence_meaning": norm_ev[:240],
                "suggested_revision": "Lütfen cümledeki olumluluk/olumsuzluk yönünü referans makaledeki kanıta göre düzeltin.",
                "hint": "Olumsuzluk farkı var (iddia ile kanıt zıt kutuplu) -> ÇELİŞİYOR.",
                "strong_override": True,
                "signal": "negation_conflict",
                "objective": True,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }

        # 1. Yüzdelik / Sayısal Karşılaştırma
        if claim_pcts and ev_pcts:
            # Aynı yüzdelik oran var mı? (Örn: %90 == yüzde doksan (%90))
            def _pct_close(a: float, b: float) -> bool:
                """Göreli tolerans: %3.2 ile %2.8 gibi farklı değerler eşleşmesin."""
                return abs(a - b) <= max(0.05, 0.01 * max(abs(a), abs(b)))

            matched_pcts = [
                cp for cp in claim_pcts if any(_pct_close(cp, ep) for ep in ev_pcts)
            ]
            if matched_pcts and sim >= 0.55:
                return {
                    "verdict": "DESTEKLENİYOR",
                    "confidence": round(max(0.90, sim), 2),
                    "explanation": (
                        f"İddia cümlesindeki anlam ve sayısal oran (%{matched_pcts[0]:g}), "
                        f"referans makaledeki kanıt paragrafıyla ({sim:.0%} semantik benzerlik ile) "
                        "doğrudan örtüşmektedir. Kelime dizilimi farklı olsa da aynı anlam çıkmaktadır."
                    ),
                    "claim_meaning": norm_claim,
                    "evidence_meaning": norm_ev[:240],
                    "suggested_revision": "",
                    "hint": f"Sayısal oran (%{matched_pcts[0]:g}) ve anlam birebir eşleşiyor -> DESTEKLENİYOR.",
                    "strong_override": False,
                    "signal": "numeric_match",
                    "objective": False,
                    "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
                }
            elif not matched_pcts and sim >= 0.45:
                # Konu aynı ama yüzdeler farklı! (ÇELİŞKİ)
                revised = re.sub(
                    r"(?:%\s*\d+(?:[.,]\d+)?|\byüzde\s+[a-zçğıöşü0-9.,']+(?:\s+[a-zçğıöşü]+)?)",
                    f"%{ev_pcts[0]:g}",
                    clean_claim,
                    count=1,
                    flags=re.IGNORECASE,
                )
                return {
                    "verdict": "ÇELİŞİYOR",
                    "confidence": 0.92,
                    "explanation": (
                        f"Sayısal Çelişki Yakalandı! Sizin cümlenizde '%{claim_pcts[0]:g}' oranı "
                        f"belirtilmiş ancak referans makalede bu değer '%{ev_pcts[0]:g}' olarak geçmektedir."
                    ),
                    "claim_meaning": norm_claim,
                    "evidence_meaning": norm_ev[:240],
                    "suggested_revision": revised,
                    "hint": f"Sayısal çelişki var: İddia %{claim_pcts[0]:g} diyor ama kanıt %{ev_pcts[0]:g} diyor -> ÇELİŞİYOR.",
                    "strong_override": True,
                    "signal": "numeric_conflict",
                    "objective": True,
                    "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
                }

        # 2. Zıt yön / Negasyon kontrolü (artırmıştır vs azaltmıştır)
        increase_words = r"\b(artır|yükselt|hızlandır|iyileştir|increas|improv|enhanc|higher|faster)\w*"
        decrease_words = r"\b(azalt|düşür|yavaşlat|kötüleştir|decreas|reduc|lower|slower|worsen)\w*"
        claim_inc = bool(re.search(increase_words, clean_claim, re.IGNORECASE))
        claim_dec = bool(re.search(decrease_words, clean_claim, re.IGNORECASE))
        ev_inc = bool(re.search(increase_words, best.text, re.IGNORECASE))
        ev_dec = bool(re.search(decrease_words, best.text, re.IGNORECASE))

        if sim >= 0.45 and ((claim_inc and ev_dec and not ev_inc) or (claim_dec and ev_inc and not ev_dec)):
            return {
                "verdict": "ÇELİŞİYOR",
                "confidence": 0.88,
                "explanation": (
                    "Yön/Etki Çelişkisi Yakalandı! İddia cümlesindeki etki yönü (artış/azalış) ile "
                    "referans makaledeki bulgu birbirinin tam tersidir."
                ),
                "claim_meaning": norm_claim,
                "evidence_meaning": norm_ev[:240],
                "suggested_revision": "Lütfen cümledeki artış/azalış yönünü referans makaledeki kanıt paragrafına göre güncelleyin.",
                "hint": "Etki yönü zıt (artış vs azalış) -> ÇELİŞİYOR.",
                "strong_override": True,
                "signal": "direction_conflict",
                "objective": True,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }

        # 3. Genel Semantik Yakınlık ve Özet/Tam Metin Kararı
        if sim >= 0.64 or (sim >= 0.56 and len(shared_terms) >= 2):
            return {
                "verdict": "DESTEKLENİYOR",
                "confidence": round(max(0.85, sim), 2),
                "explanation": (
                    f"Yazarın cümlesi ile referans makaledeki kanıt paragrafı arasında güçlü düzeyde "
                    f"({sim:.0%}) çok dilli anlamsal ve kavramsal örtüşme ({', '.join(shared_terms[:4]) or 'doğrudan anlam'}) "
                    "tespit edildi. İddia makaleden çıkarılabilir."
                ),
                "claim_meaning": norm_claim,
                "evidence_meaning": norm_ev[:240],
                "suggested_revision": "",
                "hint": f"Yüksek semantik ve kavramsal benzerlik ({sim:.2f}, ortak terimler: {', '.join(shared_terms[:4])}) -> DESTEKLENİYOR.",
                "strong_override": False,
                "signal": "semantic_support",
                "objective": False,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }
        elif not has_full_text and (
            (sim >= 0.46 and len(shared_terms) >= 1) or sim >= 0.52
        ):
            terms_str = ", ".join(shared_terms[:5]) if shared_terms else "ana konu"
            return {
                "verdict": "KISMEN_DESTEKLENİYOR",
                "confidence": round(max(0.76, sim), 2),
                "explanation": (
                    f"☑️ Özet Düzeyinde Uyumlu (Tam Metin Kilitli): Kaynak makale %100 gerçektir ve "
                    f"cümlenizdeki bilimsel kavramlar ({terms_str}) makalenin özeti/başlığı ile "
                    f"({sim:.0%} anlam yakınlığıyla) örtüşmektedir. Yayıncı tam metin PDF'i ücretli/kilitli "
                    "olduğu için spesifik alt detaylar özet düzeyinde doğrulanmıştır."
                ),
                "claim_meaning": norm_claim,
                "evidence_meaning": norm_ev[:240],
                "suggested_revision": "",
                "hint": (
                    f"Makale gerçek ancak sadece kısa özet açık ({sim:.2f} yakınlık, ortak kavramlar: {terms_str}) "
                    "-> KISMEN_DESTEKLENİYOR (Özet Düzeyinde Uyumlu)."
                ),
                "strong_override": False,
                "signal": "partial",
                "objective": False,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }
        elif sim >= 0.48:
            return {
                "verdict": "KANIT_YETERSİZ",
                "confidence": round(sim, 2),
                "explanation": (
                    f"Referans makalede benzer kavramlar geçiyor ({sim:.0%} konu yakınlığı), ancak "
                    "sizin cümlenizdeki spesifik iddiayı doğrudan doğrulayan yeterli kanıt bulunamadı."
                ),
                "claim_meaning": norm_claim,
                "evidence_meaning": norm_ev[:240],
                "suggested_revision": "",
                "hint": f"Orta düzey benzerlik ({sim:.2f}), spesifik iddia tam doğrulanamadı -> KANIT_YETERSİZ.",
                "strong_override": False,
                "signal": "insufficient",
                "objective": False,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }
        else:
            return {
                "verdict": "KANIT_YETERSİZ",
                "confidence": 0.80,
                "explanation": (
                    f"Referans makale gerçek olsa da, metni/özeti içinde bu iddiayla anlamca ilişkili "
                    f"yeterli bir bulguya rastlanmadı (Maksimum anlam yakınlığı: {sim:.0%}). Yanlış kaynağa "
                    "atıf yapılmış olabilir."
                ),
                "claim_meaning": norm_claim,
                "evidence_meaning": norm_ev[:240],
                "suggested_revision": "",
                "hint": f"Düşük anlam benzerliği ({sim:.2f}) -> KANIT_YETERSİZ.",
                "strong_override": False,
                "signal": "no_match",
                "objective": False,
                "model_used": "Çok Dilli Semantik NLI Motoru (CPU)",
            }


# ---------------------------------------------------------------------------
# Hızlı Modül Testi: Kullanıcının "Dünyanın yüzde 90'ı sudan oluşur" Örneği!
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== ADIM 4: SEMANTİK ANLAM ÇIKARIMI (rag_verifier.py) TESTİ ===")
    verifier = SemanticRAGVerifier(embedding_device="cpu")

    # Senaryo 1: Kullanıcının verdiği örnek!
    # Cümleler farklı kelimelerle yazılmış ama anlamları %100 aynı:
    # İddia: "Dünyanın yüzde 90'nı sudan oluşur [1]."
    # Referans Makale: "Yüzde doksanı su olan dünyamız biyosfer açısından eşsizdir."
    test_paper_1 = FetchedPaper(
        ref_id="1",
        query_title="Gezegen Hidrolojisi ve Su Oranları",
        query_doi="10.1000/test.2022",
        exists_online=True,
        verified_title="Gezegen Hidrolojisi ve Su Oranları",
        verified_authors=["A. Yılmaz"],
        verified_year="2022",
        abstract=(
            "Bu çalışmada gezegenlerin atmosferik ve yüzey bileşenleri incelenmiştir. "
            "Yüzde doksanı su olan dünyamız, sıvı hidrosfer tabakası sayesinde iklim dengesini korur. "
            "Diğer karasal gezegenlerde bu oran oldukça düşüktür."
        ),
    )

    claim_1 = "Dünyanın yüzde 90'nı sudan oluşur [1]."
    res_1 = verifier.verify_claim_against_paper(claim_1, test_paper_1, citation_id="TEST_1", marker="[1]")

    print(f"\n[TEST 1 - EŞDEĞER ANLAM (PARAPHRASE) TESTİ]")
    print(f"  Senin Cümlen     : {claim_1}")
    print(f"  Bulunan Kanıt    : {res_1.best_evidence}")
    print(f"  Anlam Yakınlığı  : %{res_1.semantic_similarity * 100:.1f}")
    print(f"  KARAR            : {res_1.verdict} (Güven: %{res_1.confidence_score * 100:.0f})")
    print(f"  Açıklama         : {res_1.explanation}")

    # Senaryo 2: Sayısal Çelişki ve İngilizce-Türkçe Çapraz Dil Testi!
    # İddia (Türkçe): "Derin öğrenme modeli akciğer kanseri tespitinde %95 doğruluk sağlamıştır [2]."
    # Referans Makale (İngilizce): "...our deep learning model achieved an accuracy of 82% in lung cancer detection..."
    test_paper_2 = FetchedPaper(
        ref_id="2",
        query_title="Deep Learning in Lung Cancer Detection",
        query_doi="10.1109/test.2021",
        exists_online=True,
        verified_title="Deep Learning in Lung Cancer Detection",
        verified_authors=["J. Smith"],
        verified_year="2021",
        abstract=(
            "We evaluated several neural architectures on chest CT scans. "
            "Experimental results show that our deep learning model achieved an accuracy of 82% "
            "in lung cancer detection across multi-center clinical datasets."
        ),
    )

    claim_2 = "Derin öğrenme modeli akciğer kanseri tespitinde %95 doğruluk sağlamıştır [2]."
    res_2 = verifier.verify_claim_against_paper(claim_2, test_paper_2, citation_id="TEST_2", marker="[2]")

    print(f"\n[TEST 2 - ÇAPRAZ DİL (TR <-> EN) & SAYISAL ÇELİŞKİ TESTİ]")
    print(f"  Senin Cümlen (TR): {claim_2}")
    print(f"  Bulunan Kanıt(EN): {res_2.best_evidence}")
    print(f"  Anlam Yakınlığı  : %{res_2.semantic_similarity * 100:.1f}")
    print(f"  KARAR            : {res_2.verdict} (Güven: %{res_2.confidence_score * 100:.0f})")
    print(f"  Açıklama         : {res_2.explanation}")
    print(f"  Önerilen Düzeltme: {res_2.suggested_revision}")
