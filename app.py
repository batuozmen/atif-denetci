r"""
MODÜL 4: Görsel TÜBİTAK Akademik Makale & Semantik Atıf Denetim Arayüzü (app.py)
--------------------------------------------------------------------------------
Çalıştırmak için:
    .\.venv\Scripts\streamlit.exe run app.py
veya klasördeki `baslat.bat` dosyasına çift tıklayın.
"""

from __future__ import annotations

import importlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import streamlit as st

import document_parser
import academic_fetcher
import rag_verifier

importlib.reload(document_parser)
importlib.reload(academic_fetcher)
importlib.reload(rag_verifier)

from document_parser import DocumentParser, ParsedManuscript, ReferenceItem, InTextCitation
from academic_fetcher import AcademicFetcher, FetchedPaper, AlternativePaper, safe_ref_token
from rag_verifier import SemanticRAGVerifier, VerificationResult, EvidenceChunk


# ---------------------------------------------------------------------------
# Sayfa Yapılandırması ve Şık Mikro-Stil Dokunuşları
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TÜBİTAK Akademik Atıf ve Kaynakça Denetçisi",
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Göz yormayan, orantılı kartlar ve dengeli butonlar için hafif stil iyileştirmesi
st.html(
    """
    <style>
    /* Ana içerik alanını çok geniş ekranlarda aşırı yayılmaktan koru ve ortala */
    .block-container {
        max-width: 1340px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
    }
    /* Sekme başlıklarını daha modern ve okunaklı yap */
    button[data-baseweb="tab"] {
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.15rem !important;
    }
    /* Metrik kutularının başlık ve değer orantısı */
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        opacity: 0.88;
    }
    /* Expander başlıklarını daha temiz ve ferah göster */
    details summary p {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Önbellekli Motor Nesneleri
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Anlam kontrol motoru hazırlanıyor...")
def get_verifier(ollama_model: str) -> SemanticRAGVerifier:
    return SemanticRAGVerifier(ollama_model=ollama_model, embedding_device="cpu")


def get_parser() -> DocumentParser:
    return DocumentParser()


USER_PDF_DIR = Path(".cache/user_pdfs")
USER_PDF_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS_DIR = Path("raporlar/oturumlar")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Oturum Kaydetme, Geri Yükleme ve Kaldığı Yerden Devam Etme Yardımcıları
# ---------------------------------------------------------------------------

def _session_file_for_doc(filename: str) -> Path:
    stem = safe_ref_token(Path(filename).stem, fallback="makale")
    return SESSIONS_DIR / f"{stem}.analiz.json"


def merge_verification_results(
    all_citations: List[InTextCitation],
    existing_results: List[VerificationResult],
    new_results: List[VerificationResult],
) -> List[VerificationResult]:
    """Yeni doğrulanan atıfları önceki sonuçların üzerine yazar ve makale sırasını korur."""
    by_id: Dict[str, VerificationResult] = {r.citation_id: r for r in existing_results}
    for nr in new_results:
        by_id[nr.citation_id] = nr

    ordered: List[VerificationResult] = []
    seen: set[str] = set()
    for cit in all_citations:
        if cit.citation_id in by_id:
            ordered.append(by_id[cit.citation_id])
            seen.add(cit.citation_id)
    for cid, res in by_id.items():
        if cid not in seen:
            ordered.append(res)
    return ordered


def serialize_session_bytes(
    parsed_doc: ParsedManuscript,
    fetched_map: Dict[str, FetchedPaper],
    verification_results: List[VerificationResult],
) -> bytes:
    """Tüm makale, çekilen kaynaklar ve doğrulama sonuçlarını tek bir JSON paketine dönüştürür."""
    payload = {
        "format_version": "1.0",
        "saved_at": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "filename": parsed_doc.filename,
        "verified_count": len(verification_results),
        "total_citations": len(parsed_doc.citations),
        "parsed_doc": asdict(parsed_doc),
        "fetched_map": {k: asdict(v) for k, v in fetched_map.items()},
        "verification_results": [asdict(r) for r in verification_results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def save_session_to_disk(
    parsed_doc: ParsedManuscript,
    fetched_map: Dict[str, FetchedPaper],
    verification_results: List[VerificationResult],
) -> Optional[Path]:
    """Analiz oturumunu raporlar/oturumlar/ klasörüne otomatik kaydeder."""
    try:
        target = _session_file_for_doc(parsed_doc.filename)
        target.write_bytes(serialize_session_bytes(parsed_doc, fetched_map, verification_results))
        return target
    except Exception:
        return None


def deserialize_session_bytes(
    raw_bytes: bytes,
) -> tuple[ParsedManuscript, Dict[str, FetchedPaper], List[VerificationResult], str]:
    """JSON analiz paketinden tüm veri nesnelerini eksiksiz geri oluşturur."""
    data = json.loads(raw_bytes.decode("utf-8"))
    saved_at = str(data.get("saved_at", "-"))
    p_raw = data["parsed_doc"]

    refs = [ReferenceItem(**r) for r in p_raw.get("references", [])]
    ref_by_id = {r.ref_id: r for r in refs}

    def _rebuild_cit(c_dict: Dict[str, Any]) -> InTextCitation:
        c_copy = dict(c_dict)
        m_id = c_copy.get("matched_ref_id")
        c_copy["matched_reference"] = ref_by_id.get(m_id) if m_id else None
        return InTextCitation(**c_copy)

    cits = [_rebuild_cit(c) for c in p_raw.get("citations", [])]
    ghosts = [_rebuild_cit(g) for g in p_raw.get("ghost_citations", [])]
    orphans = [ReferenceItem(**o) for o in p_raw.get("orphan_references", [])]

    parsed_doc = ParsedManuscript(
        filename=p_raw.get("filename", data.get("filename", "makale.docx")),
        total_chars=int(p_raw.get("total_chars", 0)),
        body_text=p_raw.get("body_text", ""),
        references_text=p_raw.get("references_text", ""),
        references=refs,
        citations=cits,
        ghost_citations=ghosts,
        orphan_references=orphans,
        warnings=list(p_raw.get("warnings", [])),
    )

    fetched_map: Dict[str, FetchedPaper] = {}
    for k, fp_dict in data.get("fetched_map", {}).items():
        fp_copy = dict(fp_dict)
        fp_copy.pop("has_full_text", None)
        fp_copy.pop("has_abstract", None)
        fp_copy.pop("full_text_length", None)
        fetched_map[k] = FetchedPaper(**fp_copy)

    results: List[VerificationResult] = []
    for r_dict in data.get("verification_results", []):
        r_copy = dict(r_dict)
        r_copy["top_chunks"] = [EvidenceChunk(**ch) for ch in r_copy.get("top_chunks", [])]
        r_copy["alternative_papers"] = [
            AlternativePaper(**ap) for ap in r_copy.get("alternative_papers", [])
        ]
        results.append(VerificationResult(**r_copy))

    return parsed_doc, fetched_map, results, saved_at


def activate_loaded_session(
    parsed_doc: ParsedManuscript,
    fetched_map: Dict[str, FetchedPaper],
    verification_results: List[VerificationResult],
    notice_msg: str,
) -> None:
    """Geri yüklenen oturumu aktif session_state içine yerleştirir."""
    st.session_state["restored_parsed_doc"] = parsed_doc
    st.session_state["current_parsed_doc"] = parsed_doc
    st.session_state["active_doc_name"] = parsed_doc.filename
    st.session_state["fetched_map"] = fetched_map
    st.session_state["verification_results"] = verification_results
    st.session_state["selected_cit_indices"] = set(range(1, min(6, len(parsed_doc.citations) + 1)))
    st.session_state["editor_ver"] = st.session_state.get("editor_ver", 0) + 1
    st.session_state["session_restore_notice"] = notice_msg
    for prefix in ("f1", "f3"):
        st.session_state[f"{prefix}_ALL"] = True
        for k in ALL_FILTER_KEYS:
            st.session_state[f"{prefix}_{k}"] = True


# ---------------------------------------------------------------------------
# Anlaşılır Terim ve Karar Açıklama Yardımcıları
# ---------------------------------------------------------------------------

def human_decision_source(source_code: str) -> str:
    """Teknik karar kaynağı kodunu anlaşılır Türkçeye çevirir."""
    mapping = {
        "llm": "Yapay Zekâ Hakemi",
        "kural": "Sayısal / Dilbilgisel Kural Kontrolü",
        "kural+llm": "Yapay Zekâ + Sayısal Kural Kontrolü",
    }
    return mapping.get(source_code or "", "Otomatik Kontrol")


ALL_FILTER_KEYS = ["SUP", "PART", "CON", "INS", "FAKE", "UNK", "GHOST", "RET"]


def _result_category_code(r_obj: VerificationResult) -> str:
    """Her bir doğrulama sonucunu 8 sayaç kategorisinden birine eşler."""
    if r_obj.is_retracted:
        return "RET"
    code_map = {
        "DESTEKLENİYOR": "SUP",
        "KISMEN_DESTEKLENİYOR": "PART",
        "ÇELİŞİYOR": "CON",
        "KANIT_YETERSİZ": "INS",
        "SAHTE_KAYNAK": "FAKE",
        "DOGRULANAMADI": "UNK",
        "HAYALET_ATIF": "GHOST",
    }
    return code_map.get(r_obj.verdict, "UNK")


def _init_filter_state(prefix: str) -> None:
    if f"{prefix}_ALL" not in st.session_state:
        st.session_state[f"{prefix}_ALL"] = True
        for k in ALL_FILTER_KEYS:
            st.session_state[f"{prefix}_{k}"] = True


def _on_toggle_all(prefix: str) -> None:
    val = bool(st.session_state.get(f"{prefix}_ALL", True))
    for k in ALL_FILTER_KEYS:
        st.session_state[f"{prefix}_{k}"] = val


def _on_toggle_single(prefix: str) -> None:
    st.session_state[f"{prefix}_ALL"] = all(
        bool(st.session_state.get(f"{prefix}_{k}", False)) for k in ALL_FILTER_KEYS
    )


def get_verdict_info(verdict: str, is_retracted: bool = False) -> Dict[str, str]:
    """
    Her bir teknik karar kodunu:
    - Hangi ana gruba girdiğine ('error', 'manual', 'ok'),
    - Anlaşılır başlığa,
    - Neden bu kararın verildiğine ve kullanıcının ne yapması gerektiğine çevirir.
    """
    if is_retracted:
        return {
            "group": "error",
            "badge": "🛑 Düzeltilmeli — Yayından Geri Çekilmiş Makale",
            "short_label": "Yayından Geri Çekilmiş (Retracted)",
            "why": "Atıf yaptığınız bu makale, bilimsel hata veya etik ihlal nedeniyle dergi tarafından resmi olarak geri çekilmiştir.",
            "action": "Bu kaynağı makalenizden çıkarıp yerine geçerli ve güncel bir makale eklemelisiniz.",
        }

    info_map = {
        "DESTEKLENİYOR": {
            "group": "ok",
            "badge": "✅ Sorunsuz — Kaynak Makaleyle Tam Uyumlu",
            "short_label": "Tam Uyumlu (Doğrulandı)",
            "why": "Makalenizde yazdığınız cümledeki bilgi, referans gösterdiğiniz makalenin içeriğiyle doğrudan örtüşüyor.",
            "action": "Herhangi bir düzeltme yapmanıza gerek yoktur.",
        },
        "KISMEN_DESTEKLENİYOR": {
            "group": "ok",
            "badge": "☑️ Sorunsuz — Makale Özetiyle Uyumlu (Tam Metin Kilitli)",
            "short_label": "Özetle Uyumlu (Tam Metin Ücretli/Kilitli)",
            "why": "Kaynak makale internette doğrulandı ve özeti (abstract) sizin cümlenizin konusuyla uyumlu. Ancak yayıncı tam metin PDF'ini kilitli tuttuğu için kontrol özet üzerinden yapıldı.",
            "action": "Genel olarak uyumludur. Cümlenizde çok özel bir sayı/tablo detayı varsa ve makalenin PDF'i bilgisayarınızda mevcutsa sol menüden yükleyerek tam metin kontrolü de yaptırabilirsiniz.",
        },
        "ÇELİŞİYOR": {
            "group": "error",
            "badge": "❌ Düzeltilmeli — Kaynak Makaleyle Çelişiyor",
            "short_label": "Kaynak Makaleyle Çelişen Bilgi",
            "why": "Cümlenizdeki bilgi (sayısal oran, artış/azalış yönü veya olumlu/olumsuz ifade) kaynak makaledeki bulgunun tam tersini söylüyor.",
            "action": "Cümlenizi aşağıdaki 'Önerilen Düzeltilmiş Cümle'ye göre güncelleyin veya yanlış kaynağa atıf yaptıysanız doğru kaynağı gösterin.",
        },
        "HAYALET_ATIF": {
            "group": "error",
            "badge": "🚨 Düzeltilmeli — Kaynakça Listesinde Eksik",
            "short_label": "Kaynakça Listesinde Eksik Atıf",
            "why": "Makale metni içinde bu kaynağa atıf yapmışsınız ancak makalenin sonundaki 'Kaynakça / References' listesine bu makaleyi yazmamışsınız.",
            "action": "Bu kaynağın tam künyesini makalenizin sonundaki Kaynakça listesine ekleyin veya metin içindeki atıfı silin.",
        },
        "SAHTE_KAYNAK": {
            "group": "error",
            "badge": "🚨 Düzeltilmeli — Hatalı / Uyuşmayan Kaynak Künyesi",
            "short_label": "Hatalı Künye (DOI Başka Makaleye Ait)",
            "why": "Kaynakçada yazdığınız DOI numarası veya künye bilgisi internette bambaşka bir makaleye ait görünüyor (kopyala-yapıştır hatası veya uydurma kaynak olabilir).",
            "action": "Kaynakçadaki makale başlığını ve DOI numarasını kontrol edip gerçek makale künyesiyle düzeltin.",
        },
        "KANIT_YETERSİZ": {
            "group": "manual",
            "badge": "⚠️ Kontrol Edilmeli — Makalede Bu Bilgiye Rastlanmadı",
            "short_label": "Makalede Bu Bilgi Bulunamadı",
            "why": "Kaynak makale internette bulundu ancak erişilebilen metninde sizin cümlenizdeki iddiayı doğrulayan veya çürüten bir ifade yer almıyor (bilgi kilitli tam metin sayfalarında veya tablolarda geçiyor olabilir).",
            "action": "Atıfı doğru makaleye yaptığınızdan emin olmak için ilgili kaynağı gözle kontrol edin. Makalenin tam PDF'i elinizdeyse sol menüden yükleyip tekrar denetleyebilirsiniz.",
        },
        "DOGRULANAMADI": {
            "group": "manual",
            "badge": "❓ Kontrol Edilmeli — Makaleye İnternette Erişilemedi",
            "short_label": "İnternette Bulunamadı / Erişilemedi",
            "why": "Bu kaynak uluslararası akademik veritabanlarında bulunamadı, başlık benzerliği düşük kaldı veya anlık internet bağlantı hatası oluştu. (Basılı kitap, tez veya yerel dergi olabilir; sahte olduğu anlamına gelmez).",
            "action": "Kaynakçadaki yazımını (yazar, yıl, başlık) gözle kontrol edin. Eğer PDF dosyası bilgisayarınızda varsa sol menüden yükleyebilirsiniz.",
        },
    }
    return info_map.get(
        verdict,
        {
            "group": "manual",
            "badge": f"ℹ️ {verdict}",
            "short_label": verdict,
            "why": "Değerlendirme tamamlandı.",
            "action": "Sonucu inceleyin.",
        },
    )


def render_reading_guide() -> None:
    """Kullanıcının ekrandaki tüm terimleri ve sayıları tek bakışta anlamasını sağlayan açık rehber."""
    with st.expander(":material/help_outline: Ekrandaki terimler ve yüzdeler ne anlama geliyor?", expanded=False):
        st.markdown(
            """
- **🟢 Sorunsuz / Doğrulanan Atıflar:** Kaynak makale internette bulundu ve cümlenizdeki bilgi makaleyle uyumlu. Değişiklik yapmanıza gerek yoktur.
- **🔴 Düzeltilmesi Gereken Hatalar:** Cümleniz kaynak makaleyle **çelişiyor**, atıf yaptığınız kaynak **Kaynakça listesinde eksik**, **DOI numarası başka makaleye ait** veya makale **yayından geri çekilmiş**.
- **🟡 Sizin Kontrol Etmeniz Gerekenler:** Sistem makaleye internette ulaşamadı (kitap/tez olabilir) veya makalenin açık özetinde cümlenizdeki spesifik detayı göremedi.
- **Konu Benzerliği (%):** Cümleniz ile referans makaledeki paragrafın aynı konudan bahsetme oranıdır.
- **Künye Doğruluk Oranı (%):** İnternette bulunan makalenin, kaynakçada yazdığınız makale olduğundan ne kadar emin olduğumuzu gösterir (`%100` = DOI birebir eşleşti).
"""
        )


def run_verification_pipeline(
    parsed_doc: ParsedManuscript,
    citations_to_check: List[InTextCitation],
    selected_model: str,
    enable_scihub: bool,
    enable_jina: bool,
    enable_suggestions: bool,
    existing_results: Optional[List[VerificationResult]] = None,
) -> tuple[List[VerificationResult], Dict[str, FetchedPaper]]:
    """
    Seçilen atıfların kaynaklarını internetten çeker, anlam doğrulamasını çalıştırır,
    mevcut sonuçlarla birleştirir ve her adımda diske otomatik kaydeder (auto-save).
    """
    fetcher = AcademicFetcher(enable_scihub=enable_scihub, enable_jina=enable_jina)
    verifier = get_verifier(selected_model)

    needed_refs_map: Dict[str, ReferenceItem] = {}
    for cit in citations_to_check:
        if cit.matched_reference is not None and cit.matched_ref_id:
            needed_refs_map[cit.matched_ref_id] = cit.matched_reference

    fetched_map: Dict[str, FetchedPaper] = dict(st.session_state.get("fetched_map", {}))
    refs_to_fetch = [
        r_it
        for r_id, r_it in needed_refs_map.items()
        if r_id not in fetched_map
        or not fetched_map[r_id].exists_online
        or len(fetched_map[r_id].abstract) < 200
    ]

    prog = st.progress(0, text="1/2: Kaynak makaleler internet veritabanlarında aranıyor...")

    if refs_to_fetch:
        done_r = 0
        total_r = len(refs_to_fetch)
        with ThreadPoolExecutor(max_workers=5) as executor:
            fut_map = {
                executor.submit(fetcher.verify_and_fetch, r_it, USER_PDF_DIR): r_it
                for r_it in refs_to_fetch
            }
            for fut in as_completed(fut_map):
                r_it = fut_map[fut]
                done_r += 1
                try:
                    fetched_map[r_it.ref_id] = fut.result()
                except Exception as exc:
                    fetched_map[r_it.ref_id] = FetchedPaper(
                        ref_id=r_it.ref_id,
                        query_title=r_it.title or r_it.raw_text,
                        query_doi=r_it.doi,
                        lookup_status="error",
                        lookup_error=f"{type(exc).__name__}: {exc}"[:200],
                    )
                prog.progress(
                    int((done_r / max(1, total_r * 2)) * 100),
                    text=f"1/2: Kaynak kontrol ediliyor [{done_r}/{total_r}]: {(r_it.title or r_it.raw_text)[:65]}...",
                )

    merged_results: List[VerificationResult] = list(existing_results or [])
    total_c = len(citations_to_check)
    for idx_c, cit in enumerate(citations_to_check, start=1):
        prog.progress(
            50 + int((idx_c / max(1, total_c)) * 50),
            text=f"2/2: Cümle ve kaynak uyumu inceleniyor [{idx_c}/{total_c}]: {cit.marker} (Otomatik kaydediliyor)...",
        )
        if cit.matched_ref_id is None or cit.matched_ref_id not in fetched_map:
            alt_list = fetcher.suggest_real_papers(cit.claim_sentence, top_k=3) if enable_suggestions else []
            v_res = VerificationResult(
                citation_id=cit.citation_id,
                marker=cit.marker,
                claim_sentence=cit.claim_sentence,
                context_text=cit.context_text,
                ref_id=None,
                verdict="HAYALET_ATIF",
                confidence_score=1.0,
                semantic_similarity=0.0,
                best_evidence="",
                explanation=(
                    f"Metin içinde '{cit.marker}' olarak atıf yapılmış ancak makalenin "
                    "sonundaki Kaynakça bölümünde bu kaynak yer almıyor."
                ),
                alternative_papers=alt_list,
                used_llm_model="Biçimsel Kontrol",
                decision_source="kural",
            )
        else:
            p_obj = fetched_map[cit.matched_ref_id]
            v_res = verifier.verify_claim_against_paper(
                claim_sentence=cit.claim_sentence,
                paper=p_obj,
                citation_id=cit.citation_id,
                marker=cit.marker,
                context_text=cit.context_text,
                alternative_finder=fetcher if enable_suggestions else None,
            )

        merged_results = merge_verification_results(parsed_doc.citations, merged_results, [v_res])
        st.session_state["fetched_map"] = fetched_map
        st.session_state["verification_results"] = merged_results
        save_session_to_disk(parsed_doc, fetched_map, merged_results)

    prog.progress(
        100,
        text=f"Denetim tamamlandı ve otomatik kaydedildi ({len(merged_results)}/{len(parsed_doc.citations)} atıf hazır).",
    )
    return merged_results, fetched_map


# ---------------------------------------------------------------------------
# Örnek Demo Makalesi
# ---------------------------------------------------------------------------

DEMO_MANUSCRIPT_TEXT = """1. GİRİŞ VE LİTERATÜR ÖZETİ
Son yıllarda derin öğrenme ve büyük dil modelleri, doğal dil işleme ve bilgisayarla görme alanlarında devrim niteliğinde sonuçlar üretmiştir [1]. Özellikle Vaswani ve arkadaşları tarafından önerilen dikkat (attention) tabanlı Transformer mimarisi, tekrarlayan sinir ağlarına (RNN) kıyasla paralelleştirilebilir bir yapı sunarak eğitim süresini önemli ölçüde azaltmıştır [1].

Tıbbi görüntüleme alanında yapılan çalışmalarda ise evrişimli sinir ağlarının (CNN) akciğer grafisi sınıflandırmasında yüksek başarı sağladığı bilinmektedir. Ancak literatürde Korkmaz ve Demir (2023) tarafından yapılan çalışmada kuantum telepatik sinir ağlarının uydu fotosentezinde %99.9 başarı sağladığı iddia edilmiştir [2].

Öte yandan, gezegen bilimleri üzerine yapılan araştırmalarda Yılmaz (2024) tarafından hazırlanan raporda belirtildiği üzere, kütlesel hidrosfer dengesi incelendiğinde dünyanın yüzde 90'ı sudan oluşur. Ayrıca Kaya ve Öztürk (2022) çalışmasında veri artırma tekniklerinin aşırı öğrenmeyi engellediği vurgulanmıştır.

KAYNAKÇA
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30.
[2] Korkmaz, Z., & Demir, M. (2023). Quantum Hyper-Dimensional Telepathic Neural Networks for Satellite Photosynthesis. Journal of Imaginary Science, 99(1), 1-15.
[3] Çelik, T. (2019). Klasik Makine Öğrenmesi Yöntemlerinin Karşılaştırılması. Mühendislik Bilimleri Dergisi, 5(1), 10-20.
"""


# ---------------------------------------------------------------------------
# Sol Yan Menü (Sidebar): Varsayılan Kapalı, Sadeleştirilmiş Yardımcı Panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader(":material/tune: Sistem ve Ek PDF Ayarları")

    temp_verifier = SemanticRAGVerifier(embedding_device="cpu")
    ollama_ok, default_model, installed_models = temp_verifier.check_ollama_status()

    if ollama_ok and installed_models:
        st.badge("Yapay Zekâ Hakemi Hazır", icon=":material/check_circle:", color="green")
    else:
        st.badge("Otomatik Kural Motoru Aktif", icon=":material/bolt:", color="blue")

    st.markdown("**📂 Kilitli Kaynak PDF'leriniz**")
    st.caption(
        "İnternette tam metni ücretli olan referans makalelerin PDF dosyaları bilgisayarınızda varsa buraya bırakın."
    )
    uploaded_ref_pdfs = st.file_uploader(
        "Referans PDF dosyaları:",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_ref_pdfs:
        for up_pdf in uploaded_ref_pdfs:
            target_path = USER_PDF_DIR / up_pdf.name
            target_path.write_bytes(up_pdf.getvalue())
        st.success(f"{len(uploaded_ref_pdfs)} adet kaynak PDF eklendi.")

    with st.expander(":material/settings: Gelişmiş Teknik Ayarlar", expanded=False):
        if ollama_ok and installed_models:
            selected_model = st.selectbox(
                "Dil Modeli:",
                options=installed_models,
                index=installed_models.index(default_model) if default_model in installed_models else 0,
            )
        else:
            selected_model = default_model or "qwen2.5:3b"
            st.caption(f"Model: `{selected_model}`")

        enable_suggestions = st.checkbox(
            "Alternatif gerçek makale öner",
            value=True,
        )
        enable_jina = st.checkbox(
            "Genişletilmiş web okuyucu (Jina)",
            value=True,
        )
        enable_scihub = st.checkbox(
            "Sci-Hub tam metin indirici",
            value=False,
            help="Hukuki uyarı: Telifli makaleleri Sci-Hub'dan indirmek sorumluluk gerektirir.",
        )


# ---------------------------------------------------------------------------
# Üst Başlık ve Sistemin Değerini Öne Çıkaran Karşılama Alanı
# ---------------------------------------------------------------------------

top_left, top_right = st.columns([3.4, 1.2], vertical_alignment="center")
with top_left:
    st.title(":material/school: Akademik Makale, Kaynakça ve Semantik Atıf Denetçisi")
    st.markdown(
        "Hakemli dergi ve TÜBİTAK proje standartlarında **üç aşamalı akademik denetim asistanınız:** "
        "Makalenizdeki **metin içi atıf–kaynakça bütünlüğünü** saniyeler içinde denetler, "
        "kaynaklarınızın **8 uluslararası akademik veritabanındaki gerçekliğini ve geri çekilme (retraction) durumunu** sorgular "
        "ve yazdığınız her bir cümlenin referans gösterdiğiniz makaleyle **anlamca örtüşüp örtüşmediğini** yapay zekâ hakemiyle doğrular."
    )
with top_right:
    if ollama_ok and installed_models:
        st.badge("Yapay Zekâ Hakemi + 8 Akademik Kanal Hazır", icon=":material/verified:", color="green")
    else:
        st.badge("Çok Dilli Semantik Kontrol Hazır", icon=":material/check_circle:", color="blue")

tab1, tab2, tab3 = st.tabs([
    ":material/fact_check: 1. Makale Yükle ve Denetle",
    ":material/travel_explore: 2. Cümleme Uygun Akademik Makale Bul",
    ":material/table_view: 3. Detaylı Rapor ve Tablo İndir",
])


# ===========================================================================
# SEKME 1: MAKALE YÜKLE VE DENETLE (ANA EKRAN)
# ===========================================================================

with tab1:
    # -----------------------------------------------------------------------
    # BÖLÜM 1: DOSYA VEYA KAYITLI OTURUM SEÇİMİ (AÇIKLAYICI VE SADE KART)
    # -----------------------------------------------------------------------
    saved_files = sorted(SESSIONS_DIR.glob("*.analiz.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    with st.container(border=True):
        st.markdown("**:material/upload_file: 1. Adım — Çalışma Yönteminizi Seçin**")
        st.caption(
            "Aşağıdaki üç seçenekten birine tıklayarak yeni bir makale analizi başlatabilir, "
            "daha önce başladığınız bir analize kaldığınız yerden devam edebilir veya örnek makaleyle sistemi test edebilirsiniz:"
        )
        entry_mode = st.segmented_control(
            "Çalışma yöntemi seçin:",
            options=[
                "📄 Yeni Makale Dosyası Yükle (.docx / .pdf)",
                f"💾 Kayıtlı Analizden Devam Et ({len(saved_files)} Kayıt)",
                "🧪 Örnek Makaleyle Sistemi Dene",
            ],
            default="📄 Yeni Makale Dosyası Yükle (.docx / .pdf)",
            label_visibility="collapsed",
        )

        uploaded_manuscript = None
        if entry_mode and entry_mode.startswith("💾"):
            st.caption(
                "💡 **Bu bölüm ne işe yarar?** Daha önce analiz ettiğiniz bir makaleyi 20-30 dakika tekrar beklememek için "
                "otomatik kaydedilen oturumlardan birini tek tıkla açabilir veya bilgisayarınıza indirdiğiniz `.json` analiz dosyasını yükleyebilirsiniz."
            )
            col_s1, col_s2 = st.columns([1.4, 1.0], vertical_alignment="bottom")
            with col_s1:
                if saved_files:
                    session_labels: Dict[str, Path] = {}
                    for sf in saved_files[:15]:
                        try:
                            meta = json.loads(sf.read_text(encoding="utf-8"))
                            lbl = (
                                f"{meta.get('filename', sf.stem)} — "
                                f"{meta.get('verified_count', 0)}/{meta.get('total_citations', '?')} Atıf Doğrulandı "
                                f"({meta.get('saved_at', '')})"
                            )
                        except Exception:
                            lbl = sf.name
                        session_labels[lbl] = sf

                    sel_c, btn_c = st.columns([2.8, 1.2], vertical_alignment="bottom")
                    with sel_c:
                        chosen_saved_label = st.selectbox(
                            "Bilgisayarınızdaki otomatik kaydedilen analizler:",
                            options=list(session_labels.keys()),
                            help="Siz çalışırken her atıf doğrulandığında sistemin otomatik olarak kaydettiği analizler.",
                        )
                    with btn_c:
                        if st.button(
                            ":material/folder_open: Seçili Kaydı Aç",
                            type="primary",
                            width="stretch",
                            help="Seçtiğiniz makalenin tüm doğrulama sonuçlarını 1 saniyede ekrana geri yükler.",
                        ):
                            sf_path = session_labels[chosen_saved_label]
                            p_doc, f_m, v_res, s_at = deserialize_session_bytes(sf_path.read_bytes())
                            activate_loaded_session(
                                p_doc,
                                f_m,
                                v_res,
                                f"'{p_doc.filename}' makalesinin kayıtlı analizi ({len(v_res)}/{len(p_doc.citations)} atıf) geri yüklendi.",
                            )
                            st.rerun()
                else:
                    st.caption("Henüz otomatik kaydedilmiş bir analiz bulunmuyor.")

            with col_s2:
                uploaded_session_json = st.file_uploader(
                    "Veya indirdiğiniz `.json` analiz dosyasını buraya bırakın:",
                    type=["json"],
                    key="session_json_uploader",
                )
                if uploaded_session_json is not None:
                    sig = f"{uploaded_session_json.name}_{uploaded_session_json.size}"
                    if st.session_state.get("last_uploaded_json_sig") != sig:
                        try:
                            p_doc, f_m, v_res, s_at = deserialize_session_bytes(uploaded_session_json.getvalue())
                            st.session_state["last_uploaded_json_sig"] = sig
                            activate_loaded_session(
                                p_doc,
                                f_m,
                                v_res,
                                f"'{p_doc.filename}' makalesinin analiz sonuçları ({len(v_res)}/{len(p_doc.citations)} atıf) dosyadan yüklendi.",
                            )
                        except Exception as exc:
                            st.error(f"Analiz dosyası okunamadı: {exc}")

        elif entry_mode and entry_mode.startswith("🧪"):
            demo_c1, demo_c2 = st.columns([3.0, 1.2], vertical_alignment="center")
            with demo_c1:
                st.caption(
                    "💡 **Bu buton ne işe yarar?** Kendi makalenizi yüklemeden önce sistemin gerçek makaleleri nasıl doğruladığını, "
                    "uydurma (sahte) kaynakları ve kaynakçada unutulan atıfları nasıl yakaladığını hazır bir örnek üzerinde hemen görmenizi sağlar."
                )
            with demo_c2:
                if st.button(
                    ":material/science: Örnek Makaleyi Yükle",
                    type="primary",
                    width="stretch",
                    help="Örnek TÜBİTAK makalesini yükler ve biçimsel kontrol sonuçlarını gösterir.",
                ):
                    st.session_state.pop("restored_parsed_doc", None)
                    st.session_state["manuscript_text"] = DEMO_MANUSCRIPT_TEXT
                    st.session_state["manuscript_name"] = "ornek_tubitak_makalesi.txt"
                    st.rerun()
        else:
            st.caption(
                "💡 **Nasıl çalışır?** Word (`.docx`), PDF (`.pdf`) veya metin (`.txt`) formatındaki makalenizi aşağıya bırakın. "
                "Sistem hem APA `(Yılmaz, 2022)` hem de numaralı `[1]` / `(1)` formatındaki tüm metin içi atıfları ve kaynakçayı otomatik ayrıştırır."
            )
            uploaded_manuscript = st.file_uploader(
                "Makale dosyanızı sürükleyip bırakın veya bilgisayarınızdan seçin:",
                type=["docx", "pdf", "txt"],
                key="manuscript_file_uploader",
            )

    parser = get_parser()
    parsed_doc: ParsedManuscript | None = None

    if uploaded_manuscript is not None:
        up_sig = f"{uploaded_manuscript.name}_{uploaded_manuscript.size}"
        if st.session_state.get("last_uploaded_doc_sig") != up_sig:
            st.session_state["last_uploaded_doc_sig"] = up_sig
            st.session_state.pop("restored_parsed_doc", None)
            auto_file = _session_file_for_doc(uploaded_manuscript.name)
            if auto_file.exists():
                try:
                    p_doc, f_m, v_res, s_at = deserialize_session_bytes(auto_file.read_bytes())
                    activate_loaded_session(
                        p_doc,
                        f_m,
                        v_res,
                        f"'{p_doc.filename}' için önceki analiz ({len(v_res)}/{len(p_doc.citations)} atıf, {s_at}) otomatik geri yüklendi.",
                    )
                    parsed_doc = p_doc
                except Exception:
                    parsed_doc = parser.parse_file(
                        uploaded_manuscript.getvalue(), filename=uploaded_manuscript.name
                    )
            else:
                parsed_doc = parser.parse_file(
                    uploaded_manuscript.getvalue(), filename=uploaded_manuscript.name
                )
        elif "restored_parsed_doc" in st.session_state:
            parsed_doc = st.session_state["restored_parsed_doc"]
        else:
            parsed_doc = parser.parse_file(
                uploaded_manuscript.getvalue(), filename=uploaded_manuscript.name
            )
    elif "restored_parsed_doc" in st.session_state:
        parsed_doc = st.session_state["restored_parsed_doc"]
    elif "manuscript_text" in st.session_state:
        parsed_doc = parser.parse_text(
            st.session_state["manuscript_text"],
            filename=st.session_state.get("manuscript_name", "demo_makale.txt"),
        )

    if st.session_state.get("session_restore_notice"):
        st.success(st.session_state["session_restore_notice"], icon=":material/restore:")

    if parsed_doc is not None:
        if st.session_state.get("active_doc_name") != parsed_doc.filename:
            st.session_state["active_doc_name"] = parsed_doc.filename
            st.session_state["selected_cit_indices"] = set(range(1, min(6, len(parsed_doc.citations) + 1)))
            st.session_state["editor_ver"] = 0
            st.session_state["fetched_map"] = {}
            st.session_state.pop("verification_results", None)
            st.session_state.pop("session_restore_notice", None)

        st.session_state["current_parsed_doc"] = parsed_doc

        # -------------------------------------------------------------------
        # BÖLÜM 2: BİÇİMSEL KAYNAKÇA VE ATIF ÖZETİ (4 AÇIKLAYICI METRİK KUTUSU)
        # -------------------------------------------------------------------
        st.space("small")
        st.subheader(f":material/description: 2. Adım — `{parsed_doc.filename}` Biçimsel Kaynakça ve Atıf Bütünlüğü")
        st.caption(
            "Bu aşama internet taraması yapmadan önce makalenizin kendi içindeki **Metin İçi Atıflar** ile makale sonundaki "
            "**Kaynakça (References)** listesini çapraz kontrol eder; unutulan veya açıkta kalan kaynakları anında tespit eder."
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1.container(border=True, height="stretch"):
            st.metric("Kaynakçadaki Makale", len(parsed_doc.references))
            st.caption("Makale sonundaki Kaynakça listesinde yer alan toplam referans sayısı.")
        with b2.container(border=True, height="stretch"):
            st.metric("Metin İçi Atıf", len(parsed_doc.citations))
            st.caption("Makale cümleleri içinde parantez veya numarayla yapılan toplam atıf sayısı.")
        with b3.container(border=True, height="stretch"):
            ghost_n = len(parsed_doc.ghost_citations)
            st.metric(
                "Kaynakçada Eksik Atıf",
                ghost_n,
                delta="Düzeltilmeli" if ghost_n else "Tam Uyumlu",
                delta_color="inverse" if ghost_n else "normal",
            )
            st.caption("Metin içinde atıf yapılmış ancak Kaynakça listesine eklenmesi unutulmuş kaynaklar.")
        with b4.container(border=True, height="stretch"):
            orphan_n = len(parsed_doc.orphan_references)
            st.metric(
                "Metinde Kullanılmayan Kaynak",
                orphan_n,
                delta="Düzeltilmeli" if orphan_n else "Tam Uyumlu",
                delta_color="inverse" if orphan_n else "normal",
            )
            st.caption("Kaynakça listesinde yazılı olduğu hâlde metinde hiçbir cümlede atıf yapılmamış kaynaklar.")

        if parsed_doc.ghost_citations or parsed_doc.orphan_references:
            with st.expander(
                f"🔧 Biçimsel Kaynakça Sorunlarını Gör ({ghost_n} Eksik Atıf, {orphan_n} Kullanılmayan Kaynak)",
                expanded=True,
            ):
                if parsed_doc.ghost_citations:
                    st.error(
                        f"**{len(parsed_doc.ghost_citations)} adet Kaynakçada Eksik Atıf:** Metin içinde atıf yapılmış ancak makale sonundaki Kaynakça listesine yazılmamış:",
                    )
                    for gc in parsed_doc.ghost_citations:
                        st.markdown(f"- `{gc.marker}` — *\"{gc.claim_sentence}\"*")

                if parsed_doc.orphan_references:
                    st.warning(
                        f"**{len(parsed_doc.orphan_references)} adet Metinde Kullanılmayan Kaynak:** Kaynakça listesinde var ancak makale metninde hiçbir cümlede kullanılmamış:",
                    )
                    for orp in parsed_doc.orphan_references:
                        st.markdown(f"- **[{orp.ref_id}]** {orp.raw_text}")

        with st.expander(
            f":material/list_alt: Kaynak Eşleşme Kanıt Tablosunu İncele ({len(parsed_doc.references)} Kaynak)",
            expanded=False,
        ):
            st.caption(
                "💡 **Bu tablo ne işe yarar?** Kaynakçanızdaki her bir eserin makale metni içinde kaç kez kullanıldığını ve "
                "sistemin o kaynağı metin içinde **gerçekten hangi cümlede yakaladığını (Eşleşme Kanıtı)** eksiksiz olarak gösterir. "
                "*(Uzun bir cümlenin tamamını kutu içinde görmek için ilgili hücreye çift tıklayabilir veya tabloyu sağa kaydırabilirsiniz).* "
            )
            ref_to_cits: Dict[str, List[Any]] = {}
            for c_item in parsed_doc.citations:
                if c_item.matched_ref_id:
                    ref_to_cits.setdefault(c_item.matched_ref_id, []).append(c_item)

            proof_rows = []
            for r_item in parsed_doc.references:
                matched_c_list = ref_to_cits.get(r_item.ref_id, [])
                markers_str = ", ".join(dict.fromkeys(c.marker for c in matched_c_list)) if matched_c_list else "—"
                first_sent = matched_c_list[0].claim_sentence if matched_c_list else "❌ Metinde hiç kullanılmamış"
                proof_rows.append({
                    "Kaynak No": f"[{r_item.ref_id}]",
                    "Yazar ve Yıl": f"{r_item.first_author_surname or 'Bilinmiyor'} ({r_item.year or '?'})",
                    "Kullanım": f"{r_item.cited_in_text_count} kez",
                    "Metindeki İşareti": markers_str,
                    "Kaynak Başlığı": (r_item.title or r_item.raw_text),
                    "Yakalandığı Metin İçi Cümle (Eşleşme Kanıtı)": first_sent,
                })
            st.dataframe(
                pd.DataFrame(proof_rows),
                hide_index=True,
                row_height=52,
                column_config={
                    "Kaynak No": st.column_config.TextColumn("Kaynak No", width="small"),
                    "Yazar ve Yıl": st.column_config.TextColumn("Yazar ve Yıl", width="small"),
                    "Kullanım": st.column_config.TextColumn("Kullanım", width="small"),
                    "Metindeki İşareti": st.column_config.TextColumn("Metindeki İşareti", width="medium"),
                    "Kaynak Başlığı": st.column_config.TextColumn("Kaynak Makale Başlığı", width="large"),
                    "Yakalandığı Metin İçi Cümle (Eşleşme Kanıtı)": st.column_config.TextColumn(
                        "Yakalandığı Metin İçi Cümle (Eşleşme Kanıtı)",
                        width="large",
                        help="Sistemin bu kaynağı makalenizin ana metni içinde tespit ettiği tam cümle.",
                    ),
                },
            )

        # -------------------------------------------------------------------
        # BÖLÜM 3: DENETİM BAŞLATMA MERKEZİ (KAYDIRMASIZ TAM GÖRÜNÜM + AÇIKLAYICI REHBER)
        # -------------------------------------------------------------------
        st.space("small")
        all_cits = parsed_doc.citations
        total_cits_n = len(all_cits)
        existing_res_list: List[VerificationResult] = st.session_state.get("verification_results", [])
        done_cit_ids = {r.citation_id for r in existing_res_list}
        remaining_cits = [c for c in all_cits if c.citation_id not in done_cit_ids]

        with st.container(border=True):
            st.subheader(":material/travel_explore: 3. Adım — Kaynak Gerçekliği ve Semantik Cümle Uyumu Denetimi")
            if done_cit_ids:
                st.caption(
                    f"📊 **Kayıtlı İlerleme Durumu:** Toplam **{total_cits_n}** atıftan **{len(done_cit_ids)}** tanesi zaten doğrulandı ve hafızada hazır "
                    f"(**{len(remaining_cits)}** atıf henüz doğrulanmadı). Yeni başlatacağınız denetimler mevcut sonuçların üzerine eklenir."
                )
            else:
                st.caption(
                    "Bu aşamada makalenizdeki kaynaklar **CrossRef, OpenAlex, PubMed ve Semantic Scholar** üzerinden taranarak gerçeklikleri doğrulanır; "
                    "ardından yazdığınız her bir cümlenin kaynak makaledeki bulgularla **uyumlu olup olmadığı veya çelişip çelişmediği** yapay zekâ hakemiyle denetlenir. "
                    "**Doğrulanan her atıf anında bilgisayarınıza otomatik kaydedilir.**"
                )

            seg_options = []
            if remaining_cits and done_cit_ids:
                seg_options.append(f"⏭️ Kaldığım Yerden Devam Et ({len(remaining_cits)} Kalan Atıf)")
            seg_options.extend([
                f"🚀 Tüm Makaleyi Baştan Sona Denetle ({total_cits_n} Atıf)",
                "🎯 Belirli Bir Aralığı veya Seçtiğim Atıfları Denetle",
            ])

            st.markdown("**Denetim yöntemini seçin (her iki seçenek de kaydırma yapmadan doğrudan görünür):**")
            scope_choice = st.segmented_control(
                "Denetim kapsamı:",
                options=seg_options,
                default=seg_options[0],
                label_visibility="collapsed",
            )

            citations_to_run: List[InTextCitation] = []

            if not scope_choice or scope_choice.startswith("⏭️"):
                citations_to_run = remaining_cits
                needed_unique_refs = len({c.matched_ref_id for c in citations_to_run if c.matched_ref_id})
                act_col1, act_col2 = st.columns([2.6, 1.2], vertical_alignment="center")
                with act_col1:
                    st.markdown(
                        f"💡 **Bu buton ne yapar?** Daha önce doğrulanan **{len(done_cit_ids)}** atıfa hiç dokunmadan, "
                        f"yalnızca henüz incelenmemiş kalan **{len(citations_to_run)}** atıfı (**{needed_unique_refs}** kaynak makaleyi) denetleyip mevcut sonuçlarınızla birleştirir."
                    )
                with act_col2:
                    run_btn = st.button(
                        f":material/play_arrow: Kalan {len(citations_to_run)} Atıfı Tamamla",
                        type="primary",
                        width="stretch",
                        disabled=(len(citations_to_run) == 0),
                    )

            elif scope_choice.startswith("🚀"):
                citations_to_run = all_cits
                needed_unique_refs = len({c.matched_ref_id for c in citations_to_run if c.matched_ref_id})
                act_col1, act_col2 = st.columns([2.6, 1.2], vertical_alignment="center")
                with act_col1:
                    st.markdown(
                        f"💡 **Bu buton ne yapar?** Makalenizdeki **{len(citations_to_run)}** metin içi atıfın tamamını ve "
                        f"bunlara karşılık gelen **{needed_unique_refs}** kaynak makaleyi sırayla internetten tarar, cümle uyumunu denetler ve her adımda otomatik kaydeder."
                    )
                with act_col2:
                    run_btn = st.button(
                        f":material/rocket_launch: Tüm Makaleyi Denetle ({len(citations_to_run)} Atıf)",
                        type="primary",
                        width="stretch",
                        disabled=(len(citations_to_run) == 0),
                    )

            else:
                if "selected_cit_indices" not in st.session_state:
                    st.session_state["selected_cit_indices"] = set(range(1, min(11, total_cits_n + 1)))
                if "editor_ver" not in st.session_state:
                    st.session_state["editor_ver"] = 0

                st.caption(
                    "💡 **Bu mod ne işe yarar?** Tüm makaleyi tek seferde beklemek yerine istediğiniz bir aralığı (örneğin **1 ile 15 arası** veya **doğrulanmamış ilk 10 atıfı**) "
                    "seçip parça parça denetlemenizi sağlar. Seçtiğiniz aralık doğrulandığında önceki sonuçlarınızın üzerine eklenir."
                )

                # Tek satırda sade aralık seçici + Başlat butonu
                bar1, bar2, bar3, bar4 = st.columns([1.6, 1.3, 1.1, 1.4], vertical_alignment="bottom")
                with bar1:
                    quick_preset = st.selectbox(
                        "Hızlı Seçim Şablonu:",
                        options=[
                            "Özel Aralık (Yan Çubuktan Seç)",
                            "⏳ Doğrulanmamış İlk 10 Atıf",
                            "⏳ Doğrulanmamış İlk 25 Atıf",
                            f"✅ Tüm Atıflar (1 – {total_cits_n})",
                            "🧹 Seçimi Temizle",
                        ],
                        index=0,
                        help="Hazır bir grup seçebilir veya yan taraftaki kaydırma çubuğuyla özel numara aralığı belirleyebilirsiniz.",
                    )
                with bar2:
                    range_vals = st.slider(
                        "Atıf Numarası Aralığı:",
                        min_value=1,
                        max_value=max(2, total_cits_n),
                        value=(1, min(10, max(2, total_cits_n))),
                        step=1,
                        help="Denetlemek istediğiniz başlangıç ve bitiş atıf numaralarını sürükleyerek seçin.",
                    )
                with bar3:
                    if st.button(
                        ":material/check_box: Seçimi Tabloya Uygula",
                        width="stretch",
                        help="Belirlediğiniz aralığı aşağıdaki tabloda otomatik işaretler.",
                    ):
                        if quick_preset.startswith("⏳ Doğrulanmamış İlk 10"):
                            idxs = [i for i, c_o in enumerate(all_cits, start=1) if c_o.citation_id not in done_cit_ids][:10]
                            st.session_state["selected_cit_indices"] = set(idxs)
                        elif quick_preset.startswith("⏳ Doğrulanmamış İlk 25"):
                            idxs = [i for i, c_o in enumerate(all_cits, start=1) if c_o.citation_id not in done_cit_ids][:25]
                            st.session_state["selected_cit_indices"] = set(idxs)
                        elif quick_preset.startswith("✅"):
                            st.session_state["selected_cit_indices"] = set(range(1, total_cits_n + 1))
                        elif quick_preset.startswith("🧹"):
                            st.session_state["selected_cit_indices"] = set()
                        else:
                            st.session_state["selected_cit_indices"] = set(range(range_vals[0], range_vals[1] + 1))
                        st.session_state["editor_ver"] += 1
                        st.rerun()

                current_sel_set = st.session_state["selected_cit_indices"]
                editor_rows = []
                for idx_c, c_obj in enumerate(all_cits, start=1):
                    ref_obj = c_obj.matched_reference
                    ref_summary = (
                        f"[{c_obj.matched_ref_id}] {ref_obj.first_author_surname or ''} ({ref_obj.year or '?'}) — {(ref_obj.title or ref_obj.raw_text)}"
                        if ref_obj
                        else "🚨 Kaynakça Listesinde Eksik"
                    )
                    editor_rows.append({
                        "Seç": idx_c in current_sel_set,
                        "Atıf No": idx_c,
                        "Durum": "✅ Doğrulandı" if c_obj.citation_id in done_cit_ids else "⏳ Bekliyor",
                        "Atıf İşareti": c_obj.marker,
                        "Eşleşen Kaynak": ref_summary,
                        "Makalenizdeki Cümle": c_obj.claim_sentence,
                    })

                edited_df = st.data_editor(
                    pd.DataFrame(editor_rows),
                    key=f"cit_selector_{parsed_doc.filename}_{st.session_state['editor_ver']}",
                    hide_index=True,
                    height=260,
                    row_height=48,
                    disabled=["Atıf No", "Durum", "Atıf İşareti", "Eşleşen Kaynak", "Makalenizdeki Cümle"],
                    column_config={
                        "Seç": st.column_config.CheckboxColumn("Seç", default=False, width="small"),
                        "Atıf No": st.column_config.NumberColumn("No", width="small"),
                        "Durum": st.column_config.TextColumn("Durum", width="small"),
                        "Atıf İşareti": st.column_config.TextColumn("Atıf İşareti", width="small"),
                        "Eşleşen Kaynak": st.column_config.TextColumn("Kaynakçadaki Karşılığı", width="large"),
                        "Makalenizdeki Cümle": st.column_config.TextColumn("Makalenizdeki İddia Cümlesi", width="large"),
                    },
                )

                selected_indices = [
                    int(row["Atıf No"]) for _, row in edited_df.iterrows() if bool(row["Seç"])
                ]
                st.session_state["selected_cit_indices"] = set(selected_indices)
                citations_to_run = [all_cits[i - 1] for i in selected_indices]
                needed_unique_refs = len({c.matched_ref_id for c in citations_to_run if c.matched_ref_id})

                with bar4:
                    run_btn = st.button(
                        f":material/play_arrow: Seçilen {len(citations_to_run)} Atıfı Denetle",
                        type="primary",
                        width="stretch",
                        disabled=(len(citations_to_run) == 0),
                        help="Tabloda işaretlediğiniz atıfları denetler ve otomatik olarak kaydeder.",
                    )

        if run_btn and citations_to_run:
            results_out, fetched_out = run_verification_pipeline(
                parsed_doc=parsed_doc,
                citations_to_check=citations_to_run,
                selected_model=selected_model,
                enable_scihub=enable_scihub,
                enable_jina=enable_jina,
                enable_suggestions=enable_suggestions,
                existing_results=existing_res_list,
            )
            st.session_state["verification_results"] = results_out
            st.session_state["fetched_map"] = fetched_out
            st.rerun()

        # -------------------------------------------------------------------
        # BÖLÜM 4: DENETİM SONUÇLARI (3 ÖZET KUTUSU + 9 TİKLİ FİLTRE + KARTLAR)
        # -------------------------------------------------------------------
        if "verification_results" in st.session_state and st.session_state["verification_results"]:
            results_list: List[VerificationResult] = st.session_state["verification_results"]
            f_map: Dict[str, FetchedPaper] = st.session_state.get("fetched_map", {})

            st.space("small")
            hdr_left, hdr_right = st.columns([2.5, 1.1], vertical_alignment="center")
            with hdr_left:
                st.subheader(
                    f":material/analytics: 4. Adım — Denetim Sonuçları Özeti ({len(results_list)} / {len(parsed_doc.citations)} Atıf İncelendi)"
                )
                st.caption(
                    "İncelenen atıfların doğruluk dağılımını aşağıda görebilir; sağdaki butonla bu analiz oturumunu bilgisayarınıza yedekleyebilirsiniz."
                )
            with hdr_right:
                session_bytes_t1 = serialize_session_bytes(parsed_doc, f_map, results_list)
                safe_dl_stem = safe_ref_token(Path(parsed_doc.filename).stem, fallback="makale")
                st.download_button(
                    ":material/download: Analiz Kaydını İndir (.json)",
                    data=session_bytes_t1,
                    file_name=f"{safe_dl_stem}_analiz_kaydi.json",
                    mime="application/json",
                    type="primary",
                    width="stretch",
                    help="Bu dosyayı indirerek daha sonra siteye yükleyebilir ve 20-30 dakika beklemeden aynı sonuçları 1 saniyede açabilirsiniz.",
                )

            # Alt kırılım sayıları
            cnt_sup = sum(1 for r in results_list if r.verdict == "DESTEKLENİYOR" and not r.is_retracted)
            cnt_part = sum(1 for r in results_list if r.verdict == "KISMEN_DESTEKLENİYOR" and not r.is_retracted)

            cnt_con = sum(1 for r in results_list if r.verdict == "ÇELİŞİYOR" and not r.is_retracted)
            cnt_ghost = sum(1 for r in results_list if r.verdict == "HAYALET_ATIF" and not r.is_retracted)
            cnt_fake = sum(1 for r in results_list if r.verdict == "SAHTE_KAYNAK" and not r.is_retracted)
            cnt_ret = sum(1 for r in results_list if r.is_retracted)

            cnt_ins = sum(1 for r in results_list if r.verdict == "KANIT_YETERSİZ" and not r.is_retracted)
            cnt_unk = sum(1 for r in results_list if r.verdict == "DOGRULANAMADI" and not r.is_retracted)

            total_ok = cnt_sup + cnt_part
            total_error = cnt_con + cnt_ghost + cnt_fake + cnt_ret
            total_manual = cnt_ins + cnt_unk

            # 3 BÜYÜK VE AÇIKLAYICI ÖZET KARTI
            sum_col1, sum_col2, sum_col3 = st.columns(3)

            with sum_col1.container(border=True, height="stretch"):
                st.metric("🟢 Sorunsuz / Doğrulanan", f"{total_ok} Atıf")
                st.markdown(
                    f"- ✅ **{cnt_sup} Tam Uyumlu:** Cümleniz kaynak makaleyle doğrudan örtüşüyor.\n"
                    f"- ☑️ **{cnt_part} Özetle Uyumlu:** Kaynak gerçek ve özeti cümlenizle uyumlu."
                )
                st.caption("Değişiklik yapmanıza gerek yoktur.")

            with sum_col2.container(border=True, height="stretch"):
                st.metric(
                    "🔴 Düzeltilmesi Gerekenler",
                    f"{total_error} Atıf",
                    delta="Düzeltilmeli" if total_error > 0 else "Hata Yok",
                    delta_color="inverse" if total_error > 0 else "normal",
                )
                st.markdown(
                    f"- ❌ **{cnt_con} Çelişen Bilgi:** Cümleniz kaynak makalenin tersini söylüyor.\n"
                    f"- 🚨 **{cnt_ghost} Kaynakçada Eksik:** Metinde var ama Kaynakça listesinde yok.\n"
                    f"- 🚨 **{cnt_fake} Hatalı DOI / Künye:** DOI başka bir makaleye ait.\n"
                    f"- 🛑 **{cnt_ret} Geri Çekilmiş:** Makale yayından kaldırılmış."
                )
                st.caption("Aşağıdaki listeden bu cümleleri açıp düzeltin.")

            with sum_col3.container(border=True, height="stretch"):
                st.metric(
                    "🟡 Sizin Kontrol Edecekleriniz",
                    f"{total_manual} Atıf",
                    delta="Gözle Kontrol" if total_manual > 0 else "Temiz",
                    delta_color="off",
                )
                st.markdown(
                    f"- ⚠️ **{cnt_ins} Özette Detay Yok:** Makale gerçek ama detay kilitli tam metinde olabilir.\n"
                    f"- ❓ **{cnt_unk} İnternette Erişilemedi:** Veritabanlarında bulunamadı (kitap/tez olabilir)."
                )
                st.caption("Kaynağı gözle kontrol edin veya PDF'ini yan menüden yükleyin.")

            render_reading_guide()

            # 9 TİKLİ (TÜMÜNÜ GÖSTER + 8 SAYAÇ) FİLTRELEME MERKEZİ
            _init_filter_state("f1")

            with st.container(border=True):
                st.caption(":material/filter_list: **Görmek istediğiniz sonuç türlerini işaretleyin:**")
                row1 = st.columns(5)
                with row1[0]:
                    st.checkbox(
                        f"📋 Tümünü Göster ({len(results_list)})",
                        key="f1_ALL",
                        on_change=_on_toggle_all,
                        args=("f1",),
                    )
                with row1[1]:
                    st.checkbox(
                        f"✅ Tam Desteklenen ({cnt_sup})",
                        key="f1_SUP",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )
                with row1[2]:
                    st.checkbox(
                        f"☑️ Özet Uyumlu ({cnt_part})",
                        key="f1_PART",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )
                with row1[3]:
                    st.checkbox(
                        f"❌ Çelişen ({cnt_con})",
                        key="f1_CON",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )
                with row1[4]:
                    st.checkbox(
                        f"⚠️ Kanıt Yetersiz ({cnt_ins})",
                        key="f1_INS",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )

                row2 = st.columns(5)
                with row2[0]:
                    st.checkbox(
                        f"🚨 Sahte / Hatalı DOI ({cnt_fake})",
                        key="f1_FAKE",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )
                with row2[1]:
                    st.checkbox(
                        f"❓ Doğrulanamadı ({cnt_unk})",
                        key="f1_UNK",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )
                with row2[2]:
                    st.checkbox(
                        f"👻 Hayalet / Eksik ({cnt_ghost})",
                        key="f1_GHOST",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )
                with row2[3]:
                    st.checkbox(
                        f"🛑 Geri Çekilmiş ({cnt_ret})",
                        key="f1_RET",
                        on_change=_on_toggle_single,
                        args=("f1",),
                    )

            active_codes_f1 = {k for k in ALL_FILTER_KEYS if st.session_state.get(f"f1_{k}", False)}

            filtered_results: List[VerificationResult] = [
                r for r in results_list if _result_category_code(r) in active_codes_f1
            ]

            st.caption(
                f"Toplam **{len(results_list)}** atıf içinden filtreye uyan **{len(filtered_results)}** atıf gösteriliyor."
            )

            if not filtered_results:
                st.info("İşaretlediğiniz filtreye uyan herhangi bir atıf bulunmuyor.")

            for r in filtered_results:
                v_info = get_verdict_info(r.verdict, is_retracted=r.is_retracted)
                header_text = f"{v_info['badge']}  ·  {r.marker} — \"{r.claim_sentence[:90]}...\""
                is_open_default = v_info["group"] == "error"

                with st.expander(header_text, expanded=is_open_default):
                    c_left, c_right = st.columns([3, 2])
                    with c_left:
                        st.markdown(f"**Makalenizdeki Cümle:**\n> {r.claim_sentence}")

                        with st.container(border=True):
                            st.markdown(f"**Neden Bu Sonuç Çıktı?** {v_info['why']}")
                            st.markdown(f"**Değerlendirme:** {r.explanation}")
                            st.markdown(f"**Ne Yapmalısınız?** {v_info['action']}")

                        if r.suggested_revision:
                            st.success(
                                f"**\"{r.suggested_revision}\"**",
                                title="✏️ Önerilen Düzeltilmiş Akademik Cümle",
                            )

                        if r.best_evidence:
                            st.info(
                                f"{r.best_evidence}",
                                title=f"📖 Kaynak Makaledeki İlgili Paragraf (Konu Benzerliği: %{r.semantic_similarity * 100:.0f})",
                            )

                        if r.is_secondary_citation and r.secondary_citation_note:
                            st.warning(r.secondary_citation_note, title="ℹ️ İkincil Atıf Uyarısı")

                    with c_right:
                        with st.container(border=True):
                            st.markdown("**:material/menu_book: Kaynak Makale Künyesi**")
                            if r.ref_id and r.ref_id in f_map:
                                fp = f_map[r.ref_id]
                                st.markdown(f"- **Başlık:** {fp.verified_title or fp.query_title}")
                                st.markdown(f"- **Yazarlar:** {', '.join(fp.verified_authors[:4]) or '-'}")
                                st.markdown(f"- **Yıl & Dergi:** {fp.verified_year or '-'} | {fp.verified_venue or '-'}")
                                st.markdown(f"- **DOI:** `{fp.verified_doi or 'Belirtilmemiş'}`")
                                st.markdown(f"- **Okunan Kaynak:** `{fp.content_source}`")
                                st.markdown(
                                    f"- **Künye Doğruluk Oranı:** "
                                    f"{f'%{fp.title_match_score * 100:.0f}' if fp.exists_online else 'Doğrulanamadı'}"
                                )
                                st.markdown(f"- **Değerlendiren:** {human_decision_source(r.decision_source)}")
                                if fp.local_pdf_path:
                                    st.caption(f"📄 Tam metin PDF okundu: `{fp.local_pdf_path}`")
                                for w in fp.metadata_warnings:
                                    st.warning(w)
                            else:
                                st.error("Bu atıf makale sonundaki Kaynakça listesinde bulunmuyor.")

                        if r.alternative_papers:
                            with st.container(border=True):
                                st.markdown("**:material/lightbulb: Alternatif Gerçek Makale Önerileri**")
                                for alt in r.alternative_papers:
                                    st.markdown(
                                        f"- **{alt.title}** ({alt.authors}, {alt.year}) — *{alt.venue}* "
                                        f"[DOI: `{alt.doi}` | Atıf: {alt.citation_count}]"
                                    )

            with st.expander(":material/fact_check: Kaynakça Listesindeki Makalelerin İnternet Doğrulama Tablosu", expanded=False):
                st.caption(
                    "💡 **Bu tablo ne işe yarar?** Makalenizin sonundaki Kaynakça listesinde yer alan her bir makalenin uluslararası "
                    "veritabanlarında bulunup bulunmadığını, hangi kanaldan okunduğunu ve künye doğruluk oranını toplu olarak listeler."
                )
                ref_rows = []
                for r_id, fp in f_map.items():
                    ref_rows.append({
                        "Kaynak No": f"[{r_id}]",
                        "İnternet Durumu": (
                            "🛑 Yayından Geri Çekilmiş" if fp.is_retracted
                            else ("✅ İnternette Doğrulandı" if fp.exists_online
                                  else ("❓ Bağlantı Hatası" if fp.lookup_status == "error"
                                        else "❓ Veritabanlarında Bulunamadı"))
                        ),
                        "Makale Başlığı": fp.verified_title or fp.query_title,
                        "Yazarlar": ", ".join(fp.verified_authors[:3]) if fp.verified_authors else "-",
                        "Yıl": fp.verified_year or "-",
                        "Okunan İçerik": fp.content_source,
                        "Künye Doğruluk Oranı": f"%{fp.title_match_score * 100:.0f}" if fp.exists_online else "-",
                        "Not / Uyarı": fp.lookup_error or (" | ".join(fp.metadata_warnings) if fp.metadata_warnings else "Sorun Yok"),
                    })
                st.dataframe(pd.DataFrame(ref_rows), hide_index=True, row_height=45)


# ===========================================================================
# SEKME 2: CÜMLEME UYGUN AKADEMİK MAKALE VE ATIF BULUCU (DOSYA YÜKLEMEDEN)
# ===========================================================================


def _format_citation_strings(fp: FetchedPaper, idx: int = 1) -> Dict[str, str]:
    """Bulunan bir makale için hazır metin içi atıf (APA/Vancouver) ve kaynakça künyesi üretir."""
    authors = fp.verified_authors or []
    year = fp.verified_year or "t.y."
    title = fp.verified_title or fp.query_title
    venue = fp.verified_venue or "Akademik Dergi"
    doi_str = f" https://doi.org/{fp.verified_doi}" if fp.verified_doi else ""

    def _surname(full_name: str) -> str:
        parts = full_name.strip().split()
        return parts[-1] if parts else full_name

    if not authors:
        intext_tr = f"(Anonim, {year})"
        intext_en = f"(Anonymous, {year})"
        apa_authors = "Anonim"
    elif len(authors) == 1:
        s1 = _surname(authors[0])
        intext_tr = f"({s1}, {year})"
        intext_en = f"({s1}, {year})"
        apa_authors = authors[0]
    elif len(authors) == 2:
        s1, s2 = _surname(authors[0]), _surname(authors[1])
        intext_tr = f"({s1} ve {s2}, {year})"
        intext_en = f"({s1} & {s2}, {year})"
        apa_authors = f"{authors[0]} & {authors[1]}"
    else:
        s1 = _surname(authors[0])
        intext_tr = f"({s1} vd., {year})"
        intext_en = f"({s1} et al., {year})"
        apa_authors = ", ".join(authors[:6]) + (", et al." if len(authors) > 6 else "")

    apa_ref = f"{apa_authors} ({year}). {title}. {venue}.{doi_str}".strip()
    vanc_authors = ", ".join(authors[:6]) + (", et al." if len(authors) > 6 else "")
    vanc_ref = f"[{idx}] {vanc_authors}. {title}. {venue}. {year}.{doi_str}".strip()

    return {
        "intext_tr": intext_tr,
        "intext_en": intext_en,
        "intext_num": f"[{idx}]",
        "apa_ref": apa_ref,
        "vanc_ref": vanc_ref,
    }


with tab2:
    with st.container(border=True):
        st.subheader(":material/travel_explore: Cümlenize Uygun Yayınlanmış Akademik Makale ve Atıf Bulucu")
        st.markdown(
            "Makale yazarken **\"Bu cümleme hangi akademik kaynağı atıf gösterebilirim?\"** diye düşündüğünüzde, "
            "cümlenizi Türkçe veya İngilizce olarak aşağıya yazın. Sistem **OpenAlex (250M+ eser), Semantic Scholar ve PubMed** "
            "veritabanlarını tarar, aday makalelerin özetlerini okuyarak **sizin cümlenizi bilimsel olarak en güçlü destekleyen çalışmaları** "
            "yukarıdan aşağıya sıralar ve **tek tıkla kopyalamaya hazır APA / Vancouver kaynakça künyesini** sunar."
        )

        claim_input_t2 = st.text_area(
            "📌 Makalenizdeki bilimsel iddia veya atıf cümlesini yazın (Türkçe veya İngilizce):",
            value=(
                "Siklofosfamid maruziyeti testiste oksidatif stresi artırarak "
                "sperm motilitesini ve morfolojisini bozmaktadır."
            ),
            height=95,
            placeholder="Örn: Derin öğrenme modelleri akciğer kanseri tespitinde yüksek doğruluk sağlamaktadır...",
        )

        opt_c1, opt_c2, opt_c3, opt_c4 = st.columns([1.1, 1.2, 1.2, 1.6], vertical_alignment="bottom")
        with opt_c1:
            max_papers_t2 = st.selectbox(
                "İncelenecek Makale:",
                options=[6, 10, 15],
                index=1,
                help="Veritabanlarından çekilip yapay zekâ tarafından cümlenizle karşılaştırılacak aday makale sayısı.",
            )
        with opt_c2:
            min_year_choice = st.selectbox(
                "Yayın Yılı Filtresi:",
                options=["Tüm Yıllar", "2015 ve sonrası", "2018 ve sonrası", "2020 ve sonrası", "2022 ve sonrası"],
                index=0,
                help="Yalnızca seçtiğiniz yıldan itibaren yayınlanmış güncel makaleleri getirmek için kullanın.",
            )
        with opt_c3:
            only_oa_t2 = st.checkbox(
                "Sadece Açık Erişim (PDF)",
                value=False,
                help="İşaretlenirse yalnızca tam metin PDF bağlantısı açık olan makaleleri listeler.",
            )
        with opt_c4:
            search_btn_t2 = st.button(
                ":material/search: Uygun Makaleleri Araştır ve Sırala",
                type="primary",
                width="stretch",
                help="Yazdığınız cümleyi destekleyen gerçek akademik makaleleri internetten bulur ve uyum sırasına göre listeler.",
            )

    if search_btn_t2:
        if not claim_input_t2 or len(claim_input_t2.strip()) < 8:
            st.warning("Lütfen araştırılacak anlamlı bir akademik cümle yazın.")
        else:
            year_map = {
                "Tüm Yıllar": None,
                "2015 ve sonrası": 2015,
                "2018 ve sonrası": 2018,
                "2020 ve sonrası": 2020,
                "2022 ve sonrası": 2022,
            }
            year_from_val = year_map.get(min_year_choice)

            fetcher_t2 = AcademicFetcher(enable_scihub=enable_scihub, enable_jina=enable_jina)
            verifier_t2 = get_verifier(selected_model)

            with st.status("Akademik veritabanları taranıyor ve cümlenizle karşılaştırılıyor...", expanded=True) as status_box:
                st.write("1️⃣ Bilimsel kavramlar analiz ediliyor ve OpenAlex, Semantic Scholar, PubMed taranıyor...")
                candidates_t2, query_meta_t2 = fetcher_t2.search_candidate_papers_for_claim(
                    claim_sentence=claim_input_t2.strip(),
                    max_candidates=int(max_papers_t2),
                    year_from=year_from_val,
                    only_open_access=only_oa_t2,
                    ollama_model=selected_model,
                )

                st.write(
                    f"🔎 **Arama Terimleri:** `{query_meta_t2.get('english_query', '')}` | "
                    f"**Bulunan Aday Makale:** `{len(candidates_t2)}` adet"
                )

                if not candidates_t2:
                    status_box.update(label="Uygun makale bulunamadı.", state="error", expanded=True)
                    st.session_state["tab2_ranked_results"] = []
                else:
                    st.write("2️⃣ Aday makalelerin özetleri okunuyor ve cümlenizle anlam uyumu hesaplanıyor...")
                    pre_scored: List[Tuple[float, FetchedPaper]] = []
                    for cand_p in candidates_t2:
                        top_ev = verifier_t2.retrieve_top_evidence(
                            claim_input_t2.strip(), cand_p.combined_evidence_text, top_k=2
                        )
                        best_sim = top_ev[0].semantic_score if top_ev else 0.0
                        pre_scored.append((best_sim, cand_p))

                    pre_scored.sort(key=lambda x: x[0], reverse=True)
                    shortlist = [p for _, p in pre_scored[: int(max_papers_t2)]]

                    ranked_items: List[Dict[str, Any]] = []
                    prog_t2 = st.progress(0, text="Makaleler cümlenizle karşılaştırılıyor...")
                    for idx_p, cand_paper in enumerate(shortlist, start=1):
                        prog_t2.progress(
                            idx_p / len(shortlist),
                            text=f"({idx_p}/{len(shortlist)}) İnceleniyor: {cand_paper.verified_title[:75]}...",
                        )
                        v_res = verifier_t2.verify_claim_against_paper(
                            claim_sentence=claim_input_t2.strip(),
                            paper=cand_paper,
                            citation_id=f"FIND_{idx_p}",
                            marker=f"[{idx_p}]",
                        )

                        verdict_weight = {
                            "DESTEKLENİYOR": 3.0,
                            "KISMEN_DESTEKLENİYOR": 2.0,
                            "KANIT_YETERSİZ": 1.0,
                            "ÇELİŞİYOR": 0.2,
                        }.get(v_res.verdict, 0.5)
                        citation_bonus = min(0.25, (cand_paper.citation_count / 500.0) * 0.25)
                        rank_score = verdict_weight + float(v_res.semantic_similarity) + citation_bonus

                        ranked_items.append({
                            "rank_score": rank_score,
                            "paper": cand_paper,
                            "result": v_res,
                        })

                    ranked_items.sort(key=lambda x: x["rank_score"], reverse=True)
                    st.session_state["tab2_ranked_results"] = ranked_items
                    st.session_state["tab2_last_claim"] = claim_input_t2.strip()
                    st.session_state["tab2_query_meta"] = query_meta_t2
                    status_box.update(
                        label=f"Araştırma tamamlandı! Cümlenize en uygun {len(ranked_items)} akademik makale sıralandı.",
                        state="complete",
                        expanded=False,
                    )

    if st.session_state.get("tab2_ranked_results"):
        ranked_list: List[Dict[str, Any]] = st.session_state["tab2_ranked_results"]
        q_meta: Dict[str, str] = st.session_state.get("tab2_query_meta", {})
        last_claim_str: str = st.session_state.get("tab2_last_claim", "")

        sup_n_t2 = sum(1 for item in ranked_list if item["result"].verdict in ("DESTEKLENİYOR", "KISMEN_DESTEKLENİYOR"))

        st.space("small")
        st.subheader(f":material/menu_book: Cümleniz İçin Sıralanan Akademik Makaleler ({len(ranked_list)} Makale)")
        st.caption(
            f"**Arama Terimleri:** `{q_meta.get('english_query', '-')}`  ·  "
            f"**Cümlenizi Destekleyen / Uyumlu Makale:** `{sup_n_t2}` adet"
        )

        for rank_idx, item in enumerate(ranked_list, start=1):
            fp: FetchedPaper = item["paper"]
            vr: VerificationResult = item["result"]
            cit_fmt = _format_citation_strings(fp, idx=rank_idx)

            if vr.verdict == "DESTEKLENİYOR":
                badge_t2 = "✅ Cümlenizi Tam Destekliyor"
                why_t2 = "Bu makalenin bulguları yazdığınız cümleyi doğrudan destekliyor. Güvenle atıf yapabilirsiniz."
            elif vr.verdict == "KISMEN_DESTEKLENİYOR":
                badge_t2 = "☑️ Cümlenizle Uyumlu (Özet Destekliyor)"
                why_t2 = "Bu makalenin konusu ve yönü cümlenizle örtüşüyor. Kaynak olarak kullanmaya uygundur."
            elif vr.verdict == "ÇELİŞİYOR":
                badge_t2 = "❌ Dikkat: Cümlenizle Çelişiyor"
                why_t2 = "Bu makale aynı konuyu inceliyor ancak cümlenizin tersi yönde bir bulgu bildiriyor."
            else:
                badge_t2 = "🟡 Aynı Konuda Yakın Literatür"
                why_t2 = "Bu makale cümlenizdeki konuyla ilgili ancak özetinde spesifik iddianız doğrudan yer almıyor."

            header_t2 = (
                f"#{rank_idx}  ·  {badge_t2} (Benzerlik: %{vr.semantic_similarity * 100:.0f})  ·  "
                f"{cit_fmt['intext_tr']} — {fp.verified_title[:85]}"
            )

            with st.expander(header_t2, expanded=(rank_idx <= 3)):
                left_c, right_c = st.columns([3, 2])
                with left_c:
                    st.markdown(f"**{fp.verified_title}**")
                    st.caption(
                        f"👥 {', '.join(fp.verified_authors[:6]) or 'Belirtilmemiş'}  ·  "
                        f"📅 {fp.verified_year or '-'}  ·  "
                        f"🏛️ *{fp.verified_venue or '-'}*  ·  "
                        f"📈 {fp.citation_count} Atıf"
                    )

                    if vr.verdict in ("DESTEKLENİYOR", "KISMEN_DESTEKLENİYOR"):
                        st.success(f"**Neden Uygun?** {why_t2}\n\n**Değerlendirme:** {vr.explanation}")
                    elif vr.verdict == "ÇELİŞİYOR":
                        st.error(f"**Neden Uygun Değil?** {why_t2}\n\n**Değerlendirme:** {vr.explanation}")
                    else:
                        st.warning(f"**Durum:** {why_t2}\n\n**Değerlendirme:** {vr.explanation}")

                    if vr.best_evidence:
                        st.info(
                            vr.best_evidence,
                            title=f"📌 Makaledeki En Yakın Kanıt Cümlesi (Benzerlik: %{vr.semantic_similarity * 100:.0f})",
                        )

                    if fp.abstract:
                        with st.expander("📄 Makalenin Tam Özetini (Abstract) Gör", expanded=False):
                            st.write(fp.abstract)

                with right_c:
                    with st.container(border=True):
                        st.markdown("**:material/content_copy: Kopyalamaya Hazır Atıf ve Kaynakça**")
                        st.caption("Sağ üstteki kopyala ikonuna tıklayarak Word dosyanıza yapıştırabilirsiniz:")
                        st.markdown("**Metin İçi Atıf:**")
                        st.code(
                            f"Türkçe (APA): {cit_fmt['intext_tr']}\n"
                            f"İngilizce   : {cit_fmt['intext_en']}\n"
                            f"Numaralı    : {cit_fmt['intext_num']}",
                            language="text",
                        )
                        st.markdown("**APA Kaynakça Künyesi:**")
                        st.code(cit_fmt["apa_ref"], language="text")
                        st.markdown("**Vancouver / Numaralı Künye:**")
                        st.code(cit_fmt["vanc_ref"], language="text")

                        links_md = []
                        if fp.verified_doi:
                            links_md.append(f"[🔗 DOI Sayfası](https://doi.org/{fp.verified_doi})")
                        if fp.pmid:
                            links_md.append(f"[🏥 PubMed](https://pubmed.ncbi.nlm.nih.gov/{fp.pmid}/)")
                        if fp.pdf_url:
                            links_md.append(f"[📄 Ücretsiz PDF]({fp.pdf_url})")
                        if links_md:
                            st.markdown("  ·  ".join(links_md))


# ===========================================================================
# SEKME 3: TOPLU TABLO VE RAPOR İNDİR (EXCEL / WORD)
# ===========================================================================

with tab3:
    if "verification_results" not in st.session_state or not st.session_state["verification_results"]:
        st.info(
            "Henüz bir makale denetimi çalıştırılmadı. Lütfen **1. Makale Denetle** sekmesinden makalenizi yükleyip denetimi başlatın.",
            title="ℹ️ Rapor Hazır Değil",
        )
    else:
        results_list_t3: List[VerificationResult] = st.session_state["verification_results"]
        f_map_t3: Dict[str, FetchedPaper] = st.session_state.get("fetched_map", {})
        cur_p_doc: Optional[ParsedManuscript] = st.session_state.get("current_parsed_doc")

        top_t3_l, top_t3_r = st.columns([2.2, 1.8], vertical_alignment="center")
        with top_t3_l:
            st.subheader(":material/table_view: Toplu Sonuç Tablosu ve Rapor İndirme")
            st.caption("Denetim sonuçlarını filtreleyebilir ve Excel, Word veya JSON formatında indirebilirsiniz.")

        t3_sup = sum(1 for r in results_list_t3 if r.verdict == "DESTEKLENİYOR" and not r.is_retracted)
        t3_part = sum(1 for r in results_list_t3 if r.verdict == "KISMEN_DESTEKLENİYOR" and not r.is_retracted)
        t3_con = sum(1 for r in results_list_t3 if r.verdict == "ÇELİŞİYOR" and not r.is_retracted)
        t3_ins = sum(1 for r in results_list_t3 if r.verdict == "KANIT_YETERSİZ" and not r.is_retracted)
        t3_fake = sum(1 for r in results_list_t3 if r.verdict == "SAHTE_KAYNAK" and not r.is_retracted)
        t3_unk = sum(1 for r in results_list_t3 if r.verdict == "DOGRULANAMADI" and not r.is_retracted)
        t3_ghost = sum(1 for r in results_list_t3 if r.verdict == "HAYALET_ATIF" and not r.is_retracted)
        t3_ret = sum(1 for r in results_list_t3 if r.is_retracted)

        _init_filter_state("f3")

        with st.container(border=True):
            r3_1 = st.columns(5)
            with r3_1[0]:
                st.checkbox(
                    f"📋 Tümünü Göster ({len(results_list_t3)})",
                    key="f3_ALL",
                    on_change=_on_toggle_all,
                    args=("f3",),
                )
            with r3_1[1]:
                st.checkbox(
                    f"✅ Tam Desteklenen ({t3_sup})",
                    key="f3_SUP",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )
            with r3_1[2]:
                st.checkbox(
                    f"☑️ Özet Uyumlu ({t3_part})",
                    key="f3_PART",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )
            with r3_1[3]:
                st.checkbox(
                    f"❌ Çelişen ({t3_con})",
                    key="f3_CON",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )
            with r3_1[4]:
                st.checkbox(
                    f"⚠️ Kanıt Yetersiz ({t3_ins})",
                    key="f3_INS",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )

            r3_2 = st.columns(5)
            with r3_2[0]:
                st.checkbox(
                    f"🚨 Sahte / Hatalı DOI ({t3_fake})",
                    key="f3_FAKE",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )
            with r3_2[1]:
                st.checkbox(
                    f"❓ Doğrulanamadı ({t3_unk})",
                    key="f3_UNK",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )
            with r3_2[2]:
                st.checkbox(
                    f"👻 Hayalet / Eksik ({t3_ghost})",
                    key="f3_GHOST",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )
            with r3_2[3]:
                st.checkbox(
                    f"🛑 Geri Çekilmiş ({t3_ret})",
                    key="f3_RET",
                    on_change=_on_toggle_single,
                    args=("f3",),
                )

        active_codes_f3 = {k for k in ALL_FILTER_KEYS if st.session_state.get(f"f3_{k}", False)}

        rows = []
        md_lines = [
            "# Akademik Makale, Kaynakça ve Atıf Denetim Raporu",
            f"**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n",
            "## Cümle Bazlı Atıf ve Doğrulama Sonuçları\n",
        ]

        for r in results_list_t3:
            if _result_category_code(r) not in active_codes_f3:
                continue
            fp = f_map_t3.get(r.ref_id) if r.ref_id else None
            v_info = get_verdict_info(r.verdict, is_retracted=r.is_retracted)

            rows.append({
                "Atıf İşareti": r.marker,
                "Sonuç Durumu": v_info["short_label"],
                "Neden Bu Sonuç Çıktı?": v_info["why"],
                "Ne Yapmalısınız?": v_info["action"],
                "Makalenizdeki Cümle": r.claim_sentence,
                "Hakem Açıklaması": r.explanation,
                "Önerilen Düzeltilmiş Cümle": r.suggested_revision or "-",
                "Kaynak Makale Başlığı": fp.verified_title if fp else "Kaynakça Listesinde Eksik",
                "Okunan İçerik Kaynağı": fp.content_source if fp else "Kaynakçada Yok",
                "Konu Benzerliği (%)": round(r.semantic_similarity * 100, 1),
                "Künye Doğruluk Oranı (%)": round(r.match_score * 100, 0),
                "Değerlendiren": human_decision_source(r.decision_source),
                "Kaynak Makaledeki Kanıt Paragrafı": r.best_evidence if r.best_evidence else "-",
            })

            md_lines.extend([
                f"### Atıf: `{r.marker}` — **{v_info['badge']}**",
                f"- **Makalenizdeki Cümle:** {r.claim_sentence}",
                f"- **Neden Bu Sonuç Çıktı?:** {v_info['why']}",
                f"- **Detaylı Açıklama:** {r.explanation}",
                f"- **Ne Yapmalısınız?:** {v_info['action']}",
                f"- **Önerilen Düzeltilmiş Cümle:** {r.suggested_revision or '-'}",
                f"- **Eşleşen Kaynak:** {fp.verified_title if fp else 'Bulunamadı'} ({fp.content_source if fp else '-'})",
                f"- **Kanıt Paragrafı:** {r.best_evidence if r.best_evidence else '-'}",
                "",
            ])

        st.caption(
            f"Toplam **{len(results_list_t3)}** atıf içinden filtreye uyan **{len(rows)}** kayıt listeleniyor. "
            "*(Uzun cümlelerin tamamını görmek için ilgili hücreye çift tıklayabilirsiniz).* "
        )
        df_report = pd.DataFrame(rows)
        st.dataframe(df_report, hide_index=True, row_height=50)

        csv_bytes = df_report.to_csv(index=False).encode("utf-8-sig")
        md_bytes = "\n".join(md_lines).encode("utf-8")

        st.caption("💡 **İndirme Seçenekleri:** Sonuçları Excel tablosu (`.csv`), Word uyumlu rapor (`.md`) veya tekrar siteye yüklenebilir analiz paketi (`.json`) olarak kaydedin:")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                ":material/table: Excel Tablosu İndir (.csv)",
                data=csv_bytes,
                file_name="makale_atif_denetim_raporu.csv",
                mime="text/csv",
                width="stretch",
                help="Tüm denetim tablosunu Excel'de açabileceğiniz CSV formatında indirir.",
            )
        with dl_col2:
            st.download_button(
                ":material/description: Word / Metin Raporu İndir (.md)",
                data=md_bytes,
                file_name="makale_atif_denetim_raporu.md",
                mime="text/markdown",
                width="stretch",
                help="Her bir atıfın detaylı hakem değerlendirmesini yazılı rapor olarak indirir.",
            )
        with dl_col3:
            if cur_p_doc is not None:
                s_bytes_t3 = serialize_session_bytes(cur_p_doc, f_map_t3, results_list_t3)
                s_stem_t3 = safe_ref_token(Path(cur_p_doc.filename).stem, fallback="makale")
                st.download_button(
                    ":material/save: Analiz Kaydını İndir (.json)",
                    data=s_bytes_t3,
                    file_name=f"{s_stem_t3}_analiz_kaydi.json",
                    mime="application/json",
                    type="primary",
                    width="stretch",
                    help="Bu analiz oturumunu daha sonra 1. Sekmeden yükleyip kaldığınız yerden devam etmek için indirir.",
                )
