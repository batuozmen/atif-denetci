r"""
MODÜL 4: Görsel TÜBİTAK Akademik Makale & Semantik Atıf Denetim Arayüzü (app.py)
--------------------------------------------------------------------------------
Çalıştırmak için:
    .\.venv\Scripts\streamlit.exe run app.py
veya klasördeki `baslat.bat` dosyasına çift tıklayın.
"""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

import document_parser
import academic_fetcher
import rag_verifier

importlib.reload(document_parser)
importlib.reload(academic_fetcher)
importlib.reload(rag_verifier)

from document_parser import DocumentParser, ParsedManuscript, ReferenceItem, InTextCitation
from academic_fetcher import AcademicFetcher, FetchedPaper
from rag_verifier import SemanticRAGVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Sayfa Yapılandırması
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TÜBİTAK Akademik Atıf ve Kaynakça Denetçisi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
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


# ---------------------------------------------------------------------------
# Anlaşılır Terim ve Karar Açıklama Yardımcıları
# ---------------------------------------------------------------------------

def human_decision_source(source_code: str) -> str:
    """Teknik karar kaynağı kodunu anlaşılır Türkçeye çevirir."""
    mapping = {
        "llm": "Yapay Zeka Hakemi",
        "kural": "Sayısal / Dilbilgisel Kural Kontrolü",
        "kural+llm": "Yapay Zeka + Sayısal Kural Kontrolü",
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
            "badge": "🛑 DÜZELTİLMELİ — Yayından Geri Çekilmiş Makale",
            "short_label": "Yayından Geri Çekilmiş (Retracted)",
            "why": "Atıf yaptığınız bu makale, bilimsel hata veya etik ihlal nedeniyle dergi tarafından resmi olarak geri çekilmiştir.",
            "action": "Bu kaynağı makalenizden çıkarıp yerine geçerli ve güncel bir makale eklemelisiniz.",
        }

    info_map = {
        "DESTEKLENİYOR": {
            "group": "ok",
            "badge": "✅ SORUNSUZ — Kaynak Makaleyle Tam Uyumlu",
            "short_label": "Tam Uyumlu (Doğrulandı)",
            "why": "Makalenizde yazdığınız cümledeki bilgi, referans gösterdiğiniz makalenin içeriğiyle doğrudan örtüşüyor.",
            "action": "Herhangi bir düzeltme yapmanıza gerek yoktur.",
        },
        "KISMEN_DESTEKLENİYOR": {
            "group": "ok",
            "badge": "☑️ SORUNSUZ — Makale Özetiyle Uyumlu (Tam Metin Kilitli)",
            "short_label": "Özetle Uyumlu (Tam Metin Ücretli/Kilitli)",
            "why": "Kaynak makale internette doğrulandı ve özeti (abstract) sizin cümlenizin konusuyla uyumlu. Ancak yayıncı tam metin PDF'ini kilitli tuttuğu için kontrol özet üzerinden yapıldı.",
            "action": "Genel olarak uyumludur. Cümlenizde çok özel bir sayı/tablo detayı varsa ve makalenin PDF'i bilgisayarınızda mevcutsa sol menüden yükleyerek tam metin kontrolü de yaptırabilirsiniz.",
        },
        "ÇELİŞİYOR": {
            "group": "error",
            "badge": "❌ DÜZELTİLMELİ — Kaynak Makaleyle Çelişiyor",
            "short_label": "Kaynak Makaleyle Çelişen Bilgi",
            "why": "Cümlenizdeki bilgi (sayısal oran, artış/azalış yönü veya olumlu/olumsuz ifade) kaynak makaledeki bulgunun tam tersini söylüyor.",
            "action": "Cümlenizi aşağıdaki 'Önerilen Düzeltilmiş Cümle'ye göre güncelleyin veya yanlış kaynağa atıf yaptıysanız doğru kaynağı gösterin.",
        },
        "HAYALET_ATIF": {
            "group": "error",
            "badge": "🚨 DÜZELTİLMELİ — Kaynakça Listesinde Eksik",
            "short_label": "Kaynakça Listesinde Eksik Atıf",
            "why": "Makale metni içinde bu kaynağa atıf yapmışsınız ancak makalenin sonundaki 'Kaynakça / References' listesine bu makaleyi yazmamışsınız.",
            "action": "Bu kaynağın tam künyesini makalenizin sonundaki Kaynakça listesine ekleyin veya metin içindeki atıfı silin.",
        },
        "SAHTE_KAYNAK": {
            "group": "error",
            "badge": "🚨 DÜZELTİLMELİ — Hatalı / Uyuşmayan Kaynak Künyesi",
            "short_label": "Hatalı Künye (DOI Başka Makaleye Ait)",
            "why": "Kaynakçada yazdığınız DOI numarası veya künye bilgisi internette bambaşka bir makaleye ait görünüyor (kopyala-yapıştır hatası veya uydurma kaynak olabilir).",
            "action": "Kaynakçadaki makale başlığını ve DOI numarasını kontrol edip gerçek makale künyesiyle düzeltin.",
        },
        "KANIT_YETERSİZ": {
            "group": "manual",
            "badge": "⚠️ KONTROL EDİLMELİ — Makalede Bu Bilgiye Rastlanmadı",
            "short_label": "Makalede Bu Bilgi Bulunamadı",
            "why": "Kaynak makale internette bulundu ancak erişilebilen metninde sizin cümlenizdeki iddiayı doğrulayan veya çürüten bir ifade yer almıyor (bilgi kilitli tam metin sayfalarında veya tablolarda geçiyor olabilir).",
            "action": "Atıfı doğru makaleye yaptığınızdan emin olmak için ilgili kaynağı gözle kontrol edin. Makalenin tam PDF'i elinizdeyse sol menüden yükleyip tekrar denetleyebilirsiniz.",
        },
        "DOGRULANAMADI": {
            "group": "manual",
            "badge": "❓ KONTROL EDİLMELİ — Makaleye İnternette Erişilemedi",
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
    with st.expander("📖 Ekrandaki Terimler ve Yüzdeler Ne Anlama Geliyor? (Açıklamalı Rehber)", expanded=False):
        st.markdown(
            """
#### 1) Renk Gruplarının Anlamı Nedir?
- 🟢 **Sorunsuz / Doğrulanan Atıflar:** Kaynak makale internette bulundu ve cümlenizdeki bilgi makaleyle uyumlu. Değişiklik yapmanıza gerek yoktur.
- 🔴 **Düzeltilmesi Gereken Hatalar:** Cümleniz kaynak makaleyle **çelişiyor**, atıf yaptığınız kaynak **Kaynakça listesinde eksik**, **DOI numarası başka makaleye ait** veya makale **yayından geri çekilmiş**. Makalenizi göndermeden önce mutlaka düzeltmelisiniz.
- 🟡 **Sizin Kontrol Etmeniz Gerekenler:** Sistem makaleye internette ulaşamadı (kitap/tez olabilir) veya makalenin açık özetinde cümlenizdeki spesifik detayı göremedi. Hata olmak zorunda değildir; sizin gözle kontrol etmeniz önerilir.

---

#### 2) Yüzde (%) Oranları Ne Demektir?
- **Konu Benzerliği (%):** Sizin cümleniz ile referans makaledeki paragrafın **aynı konudan** bahsetme oranıdır.
  - *Önemli Not:* Konu benzerliğinin yüksek olması cümlenizin doğru olduğu anlamına **gelmez**. Örneğin *"İlaç ağrıyı azaltır"* ile *"İlaç ağrıyı azaltmaz"* cümlelerinin konusu %90 aynıdır ama anlamları tam terstir. Sistem bu oranı sadece makale içindeki ilgili paragrafı bulmak için kullanır.
- **Künye Doğruluk Oranı (%):** İnternette bulunan makalenin, sizin kaynakçada yazdığınız makale olduğundan ne kadar emin olduğumuzu gösterir.
  - **%100:** DOI numarası birebir eşleşti (kesin doğru makale).
  - **%90 – %99:** Makale başlığı ve yazarları birebir eşleşti.
  - **%70 altı:** Benzerlik zayıf olduğu için sistem yanlış makaleyle kıyaslama yapmamak adına sonucu **"Kontrol Edilmeli"** olarak işaretler.
"""
        )


def run_verification_pipeline(
    citations_to_check: List[InTextCitation],
    selected_model: str,
    enable_scihub: bool,
    enable_jina: bool,
    enable_suggestions: bool,
) -> tuple[List[VerificationResult], Dict[str, FetchedPaper]]:
    """Seçilen atıfların kaynaklarını internetten çeker ve anlam doğrulamasını çalıştırır."""
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
                    text=f"🌐 1/2: Kaynak kontrol ediliyor [{done_r}/{total_r}]: {(r_it.title or r_it.raw_text)[:65]}...",
                )

    results: List[VerificationResult] = []
    total_c = len(citations_to_check)
    for idx_c, cit in enumerate(citations_to_check, start=1):
        prog.progress(
            50 + int((idx_c / max(1, total_c)) * 50),
            text=f"🧠 2/2: Cümle ve kaynak uyumu inceleniyor [{idx_c}/{total_c}]: {cit.marker}...",
        )
        if cit.matched_ref_id is None or cit.matched_ref_id not in fetched_map:
            alt_list = fetcher.suggest_real_papers(cit.claim_sentence, top_k=3) if enable_suggestions else []
            results.append(
                VerificationResult(
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
            results.append(v_res)

    prog.progress(100, text=f"✅ Denetim tamamlandı! Toplam {total_c} atıf incelendi.")
    return results, fetched_map


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
# Sol Yan Menü (Sidebar): Sade ve Anlaşılır Düzen
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎓 Kontrol Paneli")

    temp_verifier = SemanticRAGVerifier(embedding_device="cpu")
    ollama_ok, default_model, installed_models = temp_verifier.check_ollama_status()

    if ollama_ok and installed_models:
        st.success(
            "Yapay Zeka Hakemi ve İnternet Tarayıcısı kullanıma hazır.",
            title="🟢 Sistem Hazır",
        )
    elif ollama_ok and not installed_models:
        st.warning(
            "Ollama açık ancak dil modeli indirilmemiş. Kural tabanlı kontrol aktif.",
            title="🟡 Kısmi Mod",
        )
    else:
        st.info(
            "Yerel dil modeli (Ollama) kapalı; sistem otomatik olarak Sayısal ve Çok Dilli Kural Motoru ile çalışıyor.",
            title="ℹ️ Otomatik Kural Modu Aktif",
        )

    st.subheader("📂 Kaynak PDF'leriniz (İsteğe Bağlı)")
    st.caption(
        "İnternette tam metni ücretli/kilitli olan kaynak makalelerin PDF dosyaları bilgisayarınızda varsa "
        "buraya yükleyin. Sistem sadece özet yerine doğrudan yüklediğiniz tam metni okur."
    )
    uploaded_ref_pdfs = st.file_uploader(
        "Referans makale PDF dosyalarını seçin:",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_ref_pdfs:
        for up_pdf in uploaded_ref_pdfs:
            target_path = USER_PDF_DIR / up_pdf.name
            target_path.write_bytes(up_pdf.getvalue())
        st.success(f"📎 {len(uploaded_ref_pdfs)} adet kaynak PDF dosyası sisteme eklendi.")

    st.divider()
    st.markdown(
        """
**📌 Renklerin Kısa Anlamı:**
- 🟢 **Sorunsuz:** Kaynak ve cümle uyumlu.
- 🔴 **Düzeltilmeli:** Çelişki, eksik kaynakça veya hatalı DOI var.
- 🟡 **Kontrol Edilmeli:** Makaleye internette ulaşılamadı veya bilgi özet dışında bir sayfada.
"""
    )

    with st.expander("⚙️ Gelişmiş Teknik Ayarlar", expanded=False):
        if ollama_ok and installed_models:
            selected_model = st.selectbox(
                "Yapay Zeka Dil Modeli:",
                options=installed_models,
                index=installed_models.index(default_model) if default_model in installed_models else 0,
            )
        else:
            selected_model = default_model or "qwen2.5:3b"
            st.caption(f"Varsayılan model: `{selected_model}`")

        enable_suggestions = st.checkbox(
            "Hatalı/eksik kaynaklar için alternatif gerçek makale öner",
            value=True,
        )
        enable_jina = st.checkbox(
            "Web sayfalarını genişletilmiş okuyucu (Jina) ile tara",
            value=True,
        )
        enable_scihub = st.checkbox(
            "Sci-Hub tam metin PDF indiriciyi kullan (varsayılan kapalı)",
            value=False,
            help=(
                "Hukuki uyarı: Sci-Hub üzerinden telifli makale indirmek birçok ülkede telif hakkı "
                "ihlali sayılır. Varsayılan olarak kapalıdır; açtığınızda sorumluluk kullanıcıya aittir."
            ),
        )


# ---------------------------------------------------------------------------
# Ana Başlık ve "Neyin Nerede Olduğu" 3 Adımlı Yol Haritası
# ---------------------------------------------------------------------------

st.title("🎓 Akademik Makale, Kaynakça ve Atıf Doğrulama Sistemi")
st.markdown(
    "Makalenizdeki **atıf–kaynakça uyumunu**, kaynakların **internetteki gerçekliğini** ve "
    "yazdığınız cümlelerin **kaynak makaleyle çelişip çelişmediğini** anlaşılır şekilde kontrol eder."
)

step1_col, step2_col, step3_col = st.columns(3)
with step1_col.container(border=True, height="stretch"):
    st.markdown("**1️⃣ Adım: Makalenizi Yükleyin**")
    st.caption(
        "Word (`.docx`), PDF (`.pdf`) veya metin dosyanızı yükleyin. "
        "**Kaynakçada unutulan atıflar** ve **metinde hiç kullanılmayan kaynaklar** anında listelenir."
    )
with step2_col.container(border=True, height="stretch"):
    st.markdown("**2️⃣ Adım: Doğrulamayı Başlatın**")
    st.caption(
        "İster **tüm makaleyi**, ister **sadece seçtiğiniz atıfları** (örneğin ilk 10 atıfı) "
        "internetten taratıp cümlelerinizin doğruluğunu kontrol edin."
    )
with step3_col.container(border=True, height="stretch"):
    st.markdown("**3️⃣ Adım: Hataları Düzeltin & İndirin**")
    st.caption(
        "**Düzeltilmesi gereken** ve **kontrol edilmesi gereken** atıfları nedenleriyle görün; "
        "sonuçları Excel veya Word uyumlu rapor olarak indirin."
    )

tab1, tab2, tab3 = st.tabs([
    "📄 1. Makale Yükle ve Denetle (Ana Ekran)",
    "✍️ 2. Hızlı Tek Cümle Testi (Dosya Yüklemeden)",
    "📊 3. Toplu Tablo ve Rapor İndir (Excel / Word)",
])


# ===========================================================================
# SEKME 1: MAKALE YÜKLE VE DENETLE (ANA EKRAN)
# ===========================================================================

with tab1:
    st.subheader("1️⃣ Makale Dosyanızı Yükleyin")
    col_up, col_demo = st.columns([3, 1], vertical_alignment="bottom")
    with col_up:
        uploaded_manuscript = st.file_uploader(
            "Makalenizi buradan seçin (.docx, .pdf veya .txt):",
            type=["docx", "pdf", "txt"],
        )
    with col_demo:
        load_demo = st.button("🧪 Örnek Makaleyle Dene", width="stretch")

    if load_demo:
        st.session_state["manuscript_text"] = DEMO_MANUSCRIPT_TEXT
        st.session_state["manuscript_name"] = "ornek_tubitak_makalesi.txt"

    parser = get_parser()
    parsed_doc: ParsedManuscript | None = None

    if uploaded_manuscript is not None:
        parsed_doc = parser.parse_file(
            uploaded_manuscript.getvalue(), filename=uploaded_manuscript.name
        )
    elif "manuscript_text" in st.session_state:
        parsed_doc = parser.parse_text(
            st.session_state["manuscript_text"],
            filename=st.session_state.get("manuscript_name", "demo_makale.txt"),
        )

    if parsed_doc is not None:
        # Yeni makale yüklendiyse seçim ve sonuç önbelleğini sıfırla
        if st.session_state.get("active_doc_name") != parsed_doc.filename:
            st.session_state["active_doc_name"] = parsed_doc.filename
            st.session_state["selected_cit_indices"] = set(range(1, min(6, len(parsed_doc.citations) + 1)))
            st.session_state["editor_ver"] = 0
            st.session_state["fetched_map"] = {}
            st.session_state.pop("verification_results", None)

        st.session_state["current_parsed_doc"] = parsed_doc

        st.divider()
        st.subheader(f"2️⃣ Biçimsel Kaynakça ve Atıf Kontrolü — `{parsed_doc.filename}`")
        st.caption(
            "Bu bölüm internet araması yapmadan, doğrudan makalenizin kendi içindeki "
            "**Metin İçi Atıflar** ile **Kaynakça Listesi** arasındaki uyumu gösterir."
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1.container(border=True, height="stretch"):
            st.metric("📚 Kaynakçadaki Makale Sayısı", len(parsed_doc.references))
            st.caption("Makale sonundaki Kaynakça listesinde bulunan toplam referans sayısı.")
        with b2.container(border=True, height="stretch"):
            st.metric("📌 Metin İçi Atıf Sayısı", len(parsed_doc.citations))
            st.caption("Makale cümleleri içinde parantez veya numarayla yapılan toplam atıf sayısı.")
        with b3.container(border=True, height="stretch"):
            ghost_n = len(parsed_doc.ghost_citations)
            st.metric(
                "🚨 Kaynakçada Eksik Atıf",
                ghost_n,
                delta="Düzeltilmeli" if ghost_n else "Sorun Yok",
                delta_color="inverse" if ghost_n else "normal",
            )
            st.caption("Metin içinde atıf yapılmış ama Kaynakça listesine yazılmamış kaynaklar.")
        with b4.container(border=True, height="stretch"):
            orphan_n = len(parsed_doc.orphan_references)
            st.metric(
                "⚠️ Metinde Kullanılmayan Kaynak",
                orphan_n,
                delta="Düzeltilmeli" if orphan_n else "Sorun Yok",
                delta_color="inverse" if orphan_n else "normal",
            )
            st.caption("Kaynakça listesinde yazılı olan ama metinde hiçbir cümlede kullanılmamış kaynaklar.")

        if parsed_doc.ghost_citations or parsed_doc.orphan_references:
            with st.container(border=True):
                st.markdown("#### 🔧 Biçimsel Olarak Düzeltmeniz Gereken Kaynakça Sorunları")
                if parsed_doc.ghost_citations:
                    st.error(
                        f"Makalenizde **{len(parsed_doc.ghost_citations)} adet Kaynakçada Eksik Atıf** bulundu. "
                        "Çünkü metin içinde bu kaynaklara atıf yapmışsınız ancak makale sonundaki Kaynakça listesine eklememişsiniz:",
                        title="🚨 Kaynakça Listesine Eklenmesi Gerekenler",
                    )
                    for gc in parsed_doc.ghost_citations:
                        st.markdown(f"- **Eksik Atıf:** `{gc.marker}` — *Geçtiği Cümle:* \"{gc.claim_sentence}\"")

                if parsed_doc.orphan_references:
                    st.warning(
                        f"Kaynakça listenizde **{len(parsed_doc.orphan_references)} adet Metinde Kullanılmayan Kaynak** bulundu. "
                        "Çünkü bu kaynakları Kaynakça bölümüne yazmışsınız ancak makale metninde hiçbir cümlede atıf yapmamışsınız "
                        "(metin içinde uygun yere atıf ekleyin veya kaynakçadan çıkarın):",
                        title="⚠️ Metin İçinde Hiç Kullanılmayan Kaynaklar",
                    )
                    for orp in parsed_doc.orphan_references:
                        st.markdown(f"- **[{orp.ref_id}]** {orp.raw_text}")
        else:
            st.success(
                "Makalenizdeki tüm metin içi atıflar Kaynakça listesiyle eksiksiz eşleşti "
                "(Kaynakçada eksik atıf yok, metinde kullanılmayan kaynak yok).",
                title="✅ Biçimsel Kaynakça Kontrolü Tamamen Temiz",
            )

        with st.expander(
            f"📋 Hangi Kaynağın Hangi Cümlede Kullanıldığını Gör ({len(parsed_doc.references)} Kaynak Eşleşme Listesi)",
            expanded=False,
        ):
            st.caption(
                "Kaynakçanızdaki her bir makalenin metin içinde kaç kez kullanıldığını ve yakalandığı örnek cümleyi aşağıda görebilirsiniz."
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
                    "İlk Yazar ve Yıl": f"{r_item.first_author_surname or 'Bilinmiyor'} ({r_item.year or '?'})",
                    "Kaç Kez Kullanıldı": f"{r_item.cited_in_text_count} kez",
                    "Metindeki İşareti": markers_str,
                    "Kaynak Başlığı": (r_item.title or r_item.raw_text)[:95],
                    "Geçtiği Örnek Cümle": first_sent[:180],
                })
            st.dataframe(pd.DataFrame(proof_rows), hide_index=True)

        # -------------------------------------------------------------------
        # ADIM 3: İNTERNET VE İÇERİK DOĞRULAMASI (TAM VEYA SEÇMELİ)
        # -------------------------------------------------------------------
        st.divider()
        st.subheader("3️⃣ Kaynakların Gerçekliğini ve Cümle Uyumu Denetimini Başlatın")
        st.caption(
            "Burada kaynaklarınız internet veritabanlarından (CrossRef, PubMed, OpenAlex vb.) bulunur "
            "ve makalenizde yazdığınız her bir cümlenin kaynak makaleyle **uyumlu olup olmadığı / çelişip çelişmediği** kontrol edilir."
        )

        all_cits = parsed_doc.citations
        total_cits_n = len(all_cits)

        scope_choice = st.radio(
            "Hangi atıfları denetlemek istiyorsunuz?",
            options=[
                f"🚀 Tüm Makaleyi Denetle ({total_cits_n} atıfın tamamı)",
                "🎯 Sadece Seçtiğim Atıfları veya Belirli Bir Aralığı Denetle (Örn: İlk 5 veya 1–20 arası — Daha Hızlı)",
            ],
            horizontal=True,
        )

        citations_to_run: List[InTextCitation] = []

        if scope_choice.startswith("🚀"):
            citations_to_run = all_cits
            needed_unique_refs = len({c.matched_ref_id for c in citations_to_run if c.matched_ref_id})
            run_btn = st.button(
                f"🚀 Tüm Makaleyi Denetle ({len(citations_to_run)} Atıf / {needed_unique_refs} Kaynak Makale)",
                type="primary",
                width="stretch",
                disabled=(len(citations_to_run) == 0),
            )
        else:
            if "selected_cit_indices" not in st.session_state:
                st.session_state["selected_cit_indices"] = set(range(1, min(6, total_cits_n + 1)))
            if "editor_ver" not in st.session_state:
                st.session_state["editor_ver"] = 0

            with st.container(border=True):
                st.markdown("**Hızlı Aralık Seçimi veya Listeden İşaretleme:**")
                ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.2, 1.2, 1.8, 1.4, 1.4], vertical_alignment="bottom")
                with ctrl1:
                    r_start = st.number_input("Başlangıç No:", min_value=1, max_value=max(1, total_cits_n), value=1, step=1)
                with ctrl2:
                    r_end = st.number_input(
                        "Bitiş No:",
                        min_value=1,
                        max_value=max(1, total_cits_n),
                        value=min(10, max(1, total_cits_n)),
                        step=1,
                    )
                with ctrl3:
                    if st.button(f"🔢 {min(r_start, r_end)} – {max(r_start, r_end)} Arasını Seç", width="stretch"):
                        s_i, e_i = min(int(r_start), int(r_end)), max(int(r_start), int(r_end))
                        st.session_state["selected_cit_indices"] = set(range(s_i, e_i + 1))
                        st.session_state["editor_ver"] += 1
                        st.rerun()
                with ctrl4:
                    if st.button(f"✅ Tümünü Seç ({total_cits_n})", width="stretch"):
                        st.session_state["selected_cit_indices"] = set(range(1, total_cits_n + 1))
                        st.session_state["editor_ver"] += 1
                        st.rerun()
                with ctrl5:
                    if st.button("🧹 Seçimi Temizle", width="stretch"):
                        st.session_state["selected_cit_indices"] = set()
                        st.session_state["editor_ver"] += 1
                        st.rerun()

                current_sel_set = st.session_state["selected_cit_indices"]
                editor_rows = []
                for idx_c, c_obj in enumerate(all_cits, start=1):
                    ref_obj = c_obj.matched_reference
                    ref_summary = (
                        f"[{c_obj.matched_ref_id}] {ref_obj.first_author_surname or ''} ({ref_obj.year or '?'}) — {(ref_obj.title or ref_obj.raw_text)[:75]}"
                        if ref_obj
                        else "🚨 Kaynakça Listesinde Eksik"
                    )
                    editor_rows.append({
                        "Seç": idx_c in current_sel_set,
                        "Atıf No": idx_c,
                        "Atıf İşareti": c_obj.marker,
                        "Eşleşen Kaynak": ref_summary,
                        "Makalenizdeki Cümle": c_obj.claim_sentence,
                    })

                edited_df = st.data_editor(
                    pd.DataFrame(editor_rows),
                    key=f"cit_selector_{parsed_doc.filename}_{st.session_state['editor_ver']}",
                    hide_index=True,
                    height=280,
                    disabled=["Atıf No", "Atıf İşareti", "Eşleşen Kaynak", "Makalenizdeki Cümle"],
                    column_config={
                        "Seç": st.column_config.CheckboxColumn("☑️ Seç", default=False),
                        "Atıf No": st.column_config.NumberColumn("No", width="small"),
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

            run_btn = st.button(
                f"🚀 Seçilen {len(citations_to_run)} Atıfı ({needed_unique_refs} Kaynak Makale) Denetle",
                type="primary",
                width="stretch",
                disabled=(len(citations_to_run) == 0),
            )

        if run_btn and citations_to_run:
            results_out, fetched_out = run_verification_pipeline(
                citations_to_check=citations_to_run,
                selected_model=selected_model,
                enable_scihub=enable_scihub,
                enable_jina=enable_jina,
                enable_suggestions=enable_suggestions,
            )
            st.session_state["verification_results"] = results_out
            st.session_state["fetched_map"] = fetched_out

        # -------------------------------------------------------------------
        # ADIM 4: DENETİM SONUÇLARI (3 AÇIKLAMALI ÖZET KUTUSU + FİLTRE + KARTLAR)
        # -------------------------------------------------------------------
        if "verification_results" in st.session_state and st.session_state["verification_results"]:
            results_list: List[VerificationResult] = st.session_state["verification_results"]
            f_map: Dict[str, FetchedPaper] = st.session_state.get("fetched_map", {})

            st.divider()
            st.subheader(f"4️⃣ Denetim Sonuçları Özeti ({len(results_list)} Atıf İncelendi)")

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
                st.metric("🟢 Sorunsuz / Doğrulanan Atıflar", f"{total_ok} Atıf")
                st.markdown("**Neden sorunsuz sayıldı?**")
                st.markdown(
                    f"- ✅ **{cnt_sup} Tam Uyumlu:** Cümlenizdeki bilgi kaynak makaleyle doğrudan örtüşüyor.\n"
                    f"- ☑️ **{cnt_part} Özetle Uyumlu:** Kaynak makale gerçek ve özeti cümlenizle uyumlu "
                    f"*(tam metin PDF'i ücretli/kilitli olduğu için özet üzerinden doğrulandı)*."
                )
                st.caption("👉 **Ne yapmalısınız?** Bu atıflar için hiçbir değişiklik yapmanıza gerek yoktur.")

            with sum_col2.container(border=True, height="stretch"):
                st.metric(
                    "🔴 Düzeltilmesi Gereken Hatalar",
                    f"{total_error} Atıf",
                    delta="Mutlaka Düzeltilmeli" if total_error > 0 else "Hata Yok",
                    delta_color="inverse" if total_error > 0 else "normal",
                )
                st.markdown("**Neden düzeltilmesi gerekiyor?**")
                st.markdown(
                    f"- ❌ **{cnt_con} Çelişen Bilgi:** Cümlenizdeki sayı, yön veya yargı kaynak makaledeki bulgunun tersini söylüyor.\n"
                    f"- 🚨 **{cnt_ghost} Kaynakçada Eksik Atıf:** Metinde atıf yapılmış ama Kaynakça listesine yazılmamış.\n"
                    f"- 🚨 **{cnt_fake} Hatalı Künye / DOI:** Yazılan DOI numarası başka bir makaleye ait.\n"
                    f"- 🛑 **{cnt_ret} Geri Çekilmiş Makale:** Makale dergi tarafından yayından kaldırılmış (Retracted)."
                )
                st.caption("👉 **Ne yapmalısınız?** Aşağıdaki listeden bu cümleleri açıp önerilen düzeltmeleri uygulayın.")

            with sum_col3.container(border=True, height="stretch"):
                st.metric(
                    "🟡 Sizin Kontrol Etmeniz Gerekenler",
                    f"{total_manual} Atıf",
                    delta="Gözle Kontrol Önerilir" if total_manual > 0 else "Kontrol Gerekmiyor",
                    delta_color="off",
                )
                st.markdown("**Neden manuel kontrol gerekiyor?**")
                st.markdown(
                    f"- ⚠️ **{cnt_ins} Makalede Bu Bilgi Bulunamadı:** Makale bulundu ama açık özetinde cümlenizdeki spesifik detay yer almıyor *(bilgi kilitli tam metin sayfalarında olabilir)*.\n"
                    f"- ❓ **{cnt_unk} İnternette Erişilemedi:** Kaynak veritabanlarında bulunamadı, bağlantı zaman aşımına uğradı veya künye benzerliği düşük kaldı *(kitap/tez olabilir, sahte demek değildir)*."
                )
                st.caption("👉 **Ne yapmalısınız?** Kaynağın doğruluğunu gözle kontrol edin veya PDF'i varsa sol menüden yükleyin.")

            render_reading_guide()

            # 9 TİKLİ (TÜMÜNÜ GÖSTER + 8 SAYAÇ) FİLTRELEME MERKEZİ
            st.markdown("#### 🔍 Cümle Bazlı Detaylı İnceleme ve Filtreleme")
            st.caption(
                "Hangi atıfları görmek istiyorsanız yanındaki kutucukları işaretleyin (birden fazla işaretleyebilirsiniz). "
                "Yalnızca işaretli kutucuklara uyan atıflar aşağıda listelenir."
            )

            _init_filter_state("f1")

            with st.container(border=True):
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
                f"📌 Toplam **{len(results_list)}** atıf içinden işaretlediğiniz tiklere uyan **{len(filtered_results)}** atıf gösteriliyor."
            )

            if not filtered_results:
                st.info("İşaretlediğiniz tiklere uyan herhangi bir atıf bulunmuyor.")

            for r in filtered_results:
                v_info = get_verdict_info(r.verdict, is_retracted=r.is_retracted)
                header_text = f"{v_info['badge']} | Atıf: {r.marker} — \"{r.claim_sentence[:90]}...\""
                is_open_default = v_info["group"] in ("error", "manual")

                with st.expander(header_text, expanded=is_open_default):
                    c_left, c_right = st.columns([3, 2])
                    with c_left:
                        st.markdown(f"**📌 1. Sizin Makalenizdeki Cümle:**\n> {r.claim_sentence}")

                        with st.container(border=True):
                            st.markdown(f"**🧐 2. Neden Bu Sonuç Çıktı?**\n{v_info['why']}")
                            st.markdown(f"**🧑‍⚖️ Detaylı Değerlendirme:** {r.explanation}")
                            st.markdown(f"**🛠️ Ne Yapmalısınız?** {v_info['action']}")

                        if r.suggested_revision:
                            st.success(
                                f"Makalenize doğrudan kopyalayıp yapıştırabileceğiniz düzeltilmiş cümle:\n\n"
                                f"**\"{r.suggested_revision}\"**",
                                title="✏️ Önerilen Düzeltilmiş Akademik Cümle",
                            )

                        if r.best_evidence:
                            st.info(
                                f"{r.best_evidence}",
                                title=f"📖 3. Kaynak Makalede Bulunan İlgili Paragraf (Konu Benzerliği: %{r.semantic_similarity * 100:.0f})",
                            )

                        if r.is_secondary_citation and r.secondary_citation_note:
                            st.warning(r.secondary_citation_note, title="ℹ️ İkincil Atıf Uyarısı")

                    with c_right:
                        with st.container(border=True):
                            st.markdown("**🌐 Kaynak Makalenin Künye Bilgileri**")
                            if r.ref_id and r.ref_id in f_map:
                                fp = f_map[r.ref_id]
                                st.markdown(f"- **Başlık:** {fp.verified_title or fp.query_title}")
                                st.markdown(f"- **Yazarlar:** {', '.join(fp.verified_authors[:4]) or '-'}")
                                st.markdown(f"- **Yıl ve Dergi:** {fp.verified_year or '-'} | {fp.verified_venue or '-'}")
                                st.markdown(f"- **DOI Numarası:** `{fp.verified_doi or 'Belirtilmemiş'}`")
                                st.markdown(f"- **Nereden Bulundu:** `{fp.content_source}`")
                                st.markdown(
                                    f"- **Künye Doğruluk Oranı:** "
                                    f"{f'%{fp.title_match_score * 100:.0f}' if fp.exists_online else 'Doğrulanamadı'}"
                                )
                                st.markdown(f"- **Değerlendiren:** {human_decision_source(r.decision_source)}")
                                if fp.local_pdf_path:
                                    st.success(f"📄 Tam Metin PDF Okundu: `{fp.local_pdf_path}`")
                                for w in fp.metadata_warnings:
                                    st.warning(w)
                            else:
                                st.error("Bu atıf makale sonundaki Kaynakça listesinde bulunmuyor.")

                        if r.alternative_papers:
                            with st.container(border=True):
                                st.markdown("**💡 Bu Cümleyi Destekleyen Gerçek Makale Önerileri:**")
                                for alt in r.alternative_papers:
                                    st.markdown(
                                        f"- **{alt.title}** ({alt.authors}, {alt.year}) — *{alt.venue}* "
                                        f"[DOI: `{alt.doi}` | Atıf: {alt.citation_count}]"
                                    )

            # KAYNAKÇA LİSTESİ ÖZET TABLOSU
            with st.expander("📚 Kaynakça Listesindeki Makalelerin İnternet Doğrulama Tablosu", expanded=False):
                st.caption("Kaynakçanızdaki her bir makalenin internette bulunup bulunmadığını ve künye doğruluk oranını gösterir.")
                ref_rows = []
                for r_id, fp in f_map.items():
                    ref_rows.append({
                        "Kaynak No": f"[{r_id}]",
                        "İnternet Durumu": (
                            "🛑 Yayından Geri Çekilmiş (Retracted)" if fp.is_retracted
                            else ("✅ İnternette Doğrulandı" if fp.exists_online
                                  else ("❓ Bağlantı Hatası (Kontrol Edin)" if fp.lookup_status == "error"
                                        else "❓ Veritabanlarında Bulunamadı"))
                        ),
                        "Makale Başlığı": fp.verified_title or fp.query_title,
                        "Yazarlar": ", ".join(fp.verified_authors[:3]) if fp.verified_authors else "-",
                        "Yıl": fp.verified_year or "-",
                        "Okunan İçerik": fp.content_source,
                        "Künye Doğruluk Oranı": f"%{fp.title_match_score * 100:.0f}" if fp.exists_online else "-",
                        "Not / Uyarı": fp.lookup_error or (" | ".join(fp.metadata_warnings) if fp.metadata_warnings else "Sorun Yok"),
                    })
                st.dataframe(pd.DataFrame(ref_rows), hide_index=True)

        # İsteğe Bağlı Sistem Testi (En altta kapalı kutuda, kafa karıştırmaz)
        with st.expander("🧪 Sistemin Hataları Yakaladığını Test Et (İsteğe Bağlı Deneme Aracı)", expanded=False):
            st.caption(
                "Bu buton, yüklediğiniz makale dosyasına zarar vermeden hafızada geçici olarak "
                "1 adet 'Kaynakçada Eksik Atıf' ve 1 adet 'Metinde Kullanılmayan Kaynak' ekler ve sistemin bunları yakaladığını gösterir."
            )
            if st.button("🧨 Geçici Test Hatası Ekle ve Yakalandığını Gör", width="stretch"):
                is_numbered_doc = any(r.number is not None for r in parsed_doc.references)
                max_num = max((r.number for r in parsed_doc.references if r.number is not None), default=100)
                fake_ghost_num = max_num + 1
                fake_orphan_num = max_num + 2

                if is_numbered_doc:
                    sabotaged_body = (
                        parsed_doc.body_text
                        + f"\n\nBu kasıtlı test cümlesinde kaynakçada hiç var olmayan {fake_ghost_num} numaralı makaleye atıf yapılmıştır ({fake_ghost_num})."
                    )
                    sabotaged_refs = (
                        parsed_doc.references_text
                        + f"\n{fake_orphan_num}. Oz B, Yilmaz A. Metinde hic kullanilmayan kasitli kaynak tuzagi testi. Tubitak Test Journal. 2025;12(1):10-20."
                    )
                else:
                    sabotaged_body = (
                        parsed_doc.body_text
                        + "\n\nBu kasıtlı test cümlesinde kaynakçada hiç var olmayan bir makaleye atıf yapılmıştır (Öztürk ve Yılmaz, 2029)."
                    )
                    sabotaged_refs = (
                        parsed_doc.references_text
                        + "\nSabotajcı, T., and Deneme, K. (2024). Metinde hiç atıf yapılmayan kasıtlı kaynak tuzağı testi. J. Test Sci. 15 (2), 100-110."
                    )

                sabotaged_full = sabotaged_body + "\n\nReferences\n" + sabotaged_refs
                trap_result = parser.parse_text(sabotaged_full, filename=f"TEST_{parsed_doc.filename}")

                st.success("✅ Test tamamlandı! Sistem eklenen her iki hatayı da anında yakaladı:")
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.error(f"🚨 Yakalanan Kaynakçada Eksik Atıf: {len(trap_result.ghost_citations)} adet")
                    for gc in trap_result.ghost_citations:
                        st.markdown(f"- `{gc.marker}` — *\"{gc.claim_sentence}\"*")
                with tc2:
                    st.warning(f"⚠️ Yakalanan Metinde Kullanılmayan Kaynak: {len(trap_result.orphan_references)} adet")
                    for orp in trap_result.orphan_references:
                        st.markdown(f"- `[{orp.ref_id}]` {orp.raw_text}")


# ===========================================================================
# SEKME 2: HIZLI TEK CÜMLE TESTİ (DOSYA YÜKLEMEDEN)
# ===========================================================================

with tab2:
    st.subheader("✍️ Dosya Yüklemeden Hızlı Cümle ve Kaynak Uyumu Testi")
    st.caption(
        "Makale dosyası yüklemeden, kendi yazdığınız tek bir cümle ile kaynak makalenin özetini/paragrafını "
        "yan yana yapıştırarak birbirini destekleyip desteklemediğini hemen test edebilirsiniz."
    )

    col1, col2 = st.columns(2)
    with col1:
        single_claim = st.text_area(
            "📌 1. Sizin Makalenizde Yazdığınız Cümle:",
            value="Dünyanın yüzde 90'nı sudan oluşur [1].",
            height=140,
        )
    with col2:
        single_ref_text = st.text_area(
            "📖 2. Kaynak Makaledeki Paragraf veya Özet (Türkçe veya İngilizce):",
            value=(
                "Bu çalışmada gezegenlerin atmosferik ve yüzey bileşenleri incelenmiştir. "
                "Yüzde doksanı su olan dünyamız, sıvı hidrosfer tabakası sayesinde iklim dengesini korur. "
                "Diğer karasal gezegenlerde bu oran oldukça düşüktür."
            ),
            height=140,
        )

    if st.button("🔍 Bu Cümle Kaynak Paragrafla Uyumlu mu Kontrol Et", type="primary", width="stretch"):
        verifier = get_verifier(selected_model)
        dummy_paper = FetchedPaper(
            ref_id="TEKİL",
            query_title="Manuel Girilen Kaynak Metni",
            query_doi=None,
            exists_online=True,
            title_match_score=1.0,
            verified_title="Manuel Girilen Kaynak Metni",
            abstract=single_ref_text,
        )
        res = verifier.verify_claim_against_paper(single_claim, dummy_paper)
        v_info = get_verdict_info(res.verdict)

        if v_info["group"] == "ok":
            st.success(
                f"**Neden:** {v_info['why']}\n\n**Detay:** {res.explanation}",
                title=f"{v_info['badge']} (Konu Benzerliği: %{res.semantic_similarity * 100:.0f})",
            )
        elif v_info["group"] == "error":
            st.error(
                f"**Neden:** {v_info['why']}\n\n**Detay:** {res.explanation}\n\n**Ne Yapmalısınız:** {v_info['action']}",
                title=f"{v_info['badge']} (Konu Benzerliği: %{res.semantic_similarity * 100:.0f})",
            )
        else:
            st.warning(
                f"**Neden:** {v_info['why']}\n\n**Detay:** {res.explanation}\n\n**Ne Yapmalısınız:** {v_info['action']}",
                title=f"{v_info['badge']} (Konu Benzerliği: %{res.semantic_similarity * 100:.0f})",
            )

        if res.best_evidence:
            st.info(res.best_evidence, title="📖 Kaynak Metinde Bulunan En İlgili Cümle")
        if res.suggested_revision:
            st.success(res.suggested_revision, title="✏️ Önerilen Düzeltilmiş Cümle")


# ===========================================================================
# SEKME 3: TOPLU TABLO VE RAPOR İNDİR (EXCEL / WORD)
# ===========================================================================

with tab3:
    st.subheader("📊 Denetim Raporu Tablosu ve İndirme Ekranı")
    st.caption(
        "1. Sekmede yaptığınız denetimin tüm sonuçlarını burada toplu tablo halinde görebilir, "
        "Excel (`.csv`) veya Word/Metin (`.md`) dosyası olarak bilgisayarınıza indirebilirsiniz."
    )

    if "verification_results" not in st.session_state or not st.session_state["verification_results"]:
        st.info(
            "Henüz bir makale denetimi çalıştırılmadı. Lütfen **1. Sekmeden** makalenizi yükleyip denetimi başlatın.",
            title="ℹ️ Rapor Hazır Değil",
        )
    else:
        results_list_t3: List[VerificationResult] = st.session_state["verification_results"]
        f_map_t3: Dict[str, FetchedPaper] = st.session_state.get("fetched_map", {})

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
                "Kaynak Makaledeki Kanıt Paragrafı": r.best_evidence[:400] if r.best_evidence else "-",
            })

            md_lines.extend([
                f"### Atıf: `{r.marker}` — **{v_info['badge']}**",
                f"- **Makalenizdeki Cümle:** {r.claim_sentence}",
                f"- **Neden Bu Sonuç Çıktı?:** {v_info['why']}",
                f"- **Detaylı Açıklama:** {r.explanation}",
                f"- **Ne Yapmalısınız?:** {v_info['action']}",
                f"- **Önerilen Düzeltilmiş Cümle:** {r.suggested_revision or '-'}",
                f"- **Eşleşen Kaynak:** {fp.verified_title if fp else 'Bulunamadı'} ({fp.content_source if fp else '-'})",
                f"- **Kanıt Paragrafı:** {r.best_evidence[:500] if r.best_evidence else '-'}",
                "",
            ])

        st.caption(f"📌 Toplam **{len(results_list_t3)}** atıf içinden filtreye uyan **{len(rows)}** kayıt listeleniyor.")
        df_report = pd.DataFrame(rows)
        st.dataframe(df_report, hide_index=True)

        csv_bytes = df_report.to_csv(index=False).encode("utf-8-sig")
        md_bytes = "\n".join(md_lines).encode("utf-8")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "📥 Excel Uyumlu Tablo Olarak İndir (.csv)",
                data=csv_bytes,
                file_name="makale_atif_denetim_raporu.csv",
                mime="text/csv",
                width="stretch",
            )
        with dl_col2:
            st.download_button(
                "📥 Detaylı Rapor Olarak İndir (.md / Word Uyumlu)",
                data=md_bytes,
                file_name="makale_atif_denetim_raporu.md",
                mime="text/markdown",
                width="stretch",
            )
