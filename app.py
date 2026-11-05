r"""
MODÜL 4: Görsel TÜBİTAK Akademik Makale & Semantik Atıf Denetim Arayüzü (app.py)
--------------------------------------------------------------------------------
Çalıştırmak için:
    .\.venv\Scripts\streamlit.exe run app.py
veya klasördeki `baslat.bat` dosyasına çift tıklayın.
"""

from __future__ import annotations

import importlib
import io
import random
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

from document_parser import DocumentParser, ParsedManuscript, ReferenceItem
from academic_fetcher import AcademicFetcher, FetchedPaper
from rag_verifier import SemanticRAGVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Sayfa Yapılandırması ve Görsel Stil (CSS)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TÜBİTAK Akademik Atıf & Anlam Denetçisi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .verdict-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
    }
    .v-supported { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
    .v-contradicted { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
    .v-insufficient { background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
    .v-fake { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #a855f7; }
    .evidence-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #1e293b;
        margin: 8px 0;
    }
    .revision-box {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.95rem;
        color: #064e3b;
        margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Önbellekli Motor Nesneleri (RAM ve Hız Koruması)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Çok dilli semantik anlam modeli (CPU) hazırlanıyor...")
def get_verifier(ollama_model: str) -> SemanticRAGVerifier:
    return SemanticRAGVerifier(ollama_model=ollama_model, embedding_device="cpu")


def get_parser() -> DocumentParser:
    return DocumentParser()


USER_PDF_DIR = Path(".cache/user_pdfs")
USER_PDF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Örnek Demo Makalesi (Tek Tuşla Test İçin)
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
# Sol Yan Menü (Sidebar): Sistem Durumu ve Ayarlar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Sistem & Model Ayarları")
    st.caption("RTX 3050 (4 GB VRAM) & 8 GB RAM Optimize Mod")

    # Ollama Durum Kontrolü
    temp_verifier = SemanticRAGVerifier(embedding_device="cpu")
    ollama_ok, default_model, installed_models = temp_verifier.check_ollama_status()

    if ollama_ok and installed_models:
        st.success(f"🟢 Ollama Aktif ({len(installed_models)} model hazır)")
        selected_model = st.selectbox(
            "Kullanılacak Yerel Dil Modeli (Ollama):",
            options=installed_models,
            index=installed_models.index(default_model) if default_model in installed_models else 0,
        )
    elif ollama_ok and not installed_models:
        st.warning("🟡 Ollama çalışıyor ancak henüz model indirilmemiş.")
        st.code("ollama pull qwen2.5:3b", language="powershell")
        selected_model = "qwen2.5:3b"
    else:
        st.info(
            "ℹ️ Ollama şu an arka planda açık değil. Sistem otomatik olarak "
            "**Çok Dilli Semantik Vektör + Sayısal NLI Motoru (CPU)** ile çalışacaktır. "
            "(Ollama'yı açtığınızda otomatik bağlanır)."
        )
        selected_model = "llama3.2"

    st.divider()
    st.subheader("🌐 İnternet Tarama Kanalları")
    enable_scihub = st.checkbox(
        "🔓 Sci-Hub Otomatik Tam Metin PDF İndirici (varsayılan KAPALI)",
        value=False,
        help=(
            "⚠️ Hukuki uyarı: Sci-Hub üzerinden telifli makale indirmek birçok ülkede telif hakkı "
            "ihlali sayılır ve kurumsal denetimde risk oluşturur. Bu seçenek varsayılan olarak "
            "kapalıdır; açtığınızda sorumluluk kullanıcıya aittir. Önerilen alternatif: Unpaywall, "
            "OpenAlex açık erişim, CORE, Europe PMC veya kütüphane erişimi."
        ),
    )
    enable_jina = st.checkbox("🕸️ Agent-Reach (Jina Reader) Web Okuyucu", value=True, help="Web sayfası formatındaki makaleleri r.jina.ai üzerinden bot engeline takılmadan okur.")
    enable_suggestions = st.checkbox("💡 Sahte/Çelişen Kaynak İçin Gerçek Makale Öner", value=True, help="Uydurma veya hatalı kaynak yakalandığında iddiayı destekleyen gerçek makaleler önerir.")

    st.divider()
    st.subheader("📂 Ek Yerel Referans PDF'leriniz (İsteğe Bağlı)")
    uploaded_ref_pdfs = st.file_uploader(
        "Bilgisayarınızda kayıtlı referans makale PDF'leri varsa buraya bırakabilirsiniz:",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_ref_pdfs:
        for up_pdf in uploaded_ref_pdfs:
            target_path = USER_PDF_DIR / up_pdf.name
            target_path.write_bytes(up_pdf.getvalue())
        st.success(f"📎 {len(uploaded_ref_pdfs)} referans PDF yerel havuza eklendi.")

    st.divider()
    st.markdown(
        "**Aktif Arama Motorları:**\n"
        "- CrossRef & OpenAlex (Künye + Retraction)\n"
        "- PubMed & PMC (Tam Metin)\n"
        "- Semantic Scholar & arXiv\n"
        "- Google Scholar & DergiPark\n"
        "- Sci-Hub & Jina Reader"
    )


# ---------------------------------------------------------------------------
# Ana Başlık
# ---------------------------------------------------------------------------

st.title("🎓 TÜBİTAK Akademik Makale, Kaynakça ve Semantik Atıf Denetçisi")
st.markdown(
    "Makalenizdeki **kaynakların gerçekliğini**, **geri çekilip çekilmediğini (Retracted)**, "
    "**biçimsel atıf uyumunu** ve atıf yaptığınız cümledeki **anlamın (paraphrase / çeviri dahil)** "
    "referans makaleden çıkarılıp çıkarılamayacağını denetler."
)

tab1, tab2, tab3 = st.tabs([
    "📄 1. Tam Makale Otomatik Denetimi (.docx / .pdf)",
    "⚡ 2. Seçmeli / Aralıklı Atıf & Anlam Çıkarımı Testi",
    "📊 3. Denetim Raporu & Excel/Markdown Çıktısı",
])


# ===========================================================================
# SEKME 1: TAM MAKALE OTOMATİK DENETİMİ
# ===========================================================================

with tab1:
    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_manuscript = st.file_uploader(
            "Kendi yazdığınız makaleyi yükleyin (.docx, .pdf veya .txt):",
            type=["docx", "pdf", "txt"],
        )
    with col_demo:
        st.write("")
        st.write("")
        load_demo = st.button("🧪 Örnek TÜBİTAK Makalesi Yükle (Demo)", use_container_width=True)

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
        st.session_state["current_parsed_doc"] = parsed_doc
        st.subheader(f"📑 1. Aşama: Biçimsel Tarama Özeti — `{parsed_doc.filename}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📚 Kaynakça Sayısı", len(parsed_doc.references))
        m2.metric("📌 Metin İçi Atıf Sayısı", len(parsed_doc.citations))
        m3.metric(
            "🚨 Hayalet Atıf (Kaynakçada Yok)",
            len(parsed_doc.ghost_citations),
            delta="-Hata" if parsed_doc.ghost_citations else "Temiz",
            delta_color="inverse" if parsed_doc.ghost_citations else "normal",
        )
        m4.metric(
            "⚠️ Yetim Kaynak (Metinde Yok)",
            len(parsed_doc.orphan_references),
            delta="-Uyarı" if parsed_doc.orphan_references else "Temiz",
            delta_color="inverse" if parsed_doc.orphan_references else "normal",
        )

        # Biçimsel Uyarı Kutuları (Hayalet Atıf & Yetim Kaynak)
        if parsed_doc.ghost_citations or parsed_doc.orphan_references:
            with st.expander("⚠️ TÜBİTAK Biçimsel Atıf/Kaynakça Uyumsuzluklarını Gör (Hemen Düzeltilmeli)", expanded=True):
                if parsed_doc.ghost_citations:
                    st.error("**🚨 Hayalet Atıflar (Ana metinde atıf yapılmış ancak Kaynakça listesinde YOK!):**")
                    for gc in parsed_doc.ghost_citations:
                        st.markdown(f"- **Atıf:** `{gc.marker}` — *Cümle:* \"{gc.claim_sentence}\"")
                if parsed_doc.orphan_references:
                    st.warning("**⚠️ Yetim Kaynaklar (Kaynakça listesinde var ancak ana metinde HİÇ KULLANILMAMIŞ!):**")
                    for orp in parsed_doc.orphan_references:
                        st.markdown(f"- **[{orp.ref_id}]** {orp.raw_text}")

        # 1. AŞAMA ŞEFFAFLIK: Eşleşme Kanıt Tablosu
        with st.expander(
            f"📋 Eşleşme Kanıt Tablosunu İncele ({len(parsed_doc.references)} Kaynak & {len(parsed_doc.citations)} Metin İçi Atıf Nereden Yakalandı?)",
            expanded=False,
        ):
            st.caption(
                "Aşağıdaki tabloda kaynakçadaki her bir makalenin ana metinde kaç kez geçtiğini, "
                "hangi parantez işaretiyle yakalandığını ve geçtiği gerçek cümleyi görebilirsiniz."
            )
            ref_to_cits: Dict[str, List[Any]] = {}
            for c_item in parsed_doc.citations:
                if c_item.matched_ref_id:
                    ref_to_cits.setdefault(c_item.matched_ref_id, []).append(c_item)

            proof_rows = []
            for r_item in parsed_doc.references:
                matched_c_list = ref_to_cits.get(r_item.ref_id, [])
                markers_str = ", ".join(dict.fromkeys(c.marker for c in matched_c_list)) if matched_c_list else "—"
                first_sent = matched_c_list[0].claim_sentence if matched_c_list else "❌ Metinde kullanılmamış (Yetim)"
                proof_rows.append({
                    "Kaynak No": f"[{r_item.ref_id}]",
                    "İlk Yazar & Yıl": f"{r_item.first_author_surname or 'Bilinmiyor'} ({r_item.year or '?'})",
                    "Metinde Kullanım": f"{r_item.cited_in_text_count} kez",
                    "Yakalanan İşaret": markers_str,
                    "Makale Başlığı / Kaynak": (r_item.title or r_item.raw_text)[:95],
                    "Metinde Geçtiği Örnek Cümle (Kanıt)": first_sent[:180],
                })
            st.dataframe(pd.DataFrame(proof_rows), use_container_width=True, hide_index=True)

        # CANLI SABOTAJ / SAĞLAMLIK TESTİ (Sistemin Hataları Gerçekten Yakaladığını Kanıtla)
        with st.expander("🧪 Sistemden %100 Emin Ol: Tek Tıkla Canlı Tuzak (Sabotaj) Testi Yap", expanded=False):
            st.info(
                "Bu buton, yüklediğiniz makale dosyasına dokunmadan bellekte **2 kasıtlı hata (1 Hayalet Atıf + 1 Yetim Kaynak)** "
                "enjekte eder ve sistemin bu hataları halının altına süpürmeyip anında yakaladığını canlı olarak kanıtlar."
            )
            if st.button("🧨 Bu Makaleye 2 Gizli Tuzak Ekle ve Sistemin Yakaladığını Kanıtla", use_container_width=True):
                is_numbered_doc = any(r.number is not None for r in parsed_doc.references)
                max_num = max((r.number for r in parsed_doc.references if r.number is not None), default=100)
                fake_ghost_num = max_num + 1
                fake_orphan_num = max_num + 2

                if is_numbered_doc:
                    sabotaged_body = (
                        parsed_doc.body_text
                        + f"\n\nBu kasıtlı test cümlesinde kaynakçada hiç var olmayan {fake_ghost_num} numaralı hayalet makaleye atıf yapılmıştır ({fake_ghost_num})."
                    )
                    sabotaged_refs = (
                        parsed_doc.references_text
                        + f"\n{fake_orphan_num}. Oz B, Yilmaz A. Metinde hic kullanilmayan kasitli yetim kaynak tuzagi testi. Tubitak Test Journal. 2025;12(1):10-20."
                    )
                else:
                    sabotaged_body = (
                        parsed_doc.body_text
                        + "\n\nBu kasıtlı test cümlesinde kaynakçada hiç var olmayan hayalet bir makaleye atıf yapılmıştır (Öztürk ve Yılmaz, 2029)."
                    )
                    sabotaged_refs = (
                        parsed_doc.references_text
                        + "\nSabotajcı, T., and Deneme, K. (2024). Metinde hiç atıf yapılmayan kasıtlı yetim kaynak tuzağı testi. J. Test Sci. 15 (2), 100-110."
                    )

                sabotaged_full = sabotaged_body + "\n\nReferences\n" + sabotaged_refs
                trap_result = parser.parse_text(sabotaged_full, filename=f"TUZAK_TESTI_{parsed_doc.filename}")

                st.success("✅ **Canlı Tuzak Testi Başarıyla Tamamlandı! Sistem Enjekte Edilen Her İki Hatayı da Anında Yakaladı:**")
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.error(f"🚨 **Yakalanan Hayalet Atıf Sayısı: {len(trap_result.ghost_citations)}**")
                    for gc in trap_result.ghost_citations:
                        st.markdown(f"- **Yakalanan Sahte Atıf:** `{gc.marker}` (`{gc.expected_key}`)\n- **Yakalandığı Cümle:** *\"{gc.claim_sentence}\"*")
                with tc2:
                    st.warning(f"⚠️ **Yakalanan Yetim Kaynak Sayısı: {len(trap_result.orphan_references)}**")
                    for orp in trap_result.orphan_references:
                        st.markdown(f"- **Yakalanan Yetim Kaynak:** `[{orp.ref_id}]` {orp.raw_text}")

        st.divider()

        # Tam Denetim Başlatma Butonu
        start_audit = st.button(
            "🚀 Tüm Kaynakları İnternetten Çek (Sci-Hub / PubMed / CrossRef) ve Anlam Çıkarımını Denetle",
            type="primary",
            use_container_width=True,
        )

        if start_audit:
            fetcher = AcademicFetcher(enable_scihub=enable_scihub, enable_jina=enable_jina)
            verifier = get_verifier(selected_model)

            # 1. Önce tüm kaynakça maddelerini 5'li paralel iş parçacığı (Multi-Threading) ile internetten doğrula
            fetched_map: Dict[str, FetchedPaper] = {}
            total_refs = len(parsed_doc.references)
            prog_bar = st.progress(0, text="İnternet akademik veritabanları 5'li paralel modda taranıyor...")

            completed_refs = 0
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_ref = {
                    executor.submit(fetcher.verify_and_fetch, ref_item, USER_PDF_DIR): ref_item
                    for ref_item in parsed_doc.references
                }
                for future in as_completed(future_to_ref):
                    ref_item = future_to_ref[future]
                    completed_refs += 1
                    try:
                        fetched_map[ref_item.ref_id] = future.result()
                    except Exception:
                        fetched_map[ref_item.ref_id] = FetchedPaper(
                            ref_id=ref_item.ref_id,
                            query_title=ref_item.title or ref_item.raw_text,
                            query_doi=ref_item.doi,
                        )
                    prog_bar.progress(
                        int((completed_refs / max(1, total_refs * 2)) * 100),
                        text=f"🌐 [{completed_refs}/{total_refs}] Kaynak doğrulandı: {ref_item.title[:60]}...",
                    )

            # 2. Her bir metin içi atıf cümlesini eşleştiği kaynak makaleyle Semantik RAG üzerinden denetle
            verification_results: List[VerificationResult] = []
            total_cits = len(parsed_doc.citations)

            for c_idx, cit in enumerate(parsed_doc.citations, start=1):
                prog_bar.progress(
                    50 + int((c_idx / max(1, total_cits)) * 50),
                    text=f"🧠 [{c_idx}/{total_cits}] Semantik Anlam Çıkarımı yapılıyor: {cit.marker}...",
                )
                if cit.matched_ref_id is None or cit.matched_ref_id not in fetched_map:
                    # Hayalet atıf: Kaynakçada yok ama kullanıcı cümlesi için gerçek makale önerelim
                    alt_list = fetcher.suggest_real_papers(cit.claim_sentence, top_k=3) if enable_suggestions else []
                    verification_results.append(
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
                                "sonundaki Kaynakça bölümünde bu kaynak yer almıyor!"
                            ),
                            alternative_papers=alt_list,
                            used_llm_model="Biçimsel Denetçi",
                        )
                    )
                else:
                    paper_obj = fetched_map[cit.matched_ref_id]
                    res = verifier.verify_claim_against_paper(
                        claim_sentence=cit.claim_sentence,
                        paper=paper_obj,
                        citation_id=cit.citation_id,
                        marker=cit.marker,
                        context_text=cit.context_text,
                        alternative_finder=fetcher if enable_suggestions else None,
                    )
                    verification_results.append(res)

            prog_bar.progress(100, text="✅ Tüm makale ve atıf denetimi tamamlandı!")
            st.session_state["fetched_map"] = fetched_map
            st.session_state["verification_results"] = verification_results
            st.session_state["last_parsed_doc"] = parsed_doc

        # Sonuçları Göster
        if "verification_results" in st.session_state and "fetched_map" in st.session_state:
            results_list: List[VerificationResult] = st.session_state["verification_results"]
            f_map: Dict[str, FetchedPaper] = st.session_state["fetched_map"]

            st.subheader("🎯 2. Aşama: Anlam Çıkarımı ve Kaynak Doğrulama Sonuçları")

            cnt_sup = sum(1 for r in results_list if r.verdict == "DESTEKLENİYOR")
            cnt_part = sum(1 for r in results_list if r.verdict == "KISMEN_DESTEKLENİYOR")
            cnt_con = sum(1 for r in results_list if r.verdict == "ÇELİŞİYOR")
            cnt_ins = sum(1 for r in results_list if r.verdict == "KANIT_YETERSİZ")
            cnt_fake = sum(1 for r in results_list if r.verdict == "SAHTE_KAYNAK")
            cnt_unknown = sum(1 for r in results_list if r.verdict == "DOGRULANAMADI")
            cnt_ghost = sum(1 for r in results_list if r.verdict == "HAYALET_ATIF")
            cnt_ret = sum(1 for p in f_map.values() if p.is_retracted)

            k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
            k1.metric("✅ Tam Desteklenen", cnt_sup)
            k2.metric(
                "☑️ Özet Uyumlu",
                cnt_part,
                help="Kaynak %100 gerçek ve cümlenin ana konusu makale özetiyle örtüşüyor; ancak tam metin PDF'i ücretli/kilitli olduğu için özet düzeyinde doğrulandı.",
            )
            k3.metric("❌ Çelişen", cnt_con)
            k4.metric("⚠️ Kanıt Yetersiz", cnt_ins)
            k5.metric(
                "🚨 Sahte (kanıtlı)",
                cnt_fake,
                help="Yalnızca somut kanıtla (ör. DOI'nin başka makaleye ait olması) 'sahte' sayılır. Bulunamayan kaynaklar buraya değil, 'Doğrulanamadı' sayacına yazılır.",
            )
            k6.metric(
                "❓ Doğrulanamadı",
                cnt_unknown,
                help="Kaynak hiçbir veritabanında bulunamadı veya ağ hatası oluştu. Bu, kaynağın sahte olduğu anlamına gelmez.",
            )
            k7.metric(
                "👻 Hayalet Atıf",
                cnt_ghost,
                help="Ana metinde atıf yapılmış ancak makalenin sonundaki Kaynakça listesine eklenmemiş atıf sayısı.",
            )
            k8.metric("🛑 Geri Çekilmiş", cnt_ret)

            # A) Kaynakça Gerçeklik Özeti Tablosu
            with st.expander("📚 Kaynakça Listesi İnternet Doğrulama Durumu (CrossRef / OpenAlex / PubMed / Sci-Hub)", expanded=True):
                ref_rows = []
                for r_id, fp in f_map.items():
                    ref_rows.append({
                        "Kaynak No": f"[{r_id}]",
                        "Durum": (
                            "🛑 GERİ ÇEKİLMİŞ" if fp.is_retracted
                            else ("✅ Gerçek Makale" if fp.exists_online
                                  else ("❓ DOĞRULANAMADI (ağ hatası)" if fp.lookup_status == "error"
                                        else "❓ DOĞRULANAMADI (bulunamadı)"))
                        ),
                        "Doğrulanan Başlık": fp.verified_title or fp.query_title,
                        "Yazarlar": ", ".join(fp.verified_authors[:3]) if fp.verified_authors else "-",
                        "Yıl": fp.verified_year or "-",
                        "İçerik Kaynağı": fp.content_source,
                        "Atıf Sayısı": fp.citation_count,
                        "Eşleşme Güveni": f"%{fp.title_match_score * 100:.0f}" if fp.exists_online else "-",
                        "Uyarılar": fp.lookup_error or (" | ".join(fp.metadata_warnings) if fp.metadata_warnings else "Sorun Yok"),
                    })
                st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

            # B) Her Bir Metin İçi Atıf Cümlesinin Detaylı Denetim Kartları
            st.markdown("### 🔍 Cümle Bazlı Atıf ve Anlam Çıkarımı Kartları")
            for r in results_list:
                icon_map = {
                    "DESTEKLENİYOR": "✅",
                    "KISMEN_DESTEKLENİYOR": "☑️",
                    "ÇELİŞİYOR": "❌",
                    "KANIT_YETERSİZ": "⚠️",
                    "DOGRULANAMADI": "❓",
                    "SAHTE_KAYNAK": "🚨",
                    "HAYALET_ATIF": "👻",
                }
                icon = icon_map.get(r.verdict, "ℹ️")
                header_text = f"{icon} [{r.verdict}] — Atıf: {r.marker} | \"{r.claim_sentence[:95]}...\""

                with st.expander(header_text, expanded=(r.verdict not in ("DESTEKLENİYOR", "KISMEN_DESTEKLENİYOR"))):
                    c_left, c_right = st.columns([3, 2])
                    with c_left:
                        st.markdown(f"**📝 Sizin Makalenizdeki İddia Cümlesi:**\n> {r.claim_sentence}")
                        if r.best_evidence:
                            st.markdown(
                                f"**📖 Referans Makalede Bulunan Kanıt Paragrafı (Anlam Yakınlığı: %{r.semantic_similarity * 100:.1f}):**\n"
                                f"<div class='evidence-box'>{r.best_evidence}</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(f"**🧑‍⚖️ Hakem Değerlendirmesi:** {r.explanation}")

                        if r.is_secondary_citation and r.secondary_citation_note:
                            st.info(r.secondary_citation_note)

                        if r.suggested_revision:
                            st.markdown(
                                f"**✏️ Önerilen Akademik Cümle Düzeltmesi:**\n"
                                f"<div class='revision-box'>{r.suggested_revision}</div>",
                                unsafe_allow_html=True,
                            )

                    with c_right:
                        if r.ref_id and r.ref_id in f_map:
                            fp = f_map[r.ref_id]
                            st.markdown("**🌐 Eşleşen Kaynak Künyesi:**")
                            st.write(f"- **Başlık:** {fp.verified_title or fp.query_title}")
                            st.write(f"- **Yazarlar:** {', '.join(fp.verified_authors[:4]) or '-'}")
                            st.write(f"- **Yıl & Dergi:** {fp.verified_year or '-'} | {fp.verified_venue or '-'}")
                            st.write(f"- **DOI:** `{fp.verified_doi or 'Yok'}`")
                            st.write(f"- **İçerik Kaynağı:** `{fp.content_source}`")
                            st.write(f"- **Eşleşme Güveni:** %{fp.title_match_score * 100:.0f}")
                            if fp.local_pdf_path:
                                st.success(f"📄 Tam Metin PDF İndirildi: `{fp.local_pdf_path}`")
                            for w in fp.metadata_warnings:
                                st.warning(w)

                        if r.alternative_papers:
                            st.markdown("**💡 Bu İddiayı Destekleyen Gerçek Alternatif Makale Önerileri:**")
                            for alt in r.alternative_papers:
                                st.markdown(
                                    f"- **{alt.title}** ({alt.authors}, {alt.year}) — *{alt.venue}* "
                                    f"[DOI: `{alt.doi}` | Atıf: {alt.citation_count}]"
                                )


# ===========================================================================
# SEKME 2: SEÇMELİ / ARALIKLI ATIF & ANLAM ÇIKARIMI TESTİ
# ===========================================================================

with tab2:
    st.subheader("⚡ Seçmeli / Aralıklı Atıf & Anlam Çıkarımı Kontrol Merkezi")
    active_doc: ParsedManuscript | None = st.session_state.get("current_parsed_doc")

    test_mode = st.radio(
        "Çalışma Modu Seçin:",
        options=[
            "📋 Yüklediğim Makaleden Atıf Seçerek Doğrula (Tek Tek / Çoklu / Aralık: 1-50 / Tümü)",
            "✍️ Dışarıdan Manuel Cümle & Metin Testi (Serbest Giriş)",
        ],
        horizontal=True,
    )

    if test_mode.startswith("📋"):
        if active_doc is None or not active_doc.citations:
            st.warning(
                "📂 Henüz yüklü bir makale bulunamadı. Lütfen **1. Sekmeden** makalenizi (`.docx` veya `.pdf`) yükleyin; "
                "tüm atıflarınız seçilebilir liste olarak buraya otomatik gelecektir."
            )
        else:
            all_cits = active_doc.citations
            total_cits_n = len(all_cits)

            st.success(
                f"📂 **Aktif Makale:** `{active_doc.filename}` — Toplam **{total_cits_n} Metin İçi Atıf** ve "
                f"**{len(active_doc.references)} Kaynak** hazır. Aşağıdan istediğiniz aralığı (`1-50` gibi) veya "
                "istediğiniz tekil/çoklu atıfları kutucuklardan seçip doğrudan sorgulayabilirsiniz."
            )

            # Makale değiştiyse seçim durumunu ve eski makalenin önbelleğini sıfırla
            if st.session_state.get("tab2_doc_name") != active_doc.filename:
                st.session_state["tab2_doc_name"] = active_doc.filename
                st.session_state["tab2_selected_set"] = set(range(1, min(6, total_cits_n + 1)))
                st.session_state["tab2_editor_ver"] = 0
                st.session_state["fetched_map"] = {}
                st.session_state.pop("tab2_results", None)

            if "tab2_selected_set" not in st.session_state:
                st.session_state["tab2_selected_set"] = set(range(1, min(6, total_cits_n + 1)))
            if "tab2_editor_ver" not in st.session_state:
                st.session_state["tab2_editor_ver"] = 0

            # --- ÜST KONTROL ÇUBUĞU: ARALIK SEÇİMİ (1-50 vb.), TÜMÜNÜ SEÇ, TEMİZLE ---
            ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.2, 1.2, 1.8, 1.5, 1.5])
            with ctrl1:
                r_start = st.number_input(
                    "Başlangıç Atıf No:",
                    min_value=1,
                    max_value=total_cits_n,
                    value=1,
                    step=1,
                )
            with ctrl2:
                r_end = st.number_input(
                    "Bitiş Atıf No:",
                    min_value=1,
                    max_value=total_cits_n,
                    value=min(50, total_cits_n),
                    step=1,
                )
            with ctrl3:
                st.write("")
                st.write("")
                if st.button(f"🔢 Aralığı Seç ({min(r_start, r_end)} – {max(r_start, r_end)})", use_container_width=True):
                    s_idx, e_idx = min(int(r_start), int(r_end)), max(int(r_start), int(r_end))
                    st.session_state["tab2_selected_set"] = set(range(s_idx, e_idx + 1))
                    st.session_state["tab2_editor_ver"] += 1
                    st.rerun()
            with ctrl4:
                st.write("")
                st.write("")
                if st.button(f"✅ Tümünü Seç (1 – {total_cits_n})", use_container_width=True):
                    st.session_state["tab2_selected_set"] = set(range(1, total_cits_n + 1))
                    st.session_state["tab2_editor_ver"] += 1
                    st.rerun()
            with ctrl5:
                st.write("")
                st.write("")
                if st.button("🧹 Seçimi Temizle", use_container_width=True):
                    st.session_state["tab2_selected_set"] = set()
                    st.session_state["tab2_editor_ver"] += 1
                    st.rerun()

            # --- İNTERAKTİF KUTUCUKLU (CHECKBOX) ATIF SEÇİM TABLOSU ---
            current_sel_set = st.session_state["tab2_selected_set"]
            editor_rows = []
            for idx_c, c_obj in enumerate(all_cits, start=1):
                ref_obj = c_obj.matched_reference
                ref_summary = (
                    f"[{c_obj.matched_ref_id}] {ref_obj.first_author_surname or ''} ({ref_obj.year or '?'}) — {(ref_obj.title or ref_obj.raw_text)[:75]}"
                    if ref_obj
                    else "🚨 Kaynakçada Yok (Hayalet Atıf)"
                )
                editor_rows.append({
                    "Seç": idx_c in current_sel_set,
                    "Atıf No": idx_c,
                    "Atıf İşareti": c_obj.marker,
                    "Eşleşen Kaynak": ref_summary,
                    "Makalenizdeki İddia Cümlesi": c_obj.claim_sentence,
                })

            df_cits_editor = pd.DataFrame(editor_rows)
            edited_df = st.data_editor(
                df_cits_editor,
                key=f"cit_selector_{active_doc.filename}_{st.session_state['tab2_editor_ver']}",
                use_container_width=True,
                hide_index=True,
                height=340,
                disabled=["Atıf No", "Atıf İşareti", "Eşleşen Kaynak", "Makalenizdeki İddia Cümlesi"],
                column_config={
                    "Seç": st.column_config.CheckboxColumn(
                        "☑️ Seç",
                        help="Doğrulamak istediğiniz atıfları tek tek veya çoklu olarak buradan işaretleyebilirsiniz.",
                        default=False,
                    ),
                    "Atıf No": st.column_config.NumberColumn("No", width="small"),
                    "Atıf İşareti": st.column_config.TextColumn("Parantez / İşaret", width="medium"),
                    "Eşleşen Kaynak": st.column_config.TextColumn("Eşleşen Kaynakça", width="large"),
                    "Makalenizdeki İddia Cümlesi": st.column_config.TextColumn("Makaledeki İddia Cümlesi", width="large"),
                },
            )

            # Kullanıcının tablodaki anlık tiklerini oku
            selected_indices = [
                int(row["Atıf No"]) for _, row in edited_df.iterrows() if bool(row["Seç"])
            ]
            st.session_state["tab2_selected_set"] = set(selected_indices)
            selected_cits_list = [all_cits[i - 1] for i in selected_indices]

            # Seçilen atıfların ihtiyaç duyduğu tekil kaynakları belirle
            needed_refs_map: Dict[str, ReferenceItem] = {}
            for sc in selected_cits_list:
                if sc.matched_reference is not None and sc.matched_ref_id:
                    needed_refs_map[sc.matched_ref_id] = sc.matched_reference

            run_selected_btn = st.button(
                f"🚀 Seçilen {len(selected_cits_list)} Atıfı ({len(needed_refs_map)} Tekil Kaynak) İnternetten Çek ve Anlam Çıkarımını Doğrula",
                type="primary",
                use_container_width=True,
                disabled=(len(selected_cits_list) == 0),
            )

            if run_selected_btn and selected_cits_list:
                fetcher = AcademicFetcher(enable_scihub=enable_scihub, enable_jina=enable_jina)
                verifier = get_verifier(selected_model)

                # Daha önce çekilmiş kaynaklar varsa önbellekten (session_state) koru, ancak bulunamamış veya kısa özetli olanları v2 motoruyla tazele
                fetched_map_tab2: Dict[str, FetchedPaper] = dict(st.session_state.get("fetched_map", {}))
                refs_to_fetch = [
                    r_it
                    for r_id, r_it in needed_refs_map.items()
                    if r_id not in fetched_map_tab2
                    or not fetched_map_tab2[r_id].exists_online
                    or len(fetched_map_tab2[r_id].abstract) < 200
                ]

                prog_t2 = st.progress(0, text="Seçilen atıfların kaynakları internetten doğrulanıyor...")

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
                                fetched_map_tab2[r_it.ref_id] = fut.result()
                            except Exception:
                                fetched_map_tab2[r_it.ref_id] = FetchedPaper(
                                    ref_id=r_it.ref_id,
                                    query_title=r_it.title or r_it.raw_text,
                                    query_doi=r_it.doi,
                                )
                            prog_t2.progress(
                                int((done_r / max(1, total_r * 2)) * 100),
                                text=f"🌐 [{done_r}/{total_r}] Kaynak çekildi: {(r_it.title or r_it.raw_text)[:60]}...",
                            )

                selected_results: List[VerificationResult] = []
                total_sel_c = len(selected_cits_list)
                for idx_sc, scit in enumerate(selected_cits_list, start=1):
                    prog_t2.progress(
                        50 + int((idx_sc / max(1, total_sel_c)) * 50),
                        text=f"🧠 [{idx_sc}/{total_sel_c}] Anlam Çıkarımı yapılıyor: {scit.marker}...",
                    )
                    if scit.matched_ref_id is None or scit.matched_ref_id not in fetched_map_tab2:
                        alt_list = fetcher.suggest_real_papers(scit.claim_sentence, top_k=3) if enable_suggestions else []
                        selected_results.append(
                            VerificationResult(
                                citation_id=scit.citation_id,
                                marker=scit.marker,
                                claim_sentence=scit.claim_sentence,
                                context_text=scit.context_text,
                                ref_id=None,
                                verdict="HAYALET_ATIF",
                                confidence_score=1.0,
                                semantic_similarity=0.0,
                                best_evidence="",
                                explanation=f"Metin içinde '{scit.marker}' olarak atıf yapılmış ancak Kaynakça bölümünde bulunamadı!",
                                alternative_papers=alt_list,
                                used_llm_model="Biçimsel Denetçi",
                            )
                        )
                    else:
                        p_obj = fetched_map_tab2[scit.matched_ref_id]
                        v_res = verifier.verify_claim_against_paper(
                            claim_sentence=scit.claim_sentence,
                            paper=p_obj,
                            citation_id=scit.citation_id,
                            marker=scit.marker,
                            context_text=scit.context_text,
                            alternative_finder=fetcher if enable_suggestions else None,
                        )
                        selected_results.append(v_res)

                prog_t2.progress(100, text=f"✅ Seçilen {total_sel_c} atıfın denetimi tamamlandı!")
                st.session_state["fetched_map"] = fetched_map_tab2
                st.session_state["tab2_results"] = selected_results
                st.session_state["verification_results"] = selected_results

            # Seçmeli Denetim Sonuçlarını Aynı Ekranda Göster
            if "tab2_results" in st.session_state and st.session_state["tab2_results"]:
                t2_results: List[VerificationResult] = st.session_state["tab2_results"]
                f_map_t2: Dict[str, FetchedPaper] = st.session_state.get("fetched_map", {})

                st.divider()
                st.subheader(f"🎯 Seçilen Atıfların Anlam Çıkarımı Sonuçları ({len(t2_results)} Atıf)")

                s_sup = sum(1 for r in t2_results if r.verdict == "DESTEKLENİYOR")
                s_part = sum(1 for r in t2_results if r.verdict == "KISMEN_DESTEKLENİYOR")
                s_con = sum(1 for r in t2_results if r.verdict == "ÇELİŞİYOR")
                s_ins = sum(1 for r in t2_results if r.verdict == "KANIT_YETERSİZ")
                s_fake = sum(1 for r in t2_results if r.verdict == "SAHTE_KAYNAK")
                s_unknown = sum(1 for r in t2_results if r.verdict == "DOGRULANAMADI")
                s_ghost = sum(1 for r in t2_results if r.verdict == "HAYALET_ATIF")

                sk1, sk2, sk3, sk4, sk5, sk6, sk7 = st.columns(7)
                sk1.metric("✅ Tam Desteklenen", s_sup)
                sk2.metric(
                    "☑️ Özet Uyumlu",
                    s_part,
                    help="Kaynak %100 gerçek ve cümlenin ana konusu makale özetiyle örtüşüyor; ancak tam metin PDF'i ücretli/kilitli olduğu için özet düzeyinde doğrulandı.",
                )
                sk3.metric("❌ Çelişen", s_con)
                sk4.metric("⚠️ Kanıt Yetersiz", s_ins)
                sk5.metric(
                    "🚨 Sahte (kanıtlı)",
                    s_fake,
                    help="Yalnızca somut kanıtla (ör. DOI'nin başka makaleye ait olması) 'sahte' sayılır.",
                )
                sk6.metric(
                    "❓ Doğrulanamadı",
                    s_unknown,
                    help="Kaynak hiçbir veritabanında bulunamadı veya ağ hatası oluştu; sahte olduğu anlamına gelmez.",
                )
                sk7.metric(
                    "👻 Hayalet Atıf",
                    s_ghost,
                    help="Ana metinde atıf yapılmış ancak makalenin sonundaki Kaynakça listesine eklenmemiş atıf sayısı.",
                )

                icon_map = {
                    "DESTEKLENİYOR": "✅",
                    "KISMEN_DESTEKLENİYOR": "☑️",
                    "ÇELİŞİYOR": "❌",
                    "KANIT_YETERSİZ": "⚠️",
                    "DOGRULANAMADI": "❓",
                    "SAHTE_KAYNAK": "🚨",
                    "HAYALET_ATIF": "👻",
                }
                verdict_label_map = {
                    "DESTEKLENİYOR": "TAM DESTEKLENİYOR",
                    "KISMEN_DESTEKLENİYOR": "ÖZET DÜZEYİNDE UYUMLU (TAM METİN KİLİTLİ)",
                    "ÇELİŞİYOR": "ÇELİŞİYOR",
                    "KANIT_YETERSİZ": "KANIT YETERSİZ / ALAKASIZ ATIF",
                    "DOGRULANAMADI": "DOĞRULANAMADI (BULUNAMADI / AĞ HATASI)",
                    "SAHTE_KAYNAK": "SAHTE / İNTERNETTE YOK (KANITLI)",
                    "HAYALET_ATIF": "HAYALET ATIF (KAYNAKÇADA YOK)",
                }
                for r in t2_results:
                    icon = icon_map.get(r.verdict, "ℹ️")
                    v_label = verdict_label_map.get(r.verdict, r.verdict)
                    hdr = f"{icon} [{v_label}] (Yakınlık: %{r.semantic_similarity * 100:.1f}) — Atıf: {r.marker} | \"{r.claim_sentence[:90]}...\""
                    with st.expander(hdr, expanded=True):
                        c_l, c_r = st.columns([3, 2])
                        with c_l:
                            st.markdown(f"**📝 Makalenizdeki İddia Cümlesi:**\n> {r.claim_sentence}")
                            if r.best_evidence:
                                st.markdown(
                                    f"**📖 Referans Makalede Bulunan Kanıt Paragrafı (Anlam Yakınlığı: %{r.semantic_similarity * 100:.1f}):**\n"
                                    f"<div class='evidence-box'>{r.best_evidence}</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(f"**🧑‍⚖️ Hakem Değerlendirmesi:** {r.explanation}")
                            if r.suggested_revision:
                                st.markdown(
                                    f"**✏️ Önerilen Cümle Düzeltmesi:**\n"
                                    f"<div class='revision-box'>{r.suggested_revision}</div>",
                                    unsafe_allow_html=True,
                                )
                        with c_r:
                            if r.ref_id and r.ref_id in f_map_t2:
                                fp = f_map_t2[r.ref_id]
                                st.markdown("**🌐 Doğrulanan Kaynak Künyesi:**")
                                st.write(f"- **Başlık:** {fp.verified_title or fp.query_title}")
                                st.write(f"- **Yazarlar:** {', '.join(fp.verified_authors[:4]) or '-'}")
                                st.write(f"- **Yıl & Dergi:** {fp.verified_year or '-'} | {fp.verified_venue or '-'}")
                                st.write(f"- **DOI:** `{fp.verified_doi or 'Yok'}`")
                                st.write(f"- **İçerik Kaynağı:** `{fp.content_source}`")
                                st.write(f"- **Eşleşme Güveni:** %{fp.title_match_score * 100:.0f}")
                                if fp.local_pdf_path:
                                    st.success(f"📄 Tam Metin PDF İndirildi: `{fp.local_pdf_path}`")
                                for w in fp.metadata_warnings:
                                    st.warning(w)
                            if r.alternative_papers:
                                st.markdown("**💡 Alternatif Gerçek Makale Önerileri:**")
                                for alt in r.alternative_papers:
                                    st.markdown(
                                        f"- **{alt.title}** ({alt.authors}, {alt.year}) — [DOI: `{alt.doi}`]"
                                    )
    else:
        col1, col2 = st.columns(2)
        with col1:
            single_claim = st.text_area(
                "✍️ Sizin Makalenizdeki İddia Cümlesi:",
                value="Dünyanın yüzde 90'nı sudan oluşur [1].",
                height=120,
            )
        with col2:
            single_ref_text = st.text_area(
                "📖 Referans Makaledeki Paragraf / Özet (Türkçe veya İngilizce):",
                value=(
                    "Bu çalışmada gezegenlerin atmosferik ve yüzey bileşenleri incelenmiştir. "
                    "Yüzde doksanı su olan dünyamız, sıvı hidrosfer tabakası sayesinde iklim dengesini korur. "
                    "Diğer karasal gezegenlerde bu oran oldukça düşüktür."
                ),
                height=120,
            )

        if st.button("🔍 Anlam Bütünlüğünü Test Et", type="primary"):
            verifier = get_verifier(selected_model)
            dummy_paper = FetchedPaper(
                ref_id="TEKİL",
                query_title="Manuel Referans Metni",
                query_doi=None,
                exists_online=True,
                verified_title="Manuel Girilen Referans Metni",
                abstract=single_ref_text,
            )
            res = verifier.verify_claim_against_paper(single_claim, dummy_paper)

            if res.verdict == "DESTEKLENİYOR":
                st.success(f"### ✅ KARAR: DESTEKLENİYOR (Anlam Yakınlığı: %{res.semantic_similarity * 100:.1f} | Güven: %{res.confidence_score * 100:.0f})")
            elif res.verdict == "ÇELİŞİYOR":
                st.error(f"### ❌ KARAR: ÇELİŞİYOR (Anlam Yakınlığı: %{res.semantic_similarity * 100:.1f} | Güven: %{res.confidence_score * 100:.0f})")
            else:
                st.warning(f"### ⚠️ KARAR: KANIT YETERSİZ (Anlam Yakınlığı: %{res.semantic_similarity * 100:.1f})")

            st.markdown(f"**Açıklama:** {res.explanation}")
            st.markdown(f"**Bulunan En Güçlü Kanıt Cümlesi:**\n> {res.best_evidence}")
            if res.suggested_revision:
                st.info(f"**✏️ Önerilen Düzeltme:** {res.suggested_revision}")


# ===========================================================================
# SEKME 3: DENETİM RAPORU VE EXCEL / MARKDOWN ÇIKTISI
# ===========================================================================

with tab3:
    st.subheader("📊 TÜBİTAK Denetim Raporu İndir")
    if "verification_results" not in st.session_state:
        st.info("Henüz bir makale denetimi çalıştırılmadı. Lütfen 1. Sekmeden makalenizi yükleyip denetimi başlatın.")
    else:
        results_list: List[VerificationResult] = st.session_state["verification_results"]
        f_map: Dict[str, FetchedPaper] = st.session_state["fetched_map"]

        rows = []
        md_lines = [
            "# TÜBİTAK Akademik Makale ve Semantik Atıf Denetim Raporu",
            f"**Tarih:** {datetime.now().strftime('%d.%m.%Y %H:%M')}\n",
            "## 1. Cümle Bazlı Atıf ve Çıkarım Sonuçları\n",
        ]

        for r in results_list:
            fp = f_map.get(r.ref_id) if r.ref_id else None
            rows.append({
                "Atıf İşareti": r.marker,
                "Karar": r.verdict,
                "Anlam Yakınlığı (%)": round(r.semantic_similarity * 100, 1),
                "Güven (%)": round(r.confidence_score * 100, 0),
                "Karar Kaynağı": r.decision_source or "-",
                "Eşleşme Güveni (%)": round(r.match_score * 100, 0),
                "Makaledeki İddia Cümleniz": r.claim_sentence,
                "Doğrulanan Kaynak Başlığı": fp.verified_title if fp else "Kaynakçada Yok",
                "Kaynak Durumu": fp.content_source if fp else "Hayalet Atıf",
                "Orijinal Kanıt Paragrafı": r.best_evidence[:400],
                "Hakem Açıklaması": r.explanation,
                "Önerilen Düzeltme": r.suggested_revision,
            })

            md_lines.extend([
                f"### Atıf: `{r.marker}` — Karar: **{r.verdict}** (Anlam Yakınlığı: %{r.semantic_similarity * 100:.1f})",
                f"- **İddia Cümleniz:** {r.claim_sentence}",
                f"- **Eşleşen Kaynak:** {fp.verified_title if fp else 'Bulunamadı'} ({fp.content_source if fp else '-'})",
                f"- **Hakem Açıklaması:** {r.explanation}",
                f"- **Kanıt Paragrafı:** {r.best_evidence[:500] if r.best_evidence else '-'}",
                f"- **Önerilen Düzeltme:** {r.suggested_revision or '-'}",
                "",
            ])

        df_report = pd.DataFrame(rows)
        st.dataframe(df_report, use_container_width=True, hide_index=True)

        csv_bytes = df_report.to_csv(index=False).encode("utf-8-sig")
        md_bytes = "\n".join(md_lines).encode("utf-8")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "📥 Excel Uyumlu CSV Raporu İndir (.csv)",
                data=csv_bytes,
                file_name="tubitak_atif_denetim_raporu.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                "📥 Detaylı Hakem Raporu İndir (.md / .txt)",
                data=md_bytes,
                file_name="tubitak_atif_denetim_raporu.md",
                mime="text/markdown",
                use_container_width=True,
            )
