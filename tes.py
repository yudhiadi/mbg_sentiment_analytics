import streamlit as st
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Sistem Analisis Sentimen NLP",
    page_icon="",
    layout="wide"
)

st.title("Sistem Analisis Sentimen Berbasis NLP")
st.markdown(
    """
    Aplikasi ini menggunakan Hugging Face Transformers (model pre-trained berbasis RoBERTa, bagian dari keluarga BERT) untuk menganalisis sentimen teks, misalnya opini masyarakat terhadap Program Makan Bergizi Gratis.

    Fitur utama:
    - Analisis **1 teks** (input manual)
    - Analisis **banyak teks via upload CSV**
    - Dashboard sederhana: distribusi sentimen & tabel hasil
    """
)

# ==========================
# Load spaCy & Model HF
# ==========================
@st.cache_resource
def load_nlp_and_model():
    # spaCy untuk preprocessing ringan
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # jika model belum di-download, pakai blank English
        nlp = spacy.blank("en")

    # Load model sentimen HuggingFace
    # Coba dulu IndoBERT Sentiment, jika gagal pakai multilingual BERT
    try:
        model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
        model_type = "indonesia"
    except Exception as e:
        st.warning(f"Gagal load model Indonesia, fallback ke multilingual BERT. Error: {e}")
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        model_type = "multilingual_stars"

    return nlp, clf, model_type, model_name

with st.spinner("Memuat model NLP (Hugging Face Transformers)..."):
    nlp, sentiment_pipe, MODEL_TYPE, MODEL_NAME = load_nlp_and_model()

# Sidebar info model
with st.sidebar:
    st.header("ℹ️ Info Model")
    st.write("**Library:** Hugging Face Transformers")
    st.write(f"**Model ID:** `{MODEL_NAME}`")
    st.write(f"**Tipe Model:** `{MODEL_TYPE}`")
    st.markdown(
        """
        - Backbone: BERT / IndoBERT (Transformer)
        - Status: **Pre-trained model** di HuggingFace
        - Digunakan hanya untuk **inference** (tanpa training ulang)
        """
    )

# ==========================
# Fungsi Bantu: Mapping Label
# ==========================
def map_label(label_raw: str) -> str:
    """
    Mapping label mentah model ke Positif / Netral / Negatif.
    - Model Indonesia: LABEL_0, LABEL_1, LABEL_2
    - Model multilingual: label '1 star' s.d. '5 stars'
    """
    l = label_raw.lower()

    if MODEL_TYPE == "indonesia":
        # mapping khusus untuk mdhugol/indonesia-bert-sentiment-classification
        # asumsi umum: LABEL_0 = positive, LABEL_1 = negative, LABEL_2 = neutral
        if l in ("label_0", "positive", "pos"):
            return "Positif"
        elif l in ("label_1", "negative", "neg"):
            return "Negatif"
        else:  # label_2 / neutral / lainnya
            return "Netral"
    else:
        # multilingual 1–5 stars
        try:
            num = int(l.split()[0])
        except Exception:
            return "Netral"
        if num <= 2:
            return "Negatif"
        elif num == 3:
            return "Netral"
        else:
            return "Positif"


def analyze_text(text: str):
    """
    Preprocess teks dengan spaCy (ringan),
    lalu analisis sentimen dengan model HuggingFace.
    """
    doc = nlp(text)
    cleaned = " ".join([t.text for t in doc])

    result = sentiment_pipe(cleaned)[0]
    raw_label = result["label"]
    score = float(result["score"])
    final_label = map_label(raw_label)

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "label_model": raw_label,
        "sentimen": final_label,
        "score": score
    }

# ==========================
# TAB: Single & CSV
# ==========================
tab1, tab2 = st.tabs(["📝 Analisis Teks Tunggal", "📁 Analisis CSV (Batch)"])

# --------------------------
# TAB 1: Teks Tunggal
# --------------------------
with tab1:
    st.subheader("📝 Analisis Satu Teks")

    default_text = (
        "Program makan bergizi gratis ini sangat membantu keluarga berpenghasilan rendah. "
        "Anak-anak jadi lebih semangat belajar karena kebutuhan gizinya terpenuhi."
    )

    user_text = st.text_area(
        "Masukkan teks opini/komentar:",
        value=default_text,
        height=150
    )

    if st.button("🔍 Analisis Sentimen (Teks Tunggal)", key="single"):
        if not user_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis sentimen..."):
                res = analyze_text(user_text)

            st.markdown("### 📊 Hasil Analisis")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Sentimen:**")
                if res["sentimen"] == "Positif":
                    st.success(f"✅ {res['sentimen']}")
                elif res["sentimen"] == "Negatif":
                    st.error(f"⚠️ {res['sentimen']}")
                else:
                    st.info(f"ℹ️ {res['sentimen']}")

            with col2:
                st.metric("Confidence", f"{res['score']*100:.2f} %")

            st.markdown("**Label mentah dari model:**")
            st.code(res["label_model"])

            st.markdown("**Teks yang dianalisis:**")
            st.write(res["original_text"])

# --------------------------
# TAB 2: CSV (Batch)
# --------------------------
with tab2:
    st.subheader("📁 Analisis Sentimen dari File CSV")

    st.write(
        """
        1. Siapkan file **CSV** yang berisi kolom teks (misalnya: `komentar`, `text`, `opini`).  
        2. Upload file-nya di bawah, pilih kolom teks yang akan dianalisis.  
        3. Klik **Analisis Sentimen CSV** untuk memproses.
        """
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview data:")
        st.dataframe(df.head())

        # pilih kolom teks
        text_column = st.selectbox(
            "Pilih kolom teks yang berisi komentar/opini:",
            options=df.columns.tolist()
        )

        if st.button("🔍 Analisis Sentimen (CSV)", key="csv"):
            if text_column is None:
                st.warning("Pilih kolom teks terlebih dahulu.")
            else:
                with st.spinner("Menganalisis seluruh baris..."):
                    results = []

                    for text in df[text_column].astype(str):
                        res = analyze_text(text)
                        results.append(res)

                    res_df = pd.DataFrame(results)

                    # gabungkan kembali ke dataframe asli
                    out_df = df.copy()
                    out_df["cleaned_text"] = res_df["cleaned_text"]
                    out_df["label_model"] = res_df["label_model"]
                    out_df["sentimen"] = res_df["sentimen"]
                    out_df["score"] = res_df["score"]

                st.success(f"Analisis selesai! Total baris: {len(out_df)}")

                st.markdown("### 📊 Ringkasan Sentimen")
                sent_counts = out_df["sentimen"].value_counts().reindex(
                    ["Positif", "Netral", "Negatif"]
                ).fillna(0)

                col1, col2 = st.columns(2)

                with col1:
                    st.bar_chart(sent_counts)

                with col2:
                    st.write("Jumlah per kategori:")
                    st.table(sent_counts.rename("Jumlah"))

                st.markdown("### 📄 Tabel Hasil (5 baris pertama)")
                st.dataframe(out_df.head())

                # Download hasil sebagai CSV
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Hasil (CSV)",
                    data=csv_bytes,
                    file_name="hasil_analisis_sentimen.csv",
                    mime="text/csv"
                )

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption(
    "Sistem Analisis Sentimen Berbasis NLP menggunakan Hugging Face Transformers (pre-trained RoBERTa, bagian dari keluarga BERT) serta spaCy untuk preprocessing. Dirancang untuk analisis opini publik secara cepat dan akurat."
)
