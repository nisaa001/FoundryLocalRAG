import hashlib
import io
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
LLM_MODEL_NAME = "phi-4-mini"
TOP_K = 1

_MANAGER_LOCK = threading.Lock()
_FOUNDRY_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="foundry-worker",
)


def cosine_similarity(vector1, vector2):
    vector1 = np.asarray(vector1, dtype=np.float32)
    vector2 = np.asarray(vector2, dtype=np.float32)

    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector1, vector2) / denominator)


def get_embedding(embedding_client, text):
    response = embedding_client.generate_embedding(text)

    if not response.data:
        raise ValueError("Embedding oluşturulamadı.")

    return response.data[0].embedding


def _create_foundry_resources_on_worker():
    """
    Foundry Local yöneticisini ve iki modeli yalnızca bir kez hazırlar.

    Test dosyasında çalışan yöntem korunur:
    embedding modeli yüklüyken chat modeli de yüklenir ve ikisi birlikte
    bellekte tutulur. Böylece her soru sırasında model kaldırıp yeniden
    yükleme yapılmaz.
    """
    with _MANAGER_LOCK:
        manager = None

        try:
            manager = FoundryLocalManager.instance
        except Exception:
            manager = None

        if manager is None:
            config = Configuration(
                app_name="foundry_local_streamlit_rag"
            )

            try:
                FoundryLocalManager.initialize(config)
            except Exception as error:
                if "already been initialized" not in str(error).casefold():
                    raise RuntimeError(
                        "Foundry Local başlatılamadı. "
                        f"Orijinal hata: {type(error).__name__}: {error}"
                    ) from error

            try:
                manager = FoundryLocalManager.instance
            except Exception as error:
                raise RuntimeError(
                    "Foundry Local başlatıldı fakat manager alınamadı."
                ) from error

        if manager is None:
            raise RuntimeError(
                "FoundryLocalManager.instance None döndürdü. "
                "Terminali Ctrl+C ile kapatıp yeniden başlat."
            )

        embedding_model = manager.catalog.get_model(
            EMBEDDING_MODEL_NAME
        )
        llm_model = manager.catalog.get_model(LLM_MODEL_NAME)

        if embedding_model is None:
            raise ValueError(
                f"Embedding modeli bulunamadı: {EMBEDDING_MODEL_NAME}"
            )

        if llm_model is None:
            raise ValueError(
                f"Dil modeli bulunamadı: {LLM_MODEL_NAME}"
            )

        embedding_model.download()
        llm_model.download()

        embedding_model.load()
        embedding_client = embedding_model.get_embedding_client()

        # Testte çalışan düzene uygun olarak embedding modeli bellekteyken
        # LLM modeli de yüklenir.
        llm_model.load()

        llm_client = llm_model.get_chat_client()
        llm_client.settings.temperature = 0.1
        llm_client.settings.max_tokens = 120

        return {
            "manager": manager,
            "embedding_model": embedding_model,
            "embedding_client": embedding_client,
            "llm_model": llm_model,
            "llm_client": llm_client,
        }


@st.cache_resource(show_spinner=False)
def get_foundry_resources():
    """
    Foundry Local'ın başlatılması ve model istemcilerinin oluşturulması
    tek ve kalıcı worker thread üzerinde yapılır.
    """
    return _FOUNDRY_EXECUTOR.submit(
        _create_foundry_resources_on_worker
    ).result()


def run_foundry(function, *args, **kwargs):
    """Tüm Foundry SDK çağrılarını aynı worker thread üzerinde çalıştırır."""
    return _FOUNDRY_EXECUTOR.submit(
        function,
        *args,
        **kwargs,
    ).result()


def build_retrieval_fallback(question, sources):
    """
    Phi-4 Mini yerel cihazda iptal edilirse uygulamanın hata vermek
    yerine bulunan belge bölümünü göstermesini sağlar.
    """
    if not sources:
        return "Bu bilgi yüklenen belgede bulunamadı."

    source = sources[0]
    text = re.sub(r"\s+", " ", source["text"]).strip()

    if len(text) > 1400:
        text = text[:1400].rsplit(" ", 1)[0] + "…"

    return (
        "Phi-4 Mini bu isteği cihazda tamamlayamadı; ancak soruyla en "
        "ilgili belge bölümü başarıyla bulundu:\n\n"
        f"> {text}\n\n"
        f"**Kaynak: Sayfa {source['page']}**"
    )


class LocalRAG:
    def __init__(self, pdf_bytes):
        self.pdf_bytes = pdf_bytes
        self.chunks, self.chunk_pages, self.page_count = (
            self._read_and_split_pdf()
        )

        resources = get_foundry_resources()
        self.embedding_client = resources["embedding_client"]
        self.llm_client = resources["llm_client"]

        self.chunk_embeddings = self._create_chunk_embeddings()

    def _read_and_split_pdf(self):
        reader = PdfReader(io.BytesIO(self.pdf_bytes))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )

        chunks = []
        chunk_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()

            if not page_text or not page_text.strip():
                continue

            for chunk in splitter.split_text(page_text):
                chunks.append(chunk)
                chunk_pages.append(page_number)

        if not chunks:
            raise ValueError(
                "PDF içerisinden okunabilir metin çıkarılamadı."
            )

        return chunks, chunk_pages, len(reader.pages)

    def _create_chunk_embeddings(self):
        def create_all():
            embeddings = []

            for index, chunk in enumerate(self.chunks, start=1):
                embeddings.append(
                    get_embedding(self.embedding_client, chunk)
                )
                print(
                    f"Embedding hazırlanıyor: {index}/{len(self.chunks)}",
                    flush=True,
                )

            return embeddings

        return run_foundry(create_all)

    def _retrieve(self, question):
        query_text = (
            "Instruct: Retrieve the English document passages that "
            "directly answer the user's Turkish question.\n"
            f"Query: {question}"
        )

        question_embedding = run_foundry(
            get_embedding,
            self.embedding_client,
            query_text,
        )

        scored_chunks = []

        for index, chunk_embedding in enumerate(
            self.chunk_embeddings
        ):
            score = cosine_similarity(
                question_embedding,
                chunk_embedding,
            )
            scored_chunks.append(
                {
                    "index": index,
                    "page": self.chunk_pages[index],
                    "score": score,
                    "text": self.chunks[index],
                }
            )

        scored_chunks.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored_chunks[:TOP_K]

    def answer(self, question):
        retrieved_chunks = self._retrieve(question)

        context = "\n\n".join(
            f"[KAYNAK {rank} | Sayfa {item['page']}]\n"
            f"{item['text']}"
            for rank, item in enumerate(retrieved_chunks, start=1)
        )

        prompt = f"""
Aşağıdaki PDF bağlamını kullanarak kullanıcının sorusunu Türkçe yanıtla.

Kurallar:
- Yalnızca verilen bağlamdaki bilgilere dayan.
- Belgede bulunmayan bilgi ekleme.
- Doğal, açık ve kısa Türkçe kullan.
- Birden fazla madde isteniyorsa maddeler halinde yanıtla.
- Teknik adları değiştirme.
- Bağlam soruyu yanıtlamaya yetmiyorsa tam olarak:
  "Bu bilgi yüklenen belgede bulunamadı." yaz.
- Cevabın sonunda kullandığın sayfaları belirt.

BAĞLAM:
{context}

SORU:
{question}

CEVAP:
"""

        try:
            response = run_foundry(
                self.llm_client.complete_chat,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sen sadece verilen PDF bağlamına dayalı "
                            "Türkçe cevap veren dikkatli bir RAG asistanısın."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

        except Exception as error:
            error_text = str(error).casefold()

            if (
                "operation was cancelled" in error_text
                or "error during chat completion" in error_text
            ):
                return {
                    "answer": build_retrieval_fallback(
                        question,
                        retrieved_chunks,
                    ),
                    "sources": retrieved_chunks,
                    "used_fallback": True,
                }

            raise RuntimeError(
                "Phi-4 Mini cevap oluşturamadı. "
                f"Orijinal hata: {type(error).__name__}: {error}"
            ) from error

        answer_text = response.choices[0].message.content

        if not answer_text:
            return {
                "answer": build_retrieval_fallback(
                    question,
                    retrieved_chunks,
                ),
                "sources": retrieved_chunks,
                "used_fallback": True,
            }

        return {
            "answer": answer_text.strip(),
            "sources": retrieved_chunks,
            "used_fallback": False,
        }


def load_rag(pdf_bytes):
    return LocalRAG(pdf_bytes)


def show_source_details(sources):
    with st.expander("📄 Kaynak ve teknik ayrıntılar"):
        for rank, source in enumerate(sources, start=1):
            st.markdown(
                f"### Kaynak {rank} — Sayfa {source['page']}"
            )
            st.write(f"**Parça numarası:** {source['index']}")
            st.write(
                f"**Benzerlik puanı:** "
                f"%{source['score'] * 100:.1f}"
            )
            st.write(source["text"])

            if rank != len(sources):
                st.divider()


st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 18px;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        margin: 0.4rem 0 0 0;
        opacity: 0.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🤖 Local RAG Assistant</h1>
        <p>
            Microsoft Foundry Local ile PDF dosyalarınız hakkında
            çevrimdışı soru sorun.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📁 PDF yükle")

    uploaded_file = st.file_uploader(
        "Bir PDF dosyası seç",
        type=["pdf"],
        accept_multiple_files=False,
    )

    st.caption(
        "İlk PDF hazırlığı, parça sayısına göre biraz sürebilir."
    )

    if st.button("🗑️ Sohbeti temizle", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if uploaded_file is None:
    st.info("Başlamak için sol menüden bir PDF dosyası yükle.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("active_pdf_hash") != pdf_hash:
    st.session_state.active_pdf_hash = pdf_hash
    st.session_state.messages = []
    st.session_state.pop("rag_instance", None)
    st.session_state.pop("rag_pdf_hash", None)

try:
    if (
        "rag_instance" not in st.session_state
        or st.session_state.get("rag_pdf_hash") != pdf_hash
    ):
        with st.spinner(
            "PDF okunuyor ve embeddingler hazırlanıyor..."
        ):
            st.session_state.rag_instance = load_rag(pdf_bytes)
            st.session_state.rag_pdf_hash = pdf_hash

    rag = st.session_state.rag_instance

except Exception as error:
    st.session_state.pop("rag_instance", None)
    st.session_state.pop("rag_pdf_hash", None)
    st.error("PDF hazırlanırken bir hata oluştu.")
    st.exception(error)
    st.stop()

with st.sidebar:
    st.success("PDF hazır")
    st.write(f"**Dosya:** {uploaded_file.name}")
    st.write(f"**Sayfa:** {rag.page_count}")
    st.write(f"**Metin parçası:** {len(rag.chunks)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                show_source_details(sources)

question = st.chat_input(
    "Yüklediğin PDF hakkında bir soru sor..."
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "İlgili bölümler bulunuyor ve cevap hazırlanıyor..."
        ):
            try:
                result = rag.answer(question)
                answer = result.get("answer")
                sources = result.get("sources", [])

                if not answer:
                    raise ValueError("Cevap metni alınamadı.")

                st.markdown(answer)

                if sources:
                    show_source_details(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as error:
                st.error(
                    "Cevap hazırlanırken bir hata oluştu."
                )
                st.exception(error)