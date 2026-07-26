"""
Streamlit icinde hem embedding modelini hem de LLM modelini birlikte
kullanan minimal test. Ana uygulamadaki worker-thread deseniyle birebir
ayni. Amac: "Streamlit + embedding + chat" kombinasyonunun tek basina
soruna yol acip acmadigini izole etmek.

Calistirmak icin (proje klasorunde, .venv aktifken):
    streamlit run test_streamlit_dual.py
"""

import queue
import threading

import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
LLM_MODEL_NAME = "phi-4-mini"


class _FoundryWorker:
    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            func, args, kwargs, out = self._requests.get()
            try:
                result = func(*args, **kwargs)
                out.put(("ok", result))
            except BaseException as exc:
                out.put(("error", exc))

    def run(self, func, *args, **kwargs):
        out = queue.Queue(maxsize=1)
        self._requests.put((func, args, kwargs, out))
        status, value = out.get()
        if status == "error":
            raise value
        return value


@st.cache_resource(show_spinner=False)
def get_worker():
    return _FoundryWorker()


st.title("Streamlit + Embedding + Chat Testi")

if st.button("Test et"):
    log = st.empty()
    lines = []

    def add_line(text):
        lines.append(text)
        log.code("\n".join(lines))

    worker = get_worker()

    try:
        add_line("1 - Foundry Local başlatılıyor...")

        def _init():
            if FoundryLocalManager.instance is not None:
                manager = FoundryLocalManager.instance
            else:
                config = Configuration(app_name="foundry_local_streamlit_dual_test")
                FoundryLocalManager.initialize(config)
                manager = FoundryLocalManager.instance
            try:
                manager.download_and_register_eps(progress_callback=lambda n, p: None)
            except FoundryLocalException:
                pass
            return manager

        manager = worker.run(_init)
        add_line("2 - Foundry Local hazır (worker thread üzerinden).")

        embedding_model = worker.run(manager.catalog.get_model, EMBEDDING_MODEL_NAME)
        llm_model = worker.run(manager.catalog.get_model, LLM_MODEL_NAME)
        add_line("3 - Modeller bulundu.")

        if not embedding_model.is_cached:
            worker.run(embedding_model.download, lambda p: None)
        if not llm_model.is_cached:
            worker.run(llm_model.download, lambda p: None)
        add_line("4 - Modeller hazır.")

        worker.run(embedding_model.load)
        embedding_client = worker.run(embedding_model.get_embedding_client)
        add_line("5 - Embedding modeli yüklendi.")

        for i in range(5):
            response = worker.run(
                embedding_client.generate_embedding,
                f"Bu {i}. test cümlesidir, Foundry Local hakkında.",
            )
            add_line(f"5.{i} - embedding oluşturuldu (boyut {len(response.data[0].embedding)})")

        worker.run(llm_model.load)
        chat_client = worker.run(llm_model.get_chat_client)
        chat_client.settings.temperature = 0.1
        chat_client.settings.max_tokens = 100
        add_line("6 - LLM modeli de yüklendi (embedding hâlâ yüklü).")

        add_line("7 - Chat completion deneniyor...")
        completion = worker.run(
            chat_client.complete_chat,
            messages=[
                {"role": "system", "content": "Sen yardımcı bir asistansın."},
                {"role": "user", "content": "Merhaba, kısaca kendini tanıt."},
            ],
        )

        answer = completion.choices[0].message.content
        add_line("8 - CEVAP GELDİ:")
        add_line(answer)
        add_line("\n>>> SONUÇ: Streamlit + embedding + chat birlikte ÇALIŞTI. <<<")

        worker.run(embedding_model.unload)
        worker.run(llm_model.unload)

    except FoundryLocalException as error:
        add_line(f"HATA (FoundryLocalException): {error}")
        add_line("\n>>> SONUÇ: Streamlit + embedding + chat birlikte HATA VERDİ. Hipotez doğrulandı! <<<")

    except Exception as error:
        add_line(f"HATA (beklenmeyen): {type(error).__name__} - {error}")