"""
En minimal Streamlit testi: PDF yok, embedding yok, worker thread yok.
Sadece bir buton, tiklaninca phi-4-mini'ye "merhaba" diyor.
Amac: sorunun Streamlit'in kendi sureci (Tornado/asyncio) icinde
calismaktan mi kaynaklandigini izole etmek.

Calistirmak icin (proje klasorunde, .venv aktifken):
    streamlit run test_streamlit_minimal.py
"""

import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

LLM_MODEL_NAME = "phi-4-mini"

st.title("Minimal Foundry Local Testi")

if st.button("Modeli yükle ve test et"):
    log = st.empty()
    lines = []

    def add_line(text):
        lines.append(text)
        log.code("\n".join(lines))

    try:
        add_line("1 - Foundry Local başlatılıyor...")

        if FoundryLocalManager.instance is not None:
            manager = FoundryLocalManager.instance
        else:
            config = Configuration(app_name="foundry_local_minimal_streamlit_test")
            FoundryLocalManager.initialize(config)
            manager = FoundryLocalManager.instance

        add_line("2 - Foundry Local hazır.")

        try:
            manager.download_and_register_eps(
                progress_callback=lambda name, pct: None
            )
            add_line("2b - EP kaydı tamam.")
        except FoundryLocalException as error:
            add_line(f"2b - EP kaydı başarısız (devam): {error}")

        model = manager.catalog.get_model(LLM_MODEL_NAME)
        add_line(f"3 - Model bulundu: {model.alias}")

        if not model.is_cached:
            add_line("4 - Model indiriliyor...")
            model.download(lambda pct: None)
        add_line("5 - Model önbellekte/indirildi.")

        model.load()
        add_line("6 - Model yüklendi.")

        chat_client = model.get_chat_client()
        chat_client.settings.temperature = 0.1
        chat_client.settings.max_tokens = 50

        add_line("7 - Basit bir soru gönderiliyor...")

        completion = chat_client.complete_chat(
            messages=[
                {"role": "user", "content": "Merhaba, sen kimsin? Kısaca cevap ver."},
            ]
        )

        answer = completion.choices[0].message.content
        add_line("8 - CEVAP GELDİ:")
        add_line(answer)
        add_line("\n>>> SONUÇ: Streamlit içinde de çalıştı. Sorun Streamlit'in kendisi değil. <<<")

        model.unload()

    except FoundryLocalException as error:
        add_line(f"HATA (FoundryLocalException): {error}")
        add_line("\n>>> SONUÇ: Streamlit içinde HATA verdi. Sorun Streamlit sürecinin kendisiyle ilgili. <<<")

    except Exception as error:
        add_line(f"HATA (beklenmeyen): {type(error).__name__} - {error}")