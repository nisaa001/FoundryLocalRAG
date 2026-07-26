"""
PDF'teki gibi COK SAYIDA (40 tane) embedding cagrisi yaptiktan sonra
chat completion deniyoruz. Amac: art arda cok sayida native cagrinin
sonraki chat completion'i bozup bozmadigini izole etmek.

Calistirmak icin (proje klasorunde, .venv aktifken):
    python test_chat_many.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
LLM_MODEL_NAME = "phi-4-mini"
NUM_FAKE_CHUNKS = 40


def main():
    print("1 - Foundry Local baslatiliyor...", flush=True)
    config = Configuration(app_name="foundry_local_many_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("2 - Foundry Local hazir.", flush=True)

    try:
        manager.download_and_register_eps(
            progress_callback=lambda name, pct: print(f"   EP indirme ({name}): %{pct:.1f}", flush=True)
        )
        print("2b - EP kaydi tamam.", flush=True)
    except FoundryLocalException as error:
        print(f"2b - EP kaydi basarisiz (devam): {error}", flush=True)

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)
    llm_model = manager.catalog.get_model(LLM_MODEL_NAME)

    if not embedding_model.is_cached:
        print("2c - Embedding modeli indiriliyor...", flush=True)
        embedding_model.download(lambda pct: print(f"   embedding indirme: %{pct:.1f}", flush=True))
    if not llm_model.is_cached:
        print("2d - LLM modeli indiriliyor...", flush=True)
        llm_model.download(lambda pct: print(f"   llm indirme: %{pct:.1f}", flush=True))
    print("3 - Modeller hazir.", flush=True)

    embedding_model.load()
    print("4 - Embedding modeli yuklendi.", flush=True)

    embedding_client = embedding_model.get_embedding_client()

    print(f"5 - {NUM_FAKE_CHUNKS} adet sahte parca icin embedding cikariliyor...", flush=True)
    for i in range(NUM_FAKE_CHUNKS):
        text = (
            f"Bu {i}. numarali test parcasi. Foundry Local, RAG, "
            f"embedding ve vektor arama ile ilgili ornek bir metin "
            f"parcasidir. Numara: {i}."
        )
        response = embedding_client.generate_embedding(text)
        if not response.data:
            print(f"   HATA: {i}. embedding bos dondu!", flush=True)
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{NUM_FAKE_CHUNKS} tamamlandi", flush=True)

    print("6 - Tum embeddingler tamamlandi.", flush=True)

    print("7 - LLM modeli yukleniyor (embedding hala yuklu)...", flush=True)
    llm_model.load()
    print("7b - LLM modeli yuklendi.", flush=True)

    try:
        chat_client = llm_model.get_chat_client()
        chat_client.settings.temperature = 0.1
        chat_client.settings.max_tokens = 450

        print("8 - Chat completion deneniyor...", flush=True)

        completion = chat_client.complete_chat(
            messages=[
                {"role": "system", "content": "Sen yardimci bir asistansin."},
                {"role": "user", "content": "Merhaba, kisaca kendini tanit."},
            ]
        )

        answer = completion.choices[0].message.content
        print("\n9 - CEVAP GELDI:\n")
        print(answer)
        print("\n>>> SONUC: Cok sayida embedding cagrisindan sonra da chat CALISTI. Sorun baska yerde. <<<")

    except FoundryLocalException as error:
        print("\nHATA (FoundryLocalException):")
        print(error)
        print("\n>>> SONUC: Cok sayida embedding cagrisindan sonra chat HATA VERDI. Hipotez dogrulandi. <<<")

    except Exception as error:
        print("\nHATA (beklenmeyen):")
        print(type(error).__name__, "-", error)

    finally:
        embedding_model.unload()
        llm_model.unload()
        print("\n10 - Modeller bellekten kaldirildi.", flush=True)


if __name__ == "__main__":
    main()