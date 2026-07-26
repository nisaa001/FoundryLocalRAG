"""
RAG hattindan tamamen bagimsiz, sadece phi-4-mini'nin sohbet
tamamlamasinin bu makinede calisip calismadigini test eden minimal script.

Calistirmak icin (proje klasorunde, .venv aktifken):
    python test_chat.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

LLM_MODEL_NAME = "phi-4-mini"


def main():
    print("1 - Foundry Local baslatiliyor...", flush=True)
    config = Configuration(app_name="foundry_local_chat_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("2 - Foundry Local hazir.", flush=True)

    print("2b - Execution provider(lar) kaydediliyor (donanim hizlandirma)...", flush=True)
    try:
        result = manager.download_and_register_eps(
            progress_callback=lambda name, pct: print(f"\r   {name}: %{pct:.1f}", end="", flush=True)
        )
        print()
        print(f"2c - EP kaydi sonucu: success={result.success}, kayitli={result.registered_eps}", flush=True)
    except FoundryLocalException as error:
        print(f"2c - EP kaydi basarisiz oldu (devam ediliyor): {error}", flush=True)

    model = manager.catalog.get_model(LLM_MODEL_NAME)
    if model is None:
        raise ValueError(f"Model bulunamadi: {LLM_MODEL_NAME}")
    print(f"3 - Model katalogda bulundu: {model.alias}", flush=True)

    if not model.is_cached:
        print("4 - Model indiriliyor (ilk seferde gerekli)...", flush=True)
        model.download(lambda pct: print(f"\r   indirme: %{pct:.1f}", end="", flush=True))
        print()
    else:
        print("4 - Model zaten onbellekte, indirme atlaniyor.", flush=True)

    print("5 - Model yukleniyor...", flush=True)
    model.load()
    print("6 - Model yuklendi. Sohbet istemcisi olusturuluyor...", flush=True)

    try:
        chat_client = model.get_chat_client()
        chat_client.settings.temperature = 0.1
        chat_client.settings.max_tokens = 50

        print("7 - Basit bir soru gonderiliyor: 'Merhaba, sen kimsin?'", flush=True)

        completion = chat_client.complete_chat(
            messages=[
                {"role": "user", "content": "Merhaba, sen kimsin? Kisaca cevap ver."},
            ]
        )

        answer = completion.choices[0].message.content
        print("\n8 - CEVAP GELDI:\n")
        print(answer)

    except FoundryLocalException as error:
        print("\nHATA (FoundryLocalException):")
        print(error)

    except Exception as error:
        print("\nHATA (beklenmeyen):")
        print(type(error).__name__, "-", error)

    finally:
        model.unload()
        print("\n9 - Model bellekten kaldirildi.", flush=True)


if __name__ == "__main__":
    main()