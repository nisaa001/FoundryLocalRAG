"""
Embedding modeli VE LLM modelini AYNI ANDA yukleyip, sonra chat
completion deniyoruz. Amac: gercek RAG akisindaki "iki model ayni
anda yuklu" durumunun "Operation was cancelled" hatasina yol acip
acmadigini izole sekilde dogrulamak.

Calistirmak icin (proje klasorunde, .venv aktifken):
    python test_chat_dual.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
LLM_MODEL_NAME = "phi-4-mini"


def main():
    print("1 - Foundry Local baslatiliyor...", flush=True)
    config = Configuration(app_name="foundry_local_dual_test")
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
    print("3 - Modeller katalogda bulundu.", flush=True)

    if not embedding_model.is_cached:
        print("4 - Embedding modeli indiriliyor...", flush=True)
        embedding_model.download(lambda pct: print(f"   embedding indirme: %{pct:.1f}", flush=True))
    if not llm_model.is_cached:
        print("4b - LLM modeli indiriliyor...", flush=True)
        llm_model.download(lambda pct: print(f"   llm indirme: %{pct:.1f}", flush=True))
    print("5 - Modeller onbellekte/indirildi.", flush=True)

    print("6 - Embedding modeli yukleniyor...", flush=True)
    embedding_model.load()
    print("6b - Embedding modeli YUKLU. Bir embedding olusturuluyor...", flush=True)

    embedding_client = embedding_model.get_embedding_client()
    emb_response = embedding_client.generate_embedding("Bu bir test cumlesidir.")
    print(f"6c - Embedding olusturuldu (boyut: {len(emb_response.data[0].embedding)}).", flush=True)
    print("6d - Embedding modeli UNLOAD EDILMIYOR, yuklu kalacak (RAG'daki gibi).", flush=True)

    print("7 - LLM modeli yukleniyor (embedding hala yuklu)...", flush=True)
    llm_model.load()
    print("7b - LLM modeli de YUKLU. Simdi ikisi de ayni anda yuklu.", flush=True)

    try:
        chat_client = llm_model.get_chat_client()
        chat_client.settings.temperature = 0.1
        chat_client.settings.max_tokens = 450

        print("8 - Chat completion deneniyor (iki model ayni anda yukluyken)...", flush=True)

        completion = chat_client.complete_chat(
            messages=[
                {"role": "system", "content": "Sen yardimci bir asistansin."},
                {"role": "user", "content": "Merhaba, kisaca kendini tanit."},
            ]
        )

        answer = completion.choices[0].message.content
        print("\n9 - CEVAP GELDI:\n")
        print(answer)
        print("\n>>> SONUC: Iki model ayni anda yukluyken de calisti. Sorun baska yerde. <<<")

    except FoundryLocalException as error:
        print("\nHATA (FoundryLocalException):")
        print(error)
        print("\n>>> SONUC: Iki model ayni anda yukluyken hata VERDI. Hipotez dogrulandi. <<<")

    except Exception as error:
        print("\nHATA (beklenmeyen):")
        print(type(error).__name__, "-", error)

    finally:
        embedding_model.unload()
        llm_model.unload()
        print("\n10 - Modeller bellekten kaldirildi.", flush=True)


if __name__ == "__main__":
    main()