"""
RAG'daki gercek promptun boyutunu taklit eden, ama PDF/embedding
icermeyen izole bir test. "Operation was cancelled" hatasinin
prompt uzunlugu/max_tokens ile mi, yoksa RAG akisiyla mi ilgili
oldugunu ayirt etmek icin.

Calistirmak icin (proje klasorunde, .venv aktifken):
    python test_chat_long.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

LLM_MODEL_NAME = "phi-4-mini"

FAKE_CONTEXT = """
[KAYNAK 1 | Sayfa 3]
Phase 1 - Foundational Learning (Weeks 1-2): Introduction to RAG concepts,
Foundry Local, embeddings, vector search, SQLite, and prompt engineering
fundamentals. Students will build a strong conceptual foundation in RAG
and local AI tools during this phase.

[KAYNAK 2 | Sayfa 4]
Week 1: RAG Concept and Local AI Setup. Topics and activities include an
introduction to Retrieval-Augmented Generation, understanding Foundry
Local and environment setup, and basic Python app structure. Students
install Foundry Local SDK on their machines and run a Hello Model test.

[KAYNAK 3 | Sayfa 5]
Milestones by end of Week 1: All students have Foundry Local installed
and working on their machines, have a basic project folder with a
main.py file, and can run a trivial Foundry Local inference to confirm
proper installation.
""".strip()

PROMPT = f"""
Asagidaki PDF baglamini kullanarak kullanicinin sorusunu Turkce yanitla.

Kurallar:
- Yalnizca verilen baglamdaki bilgilere dayan.
- Belgede bulunmayan bilgi ekleme.
- Cevabi dogal ve anlasilir Turkce ile yaz.
- Soru birden fazla madde istiyorsa maddeler halinde yanitla.
- Gereksiz uzun aciklama yapma.
- Cevabin sonunda kullandigin sayfalari yaz.

BAGLAM:
{FAKE_CONTEXT}

SORU:
1. haftada ogrenciler neler yapiyor?

CEVAP:
"""


def main():
    print("1 - Foundry Local baslatiliyor...", flush=True)
    config = Configuration(app_name="foundry_local_chat_long_test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    print("2 - Foundry Local hazir.", flush=True)

    print("2b - Execution provider(lar) kaydediliyor...", flush=True)
    try:
        result = manager.download_and_register_eps(
            progress_callback=lambda name, pct: print(f"   {name}: %{pct:.1f}", flush=True)
        )
        print(f"2c - EP kaydi sonucu: success={result.success}, kayitli={result.registered_eps}", flush=True)
    except FoundryLocalException as error:
        print(f"2c - EP kaydi basarisiz (devam ediliyor): {error}", flush=True)

    model = manager.catalog.get_model(LLM_MODEL_NAME)
    print(f"3 - Model bulundu: {model.alias}", flush=True)

    if not model.is_cached:
        print("4 - Model indiriliyor...", flush=True)
        model.download(lambda pct: print(f"   indirme: %{pct:.1f}", flush=True))
    else:
        print("4 - Model onbellekte.", flush=True)

    print("5 - Model yukleniyor...", flush=True)
    model.load()
    print("6 - Model yuklendi.", flush=True)

    try:
        chat_client = model.get_chat_client()
        chat_client.settings.temperature = 0.1
        chat_client.settings.max_tokens = 450

        print(f"7 - Prompt uzunlugu: {len(PROMPT)} karakter. Gonderiliyor...", flush=True)

        completion = chat_client.complete_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen yalnizca verilen belge baglamini kullanarak "
                        "Turkce cevap veren dikkatli bir RAG asistanisin."
                    ),
                },
                {"role": "user", "content": PROMPT},
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