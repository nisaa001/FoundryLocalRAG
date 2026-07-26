from foundry_local_sdk import Configuration, FoundryLocalManager


config = Configuration(app_name="foundry_local_rag")
FoundryLocalManager.initialize(config)

manager = FoundryLocalManager.instance


model = manager.catalog.get_model("qwen3-embedding-0.6b")

if model is None:
    raise ValueError("Embedding modeli katalogda bulunamadı.")


print("Model bulundu:", model.alias)


model.download(
    lambda progress: print(
        f"\rModel indiriliyor: %{progress:.2f}",
        end="",
        flush=True
    )
)

print("\nModel indirildi.")


model.load()

print("Embedding modeli hazır.")


client = model.get_embedding_client()

text = "Foundry Local modelleri bilgisayarda çalıştırır."

response = client.generate_embedding(text)

embedding = response.data[0].embedding


print("Metin:", text)
print("Embedding uzunluğu:", len(embedding))
print("İlk 10 sayı:", embedding[:10])


model.unload()

print("Model bellekten kaldırıldı.")