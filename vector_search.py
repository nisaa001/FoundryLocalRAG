from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from foundry_local_sdk import Configuration, FoundryLocalManager
import numpy as np


# PDF'yi oku
reader = PdfReader("data/foundry_local_plan.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# Metni parçalara ayır
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Toplam parça sayısı:", len(chunks))


# Foundry Local'ı başlat
config = Configuration(app_name="foundry_local_rag")
FoundryLocalManager.initialize(config)

manager = FoundryLocalManager.instance

model = manager.catalog.get_model("qwen3-embedding-0.6b")

if model is None:
    raise ValueError("Embedding modeli bulunamadı.")


model.download()
model.load()

client = model.get_embedding_client()


# Tüm chunk'ların embedding'ini oluştur
chunk_embeddings = []

print("Parçaların embeddingleri oluşturuluyor...")

for index, chunk in enumerate(chunks):
    response = client.generate_embedding(chunk)
    embedding = response.data[0].embedding

    chunk_embeddings.append(embedding)

    print(f"{index + 1}/{len(chunks)} tamamlandı")


# Kullanıcıdan soru al
question = input("\nPDF hakkında bir soru sor: ")

question_response = client.generate_embedding(question)
question_embedding = question_response.data[0].embedding


# Cosine similarity
def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


# Soruyla tüm parçaları karşılaştır
scores = []

for index, chunk_embedding in enumerate(chunk_embeddings):
    similarity = cosine_similarity(
        question_embedding,
        chunk_embedding
    )

    scores.append((similarity, index))


# En yüksek puanlı parçaları sırala
scores.sort(reverse=True)

top_results = scores[:5]


print("\nEn ilgili parçalar:\n")

for score, index in top_results:
    print("-" * 60)
    print("Benzerlik puanı:", score)
    print("Parça numarası:", index)
    print(chunks[index])
    print()


model.unload()