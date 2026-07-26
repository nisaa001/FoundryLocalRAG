import re

import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APITimeoutError, OpenAI
from pypdf import PdfReader


PDF_PATH = "data/foundry_local_plan.pdf"
EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"
LLM_MODEL_NAME = "phi-4-mini"
FOUNDRY_BASE_URL = "http://127.0.0.1:58850/v1"
TOP_K = 3


def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1, dtype=np.float32)
    vector2 = np.array(vector2, dtype=np.float32)

    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector1, vector2) / denominator)


def split_into_sentences(text):
    """Metni kısa cevap adayı olabilecek cümlelere ayırır."""
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    if not cleaned_text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)

    return [
        sentence.strip()
        for sentence in sentences
        if 25 <= len(sentence.strip()) <= 500
    ]


def get_embedding(embedding_client, text):
    response = embedding_client.generate_embedding(text)

    if not response.data:
        raise ValueError("Embedding oluşturulamadı.")

    return response.data[0].embedding


# 1. PDF'yi oku
reader = PdfReader(PDF_PATH)
text_parts = []

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text_parts.append(page_text)

text = "\n".join(text_parts)

if not text.strip():
    raise ValueError("PDF içerisinden metin okunamadı.")


# 2. Metni parçalara ayır
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)

chunks = splitter.split_text(text)

if not chunks:
    raise ValueError("PDF metni parçalara ayrılamadı.")

print("PDF başarıyla okundu.")
print("Toplam parça sayısı:", len(chunks))


# 3. Foundry Local embedding modelini başlat
config = Configuration(app_name="foundry_local_rag")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_NAME)

if embedding_model is None:
    raise ValueError(
        f"Embedding modeli bulunamadı: {EMBEDDING_MODEL_NAME}"
    )

embedding_loaded = False

try:
    embedding_model.download()
    embedding_model.load()
    embedding_loaded = True

    embedding_client = embedding_model.get_embedding_client()
    print("\nEmbedding modeli hazır.")

    # 4. Tüm parçaların embeddinglerini oluştur
    chunk_embeddings = []
    print("\nParçaların embeddingleri oluşturuluyor...\n")

    for i, chunk in enumerate(chunks, start=1):
        chunk_embeddings.append(get_embedding(embedding_client, chunk))
        print(f"{i}/{len(chunks)} tamamlandı")

    print("\nTüm embeddingler hazır.")
    print("Toplam embedding:", len(chunk_embeddings))

    # 5. Kullanıcıdan soru al
    question = input("\nPDF hakkında bir soru sor: ").strip()

    if not question:
        raise ValueError("Soru boş bırakılamaz.")

    # 6. Türkçe sorunun embeddingini doğrudan oluştur
    # Qwen3 Embedding çok dilli olduğu için ayrı soru çevirisi gerekmiyor.
    query_text = (
        "Instruct: Retrieve the English document passage that directly "
        "answers the user's question.\n"
        f"Query: {question}"
    )
    question_embedding = get_embedding(embedding_client, query_text)

    # 7. En ilgili PDF parçalarını bul
    scores = []

    for index, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_embedding)
        scores.append((score, index))

    scores.sort(key=lambda item: item[0], reverse=True)
    top_results = scores[:TOP_K]

    print("\nEn ilgili parçalar:\n")

    for score, index in top_results:
        print("-" * 60)
        print(f"Benzerlik puanı: {score:.4f}")
        print("Parça numarası:", index)
        print(chunks[index])
        print()

    # 8. LLM kullanmadan en uygun cevap cümlesini seç
    # Böylece uzun ikinci model çağrısında yaşanan timeout ortadan kalkar.
    candidate_sentences = []
    seen_sentences = set()

    for _, index in top_results:
        for sentence in split_into_sentences(chunks[index]):
            normalized = sentence.casefold()

            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                candidate_sentences.append(sentence)

    if not candidate_sentences:
        raise ValueError("İlgili parçalardan cevap adayı cümle çıkarılamadı.")

    best_sentence = None
    best_sentence_score = -1.0

    print("\nBelgeden cevap cümlesi seçiliyor...")

    for sentence in candidate_sentences:
        sentence_embedding = get_embedding(embedding_client, sentence)
        sentence_score = cosine_similarity(
            question_embedding,
            sentence_embedding,
        )

        if sentence_score > best_sentence_score:
            best_sentence_score = sentence_score
            best_sentence = sentence

    english_answer = best_sentence.strip()

    print("Cevap cümlesi seçildi.")
    print(f"Cümle benzerlik puanı: {best_sentence_score:.4f}")
    print("\nİngilizce cevap:\n")
    print(english_answer)

    # 9. Embedding modelini kaldır; Phi-4 Mini için belleği boşalt
    embedding_model.unload()
    embedding_loaded = False
    print("\nEmbedding modeli bellekten kaldırıldı.")

    # 10. Yalnızca kısa cevabı Türkçeye çevir
    llm_client = OpenAI(
        base_url=FOUNDRY_BASE_URL,
        api_key="not-needed",
        timeout=90.0,
        max_retries=0,
    )

    translation_prompt = f"""
Translate the English sentence below into natural Turkish.

Rules:
- Return only the Turkish translation.
- Do not repeat the English sentence.
- Do not explain.
- Do not add or remove information.
- Keep Microsoft Foundry Local and RAG unchanged.

English sentence:
{english_answer}

Turkish translation:
"""

    print("\nTürkçe çeviri hazırlanıyor...")

    try:
        translation_response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate one short English sentence into "
                        "natural Turkish. Return only the translation."
                    ),
                },
                {
                    "role": "user",
                    "content": translation_prompt,
                },
            ],
            temperature=0.0,
            max_tokens=100,
        )

        turkish_answer = translation_response.choices[0].message.content

        if not turkish_answer:
            raise ValueError("Model boş Türkçe cevap döndürdü.")

        turkish_answer = turkish_answer.strip().strip('"')

        print("\nTürkçe cevap:\n")
        print(turkish_answer)

    except APITimeoutError:
        print("\nTürkçe çeviri 90 saniye içinde tamamlanamadı.")
        print("İngilizce cevap yukarıda başarıyla gösterildi.")

    except APIConnectionError as error:
        print("\nFoundry Local bağlantı hatası:")
        print(error)

    except Exception as error:
        print("\nTürkçe çeviri hazırlanırken hata oluştu:")
        print(type(error).__name__)
        print(error)

finally:
    if embedding_loaded:
        embedding_model.unload()
        print("\nEmbedding modeli bellekten kaldırıldı.")