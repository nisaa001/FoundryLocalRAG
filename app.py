from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:58850/v1",
    api_key="not-needed"
)

# Dosyayı oku
with open("data/bilgi.txt", "r", encoding="utf-8") as f:
    context = f.read()

question = input("Sorunuzu yazın: ")

prompt = f"""
Aşağıdaki bilgileri kullanarak cevap ver.

Bilgiler:
{context}

Soru:
{question}
"""

response = client.chat.completions.create(
    model="phi-4-mini",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nCevap:")
print(response.choices[0].message.content)