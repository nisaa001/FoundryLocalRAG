from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("data/foundry_local_plan.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Parça sayısı:", len(chunks))

print()

print(chunks[0])