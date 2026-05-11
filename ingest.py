import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

df = pd.read_csv("data/ragcare_qa.csv.csv")

documents = []

for index, row in df.iterrows():
    text = f"""
    Question:
    {row['Question']}

    Context:
    {row['Context']}

    Answer:
    {row['Text Answer']}

    """
    documents.append(
        Document(
            page_content=text,
            metadata={"source": f"row_{index}"}
        )
    )

print("Total Documents:", len(documents))
 
from langchain_community.document_loaders import JSONLoader
jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"
chatdoctor_loader = JSONLoader(
    file_path="data/chatdoctor5k.json",
    jq_schema=jq_schema,
    text_content=False
)
chatdoctor_docs = chatdoctor_loader.load()


from langchain_community.document_loaders import CSVLoader
format_loader = CSVLoader(
   file_path="data/format_dataset.csv"
)
format_docs = format_loader.load()


final_documents = (documents + chatdoctor_docs + format_docs)
print("Final Combined Docs:", len(final_documents))

recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100,
                                                    separators=["\n\n", "\n", " ", "", ".",",", ";"])
recursive_tokens = recursive_splitter.split_documents(final_documents)
 
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_db = FAISS.from_documents(
    documents=recursive_tokens,
    embedding=hf_embeddings
)
 
faiss_db.save_local("faiss_index")

if __name__ == "__main__":
    print("Done")
 