import os
import fitz  # PyMuPDF
import pytesseract  # OCR for PNG images
import tiktoken
from PIL import Image  # For opening PNG images
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Initialize Pinecone and Sentence Transformer
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "icici-reports"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_and_images_from_pdf(file_path, image_folder):
    doc = fitz.open(file_path)
    text = ""
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_path = os.path.join(image_folder,
                                      f"{os.path.basename(file_path).split('.')[0]}_page_{page_num + 1}_img_{img_index + 1}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            images.append(image_path)

    return text, images


def extract_text_from_png(png_path):
    image = Image.open(png_path)
    text = pytesseract.image_to_string(image)
    return text


def read_files_from_folder(folder_path, image_folder):
    documents = []
    all_images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            text, images = extract_text_and_images_from_pdf(file_path, image_folder)
            documents.append(text)
            all_images.extend(images)
        elif filename.endswith('.png'):
            text = extract_text_from_png(file_path)
            documents.append(text)
            all_images.append(file_path)
    return documents, all_images


def index_documents(documents):
    upserted_data = []
    for i, doc in enumerate(documents):
        vector = model.encode(doc).tolist()
        upserted_data.append((str(i), vector, {'text': doc}))
    index.upsert(vectors=upserted_data)


def semantic_search(query, top_k=1):
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results['matches']


# Summarization function using GPT-3.5-turbo
def summarize_text(text, max_tokens=2000):
    summary_prompt = f"Summarize the following text in {max_tokens} tokens:\n\n{text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": summary_prompt}
        ]
    )

    summary = response.choices[0].message.content
    return summary


# Split context into manageable chunks
def split_context_into_chunks(context, max_tokens=4000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(context)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunks.append(enc.decode(tokens[i:i + max_tokens]))

    return chunks


def get_answer_from_gpt4(context, query):
    # Check token limit before sending the request
    enc = tiktoken.encoding_for_model("gpt-4")
    total_tokens = len(enc.encode(context + '\n' + query))

    if total_tokens > 8192:
        context = summarize_text(context)  # Summarize the context if it's too long
        total_tokens = len(enc.encode(context + '\n' + query))
        if total_tokens > 8192:
            return "Context is too large, even after summarization."

    messages = [
        {"role": "system",
         "content": "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text and requires some latest information to be updated, print 'Sorry Not Sufficient context to answer query'"},
        {"role": "user", "content": context + '\n' + query}
    ]

    # Correct the API call here
    response = openai.ChatCompletion.create(  # Corrected here
        model="gpt-4",
        messages=messages
    )

    message_content = response.choices[0].message.content
    return message_content

def document_based_qa(query):
    results = semantic_search(query)
    if not results:
        return "No relevant document found."

    if 'metadata' in results[0] and 'text' in results[0]['metadata']:
        context = results[0]['metadata']['text']
    else:
        return "No content found in the retrieved document."

    # If the context is still too large, break it into chunks
    enc = tiktoken.encoding_for_model("gpt-4")
    total_tokens = len(enc.encode(context))

    if total_tokens > 8192:
        context_chunks = split_context_into_chunks(context)
        # Process chunks and collect answers
        answers = []
        for chunk in context_chunks:
            answer = get_answer_from_gpt4(chunk, query)
            answers.append(answer)
        return '\n'.join(answers)

    # If the context is within the limit, get the answer directly
    answer = get_answer_from_gpt4(context, query)
    return answer


# Add the missing upload_folder function
def upload_folder(folder_path):
    image_folder = os.path.join(folder_path, 'images')
    os.makedirs(image_folder, exist_ok=True)
    documents, _ = read_files_from_folder(folder_path, image_folder)
    index_documents(documents)
    return "Documents uploaded and indexed successfully."
