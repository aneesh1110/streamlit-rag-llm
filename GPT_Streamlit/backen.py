import os
import fitz  # PyMuPDF
import pytesseract  # OCR for PNG images
from PIL import Image  # For opening PNG images
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Initialize Pinecone
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

# Set dimension for the model's output vector
dimension = 384  # Vector dimension for 'all-MiniLM-L6-v2'

# Initialize OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Function to extract text and images from a PDF
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


# Extract text from a PNG file
def extract_text_from_png(png_path):
    image = Image.open(png_path)
    text = pytesseract.image_to_string(image)
    return text


# Read files from folder and process them
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


# Index documents in Pinecone, with batching
def index_documents(documents, batch_size=100):
    upserted_data = []
    for i, doc in enumerate(documents):
        vector = model.encode(doc).tolist()

        # Ensure vector dimension matches the index dimension
        assert len(vector) == dimension, f"Vector dimension {len(vector)} does not match index dimension {dimension}"

        # Add vector to batch
        upserted_data.append((str(i), vector, {'text': doc}))

        # Upsert in batches
        if len(upserted_data) >= batch_size:
            print(f"Upserting batch of {batch_size} vectors...")
            index.upsert(vectors=upserted_data)
            upserted_data = []  # Clear the batch after upserting

    # Upsert any remaining data
    if upserted_data:
        print(f"Upserting final batch of {len(upserted_data)} vectors...")
        index.upsert(vectors=upserted_data)


# Perform semantic search using Pinecone
def semantic_search(query, top_k=1):
    query_vector = model.encode(query).tolist()

    # Query Pinecone for the most similar documents
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results['matches']


# Get answer from GPT-4 using OpenAI API
def get_answer_from_gpt4(context, query):
    messages = [
        {"role": "system",
         "content": "Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text and requires some latest information to be updated, print 'Sorry Not Sufficient context to answer query'"},
        {"role": "user", "content": context + '\n' + query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract the assistant's message from the response object
    message_content = response.choices[0].message.content
    return message_content


# Perform document-based QA using Pinecone and GPT-4
def document_based_qa(query):
    results = semantic_search(query)
    if not results:
        return "No relevant document found."

    if 'metadata' in results[0] and 'text' in results[0]['metadata']:
        context = results[0]['metadata']['text']
    else:
        return "No content found in the retrieved document."

    answer = get_answer_from_gpt4(context, query)
    return answer


# Upload folder and index documents
def upload_folder(folder_path):
    image_folder = os.path.join(folder_path, 'images')
    os.makedirs(image_folder, exist_ok=True)
    documents, _ = read_files_from_folder(folder_path, image_folder)
    index_documents(documents)
    return "Documents uploaded and indexed successfully."
