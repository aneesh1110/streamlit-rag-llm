GPT-4 Streamlit Chatbot with Document-Based QA

This project is a Streamlit-based chatbot powered by GPT-4, capable of answering user queries by retrieving relevant information from indexed documents. It leverages Pinecone for vector-based retrieval, BM25/Tf-IDF for keyword-based search, and OpenAI's GPT-4 for generating responses.

Features

Chat Interface: Allows users to have interactive conversations.

Document-Based QA: Retrieves answers from uploaded PDFs and images.

Hybrid Search: Uses semantic search (Pinecone) and keyword-based search (BM25/Tf-IDF) for better accuracy.

Persistent Chat History: Stores chat sessions using joblib.

Document Upload & Indexing: Users can upload folders containing PDFs and images, which are then indexed for retrieval.

Installation

1. Clone the Repository

git clone https://github.com/yourusername/your-repo.git
cd your-repo

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up API Keys

Create a .env file in the root directory and add:

PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key

Usage

1. Run the Application

streamlit run App.py

This will start the chatbot on a local Streamlit web interface.

2. Upload Documents for Indexing

The chatbot indexes PDFs and PNG images from the specified folder_path.

It extracts text using PyMuPDF (PDFs) and Tesseract OCR (images).

Indexed data is stored in Pinecone for retrieval.

3. Start a Chat

Users can enter a query in the chat input.

The system retrieves relevant content using hybrid search and generates a response using GPT-4.

Past chats are stored and can be accessed from the sidebar.

Folder Structure

.
├── GPT_Streamlit/
│   ├── App.py              # Streamlit chatbot application
│   ├── gpt_backend.py      # Document processing, retrieval & GPT-4 interaction
│   ├── data/               # Stores chat history
│   ├── images/Docs/        # Folder to upload PDF/PNG documents
│   ├── __pycache__/        # Compiled Python files (should be ignored in .gitignore)
├── .env                    # API keys (DO NOT SHARE)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation

Technologies Used

Python

Streamlit (UI)

OpenAI GPT-4 API (Natural Language Processing)

Pinecone (Vector Search)

SentenceTransformers (Embeddings)

BM25/Tf-IDF (Keyword Search)

PyMuPDF (PDF Processing)

Tesseract OCR (Image Text Extraction)

Joblib (Saving Chat History)

Troubleshooting

API Key Issues

If you encounter errors related to API keys, ensure:

.env contains the correct credentials.

You have sourced the .env variables (source .env).

Pinecone Index Not Found

Ensure you have created the index before running queries:

pc.create_index(name='icici-reports', dimension=384, metric='cosine')

Streamlit Not Running Properly

Try clearing cached files:

streamlit cache clear

Future Enhancements

User Authentication (to save personalized chat history)

Multi-document summarization for better context retrieval

Improved UI/UX with additional chatbot customization

License

This project is open-source under the MIT License.

Author

Aneesh Vinod
