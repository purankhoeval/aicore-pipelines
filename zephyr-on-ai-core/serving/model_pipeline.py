import torch
import transformers
import traceback  # Add this import
import sys, os
import pysftp
import huggingface_hub
from pprint import pprint
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import MetadataMode
from llama_index.postprocessor import MetadataReplacementPostProcessor
from llama_index.vector_stores import ChromaVectorStore,VectorStoreQuery
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    Document
)
from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.embeddings import HuggingFaceEmbedding
import chromadb
from chromadb.utils import embedding_functions
from llama_index import download_loader
from pathlib import Path
import pytesseract
import pdf2image
from pdf2image import convert_from_path
import fitz

transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

os.environ["TRANSFORMERS_CACHE"] = "shared/IMR/llm2023/cache"

def download_pdf_from_sftp(sftp_host, sftp_username, sftp_password, sftp_port, remote_path, local_path):
    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None  # Disable host key checking (not recommended for production)

    with pysftp.Connection(sftp_host, username=sftp_username, password=sftp_password, port=sftp_port, cnopts=cnopts) as sftp:
        # Create the local directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        # sftp.get(remote_path, local_path)
        remote_files = sftp.listdir(remote_path)

        # Download each file individually
        for remote_file in remote_files:
            remote_file_path = os.path.join(remote_path, remote_file)
            local_file_path = os.path.join(local_path, remote_file)
            sftp.get(remote_file_path, local_file_path)

# SFTP details
sftp_host = 'mysftp.e613f23.kyma.ondemand.com'
sftp_port = 2222
sftp_username = 'puran'
sftp_password = 'sappass'
remote_pdf_path = '/upload/'
local_pdf_path = './data/'

# Download PDF from SFTP
download_pdf_from_sftp(sftp_host, sftp_username, sftp_password, sftp_port, remote_pdf_path, local_pdf_path)

ImageReader = download_loader("ImageReader")

# Use the model_kwargs to pass options to the parser function
loader = ImageReader(text_type="plain_text")

poppler_path = 'C:\\Program Files\\poppler-23.11.0\\Library\\bin'

image_paths = []
documents = []

def is_text_based_pdf(pdf_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)

        # Iterate through each page and check for text
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text = page.get_text()

            # If text is found on any page, it's likely a text-based PDF
            if text.strip():
                return True

        # No text found on any page, it might be an image-based PDF
        return False

    except Exception as e:
        # Handle exceptions (e.g., if the PDF is encrypted or malformed)
        print(f"Error checking PDF: {e}")
        return False

def process_pdf_file(pdf_path):
    is_text_based = is_text_based_pdf(pdf_path)

    # Check if the PDF is text-based or image-based
    if is_text_based:
        directory_reader = SimpleDirectoryReader(input_files=[pdf_path])

        # Load data from the specified file path
        documentspdf = directory_reader.load_data()

        # Create a llamaindex Document from ImageDocument
        doc1 = documentspdf[0]
        doc1 = Document(doc_id=doc1.id_, text=doc1.text, metadata=doc1.metadata)
        documents.append(doc1)
        doc1 = []
    else:
        print("The PDF is image-based.")

        # Convert the PDF to images
        pdf2image.poppler_path = 'C:\\Program Files\\poppler-23.11.0\\Library\\bin'  # Set the Poppler path
        images = convert_from_path(pdf_path, poppler_path=pdf2image.poppler_path)

        # Save each image to a file and load as ImageDocuments
        for i, image in enumerate(images):
            image_path = Path(f"./data/temp/page_{i}.png")
            image.save(image_path)
            image_paths.append(image_path)
            doc = loader.load_data(file=image_path)
            documents.extend(doc)

# Process files in the directory
def process_files_in_directory(directory_path):
    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check file extension
        _, file_extension = os.path.splitext(filename)

        # Call the appropriate function based on the file type
        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            local_image_path = "./data/remit.png"
            ImageReader = download_loader("ImageReader")
            # Use the model_kwargs to pass options to the parser function
            loader = ImageReader(text_type="plain_text")
            documentsimg = loader.load_data(file_path)
            documents.extend(documentsimg)
        elif file_extension.lower() == '.pdf':
            process_pdf_file(file_path)

class Model:
    generator = None

    @staticmethod
    def setup():
        """model setup"""
        print("START LOADING SETUP ZEPHYR 7B", file=sys.stderr)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "HuggingFaceH4/zephyr-7b-beta"
        
        HUB_TOKEN = "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"
        huggingface_hub.login(token=HUB_TOKEN)

        llm = HuggingFaceInferenceAPI(
            model_name="HuggingFaceH4/zephyr-7b-beta", token=HUB_TOKEN
        )
        print("SETUP DONE", file=sys.stderr)

    @staticmethod
    def predict(prompt, args):
        """model setup"""
        return Model.generator(prompt, args) 
    
    @staticmethod
    def query(question):
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        # Set up ChromaDB client and collection
        chroma_host = "http://aricord.chromadb.e613f23.kyma.ondemand.com"
        chroma_port = 8000
        chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        # chroma_client = chromadb.PersistentClient(path='./sentence_index')
        print('HEARTBEAT:', chroma_client.heartbeat())
        chroma_collection_name = "multidoc" 
        chroma_collection = chroma_client.get_collection(name=chroma_collection_name, embedding_function=sentence_transformer_ef)

        HUB_TOKEN = "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"
        huggingface_hub.login(token=HUB_TOKEN)

        llm = HuggingFaceInferenceAPI(
            model_name="HuggingFaceH4/zephyr-7b-beta", token=HUB_TOKEN
        )

        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", max_length=512)

        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
        retrieved_sentence_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=ctx_sentence)

        sentence_query_engine = retrieved_sentence_index.as_query_engine(
            similarity_top_k=5,
            verbose=True,
            # the target key defaults to `window` to match the node_parser's default
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ],
        )

        import json
        try:
            sentence_response = sentence_query_engine.query(question)

            # Check if the result is empty
            if not sentence_response:
                result_message = {"success": False, "message": "No results found."}
            else:
                # Extract relevant information from sentence_response
                extracted_info = {"response": sentence_response.response}
                result_message = {"success": True, "results": extracted_info}

            # Print the JSON representation
            print(json.dumps(result_message))
            
        except Exception as e:
            error_message = {"success": False, "message": f"Error during query execution: {str(e)}"}
            print(json.dumps(error_message))
            traceback.print_exc()
            sys.exit(1)

    @staticmethod
    def DataIngestion():
        print("Data Ingestion Started")
        directory_path = "./data/"
        process_files_in_directory(directory_path)
        sentence_node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        nodes = sentence_node_parser.get_nodes_from_documents(documents)

        HUB_TOKEN = "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"
        huggingface_hub.login(token=HUB_TOKEN)

        llm = HuggingFaceInferenceAPI(
            model_name="HuggingFaceH4/zephyr-7b-beta", token=HUB_TOKEN
        )

        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
        chroma_host = "http://aricord.chromadb.e613f23.kyma.ondemand.com"
        chroma_port = 8000
        chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        chroma_collection_name = "multidoc"
        chroma_client.delete_collection(name=chroma_collection_name)
        chroma_collection = chroma_client.get_or_create_collection(name=chroma_collection_name, embedding_function=sentence_transformer_ef)

        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", max_length=512)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        ctx_sentence = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=sentence_node_parser)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        sentence_index = VectorStoreIndex(nodes, service_context=ctx_sentence, storage_context=storage_context)
        sentence_index.storage_context.persist()

if __name__ == "__main__":
    # for local testing
    Model.setup()
    print(Model.predict("Hello, who are you?", {}))
