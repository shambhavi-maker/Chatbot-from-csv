
import os
from dotenv import load_dotenv  # Load environment variables from a .env file

import textwrap  # (Optional) For formatting text if needed later

# Import various libraries used in the script
import langchain
import chromadb
import transformers
import openai
import torch
import requests
import json
from typing import Optional, List

# Import specific classes and functions from libraries
from transformers import AutoTokenizer
#from langchain_community.llms import HuggingFacePipelin
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain_ollama.llms import OllamaLLM
# Import OpenAI + Gemini
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain.llms.base import LLM
# Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Custom Gemini Wrapper for LangChain
# =============================================================================
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import PrivateAttr


class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash"
    temperature: float = 0.2

    # Private attribute for the Gemini client (not part of pydantic fields)
    _client: any = PrivateAttr()

    def __init__(self, google_api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.2):
        super().__init__(model=model, temperature=temperature)
        genai.configure(api_key=google_api_key)
        self._client = genai.GenerativeModel(model)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            )
        )
        return response.text


# =============================================================================
# 1. Environment Setup and API Key Loading
# =============================================================================

# Load environment variables from the .env file (e.g., API keys)


# =============================================================================
# 6. Create the RetrievalQA Chain
# =============================================================================

# The RetrievalQA chain combines:
#   - The language model (model) to generate responses.
#   - A retriever (db.as_retriever) that fetches relevant document chunks based on the query.
#   - A prompt that provides instructions on how to answer the query.
def run(inpu,provider="gemini"):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    load_dotenv()
    # Retrieve tokens from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # =============================================================================
    # 2. Initialize the Language Model with Ollama
    # =============================================================================

    # Here, we use a locally pulled model called "deepseek-r1" through OllamaLLM.
    # This model will be used later in our RetrievalQA chain.
    if provider == "openai":
        model = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini",  # or gpt-4o, gpt-3.5-turbo, etc.
            temperature=0.2
        )
    elif provider == "gemini":
        model = GeminiLLM(
            google_api_key=google_api_key,
            model="gemini-1.5-flash",
            temperature=0.2
        )
    else:
        raise ValueError("Provider must be 'openai' or 'gemini'")

    # =============================================================================
    # 3. Document Preprocessing Function
    # =============================================================================

    def docs_preprocessing_helper(file):
        """
        Helper function to load and preprocess a CSV file containing data.
        
        This function performs two main tasks:
        1. Loads the CSV file using CSVLoader from LangChain.
        2. Splits the loaded documents into smaller text chunks using CharacterTextSplitter.
        
        Args:
            file (str): Path to the CSV file.
            
        Returns:
            list: A list of document chunks ready for embedding and indexing.
        
        Raises:
            TypeError: If the output is not in the expected dataframe/document format.
        """
        # Load the CSV file using LangChain's CSVLoader.
        loader = CSVLoader(file)
        docs = loader.load()
        
        # Create a text splitter that divides the documents into chunks of up to 1000 characters
        # with no overlapping text between chunks.
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(docs)
        
        return docs

    # Preprocess the CSV file "survey.csv" and store the document chunks in 'docs'.
    docs = docs_preprocessing_helper('survey.csv')

    # =============================================================================
    # 4. Set Up the Embedding Function and Chroma Database
    # =============================================================================

    # Initialize the embedding function from OpenAI. This converts text into numerical vectors.
    # The OpenAIEmbeddings class uses the openai_api_key for authentication.
    def get_embedding_function(provider: str, openai_api_key: str):
        """
        Return embedding function based on provider.
        """
        if provider == "openai":
            return OpenAIEmbeddings(openai_api_key=openai_api_key)
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError("Embedding provider must be 'openai' or 'huggingface'")

    embedding_function = get_embedding_function("openai", openai_api_key)

    # Create a vector store (ChromaDB) from the document chunks using the embedding function.
    # The Chroma database will be used to retrieve the most relevant documents based on the query.
    db = Chroma.from_documents(docs, embedding_function)

    # =============================================================================
    # 5. Define and Initialize the Prompt Template
    # =============================================================================

    # Define a prompt template that instructs the chatbot on how to answer customer queries.
    # The template includes context information and instructs the bot to use only provided data.
    template = """You are a medical consultant chatbot.

    Answer the customer's/patient  questions. When relevant questions come, use the provided documents. Please answer to their specific question. If you are unsure, say "I don't know, please call our customer support". Use engaging, courteous, and professional language similar to a customer representative.
    Keep your answers concise.

    {context}

    """

    # Create a PromptTemplate object from LangChain with the defined template.
    # It expects a variable called "context" that can be filled later.
    prompt = PromptTemplate(template=template, input_variables=["context"])

    # Format the prompt with a general context message.
    # This additional context tells the chatbot the scenario in which it will be answering questions.
    formatted_prompt = prompt.format(
        context="A customer is on the clothing company website and wants to chat with the website chatbot. They will ask you a question. Please answer to their specific question"
    )
    chain_type_kwargs = {"prompt": prompt}  # Pass our custom prompt template to the chain.
    chain = RetrievalQA.from_chain_type(
        llm=model,  # The language model (OllamaLLM with deepseek-r1)
        chain_type="stuff",  # The "stuff" chain type combines all retrieved documents into one context.
        retriever=db.as_retriever(search_kwargs={"k": 1}),  # Retrieve the top relevant document chunk.
        chain_type_kwargs=chain_type_kwargs,
    )
    response = chain.run(input)
    return response


if __name__ == "__main__":
    print(run("What are the early symptoms of diabetes? ","openai"))


