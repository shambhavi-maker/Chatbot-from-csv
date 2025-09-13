# CSV-Powered Medical Consultant Chatbot

## Intelligent Q\&A from your Structured Data

-----

## Introduction

The **CSV-Powered Medical Consultant Chatbot** is an innovative, Streamlit-based application designed to provide intelligent, context-aware answers to user queries by leveraging information stored in a Comma Separated Values (CSV) file. This project addresses the challenge of extracting specific information from large datasets and presenting it in a conversational format, making data accessible and interactive. It's particularly useful for domains requiring quick information retrieval from structured data, such as medical FAQs, product catalogs, or customer support knowledge bases, empowering users with instant, reliable answers sourced directly from your data. Configured as a professional "Medical Consultant Chatbot," it offers concise, empathetic, and accurate responses based on your provided data.

-----

## Features

  * **Conversational Interface:** User-friendly chat interface built with Streamlit for a seamless experience.
  * **Retrieval-Augmented Generation (RAG):** Intelligently retrieves relevant information from CSV data to inform LLM responses, ensuring answers are grounded in your specific knowledge base.
  * **Flexible LLM Support:** Seamless integration with both **OpenAI (GPT models like `gpt-4o-mini`)** and **Google Gemini (models like `gemini-1.5-flash`)**, allowing users to choose their preferred large language model.
  * **Customizable Data Source:** Easily adaptable to any CSV file for diverse use cases; currently configured for medical inquiries via `survey.csv`.
  * **Dynamic Embeddings:** Supports **OpenAI Embeddings** and provides an option for **HuggingFace Embeddings** for efficient text vectorization and semantic search.
  * **ChromaDB Integration:** Utilizes ChromaDB as a robust vector store for fast and accurate document similarity search.
  * **Custom Persona:** Configured as a professional "Medical Consultant Chatbot" to deliver engaging, courteous, and precise answers.
  * **Environment Variable Management:** Securely handles API keys and sensitive information via `.env` files.

-----

## Installation

### Prerequisites

  * Python 3.8+
  * An active OpenAI API Key or Google API Key for chosen LLM and embeddings.
  * Git (for cloning the repository)

### Steps

1.  **Clone the repository:**
    ```
    git clone https://github.com/shambhavi-maker/Chatbot-from-csv.git
    cd Chatbot-from-csv
    ```
2.  **Create a virtual environment (recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
4.  **Prepare your data:** Ensure you have a CSV file named `survey.csv` in the project root directory. This file will be used as the knowledge base for the chatbot. If your CSV has a different name, you will need to update the `docs_preprocessing_helper` call in `model.py`.
5.  **Configure Environment Variables:** Create a `.env` file in the project root directory and add your API keys:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    You only need to provide the API key for the LLM provider you intend to use.

-----

## Usage

Once installed and configured, you can run the Streamlit application to interact with the chatbot.

### Running the Chatbot

```
streamlit run frontend.py
```

This command will open the chatbot interface in your web browser. You can then type your questions into the chat input field.

### Example Interaction

*(Assuming `survey.csv` contains medical information relevant to diabetes, heart disease, and flu, as implied by the chatbot's persona)*
**User:** `What are the early symptoms of diabetes?`
**Chatbot:** `Early symptoms of diabetes may include frequent urination, increased thirst, and unexplained weight loss. It is advisable to consult a healthcare professional for a proper diagnosis.`

**User:** `How can I prevent heart disease?`
**Chatbot:** `To prevent heart disease, maintaining a balanced diet, engaging in regular physical activity, and avoiding smoking are crucial. Regular check-ups with your doctor are also highly recommended.`

**User:** `Tell me about flu.`
**Chatbot:** `The flu is a contagious respiratory illness caused by influenza viruses. Symptoms can include fever, cough, sore throat, and body aches. Vaccination is an effective way to prevent the flu.`

The chatbot will retrieve relevant information from your `survey.csv` data and use the chosen LLM to formulate a concise and professional response.

-----

## Programmatic Usage (`model.py`)

While the primary interface is a Streamlit application, the core logic for the chatbot is encapsulated in the `run` function within `model.py`. This allows for programmatic interaction if you wish to integrate the chatbot's intelligence into other applications or scripts.

### `run(input, provider="gemini")`

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `input` | `str` | The user's query or question. | |
| `provider`| `str` | The LLM provider to use. Accepts `"openai"` or `"gemini"`. | `"gemini"` |

### Example Programmatic Call

```
from model import run

# Using OpenAI LLM
response_openai = run("What are the early symptoms of diabetes?", "openai")
print(f"OpenAI Chatbot: {response_openai}")

# Using Gemini LLM (requires GOOGLE_API_KEY configured)
response_gemini = run("How can I prevent heart disease?", "gemini")
print(f"Gemini Chatbot: {response_gemini}")
```

-----

## Technologies Used

  * **Python** (Core language)
  * **Streamlit** & **Streamlit-Chat** (Frontend UI)
  * **LangChain** (Framework for LLM applications)
  * **OpenAI** & **Google Generative AI** (Large Language Models: GPT-4o-mini, Gemini 1.5 Flash)
  * **ChromaDB** (Vector Store)
  * **OpenAI Embeddings** & **HuggingFace Embeddings** (Text Embeddings)
  * **python-dotenv** (Environment variable management)
  * **transformers** (Underlying for some LangChain components)
  * **tiktoken** (OpenAI tokenizer)
  * **accelerate** (HuggingFace utility)

-----

## Project Structure

```
.
├── .env                     # Stores environment variables (API keys)
├── frontend.py              # Streamlit web application interface for chat interaction
├── model.py                 # Core chatbot logic (LLM, RAG, embeddings, data loading)
├── requirements.txt         # Python dependencies required for the project
└── survey.csv               # Example/placeholder data source (CSV knowledge base) for the chatbot
```

The `frontend.py` file orchestrates the user interface and interaction, while `model.py` encapsulates all the complex logic for data processing, LLM integration, and response generation. The `survey.csv` file serves as the foundational knowledge base that the chatbot queries.

-----

## Configuration

The project relies on environment variables for sensitive information like API keys. Ensure you have a `.env` file in the root directory of the project with the following:

```
OPENAI_API_KEY="your_openai_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

Replace `"your_openai_api_key_here"` and `"your_google_api_key_here"` with your actual API keys. You only need to provide the key for the LLM provider you intend to use (e.g., if you only use Gemini, `OPENAI_API_KEY` can be omitted or left blank). If both are provided, you can specify which one to use in the `run` function call.

The default LLM provider in the `run` function within `model.py` is set to `"gemini"`. You can change this default or explicitly specify `"openai"` when calling the function.

-----

## Documentation

This README serves as the primary documentation for the CSV-Powered Medical Consultant Chatbot, covering setup, usage, and an overview of the system architecture.

  * **README.md:** Comprehensive guide for project setup, installation, and usage.
  * **Code Comments:** Detailed comments within `model.py` and `frontend.py` explain individual functions, classes, and logic flow, offering insights into the codebase.

For more in-depth understanding of the underlying technologies, please refer to the official documentation for LangChain, Streamlit, OpenAI, Google Generative AI, and ChromaDB.

-----

## Contributing

We welcome contributions to enhance the CSV-Powered Medical Consultant Chatbot\! If you'd like to contribute, please follow these steps:

1.  **Fork the repository:** Click the "Fork" button at the top right of this GitHub page.
2.  **Clone your forked repository:**
    ```
    git clone https://github.com/YOUR_USERNAME/Chatbot-from-csv.git
    ```
3.  **Create a new branch:**
    ```
    git checkout -b feature/your-feature-name
    ```
4.  **Make your changes:** Implement your new features, bug fixes, or improvements.
5.  **Commit your changes:** Write clear and descriptive commit messages that explain the purpose of your changes.
6.  **Push to your branch:**
    ```
    git push origin feature/your-feature-name
    ```
7.  **Open a Pull Request:** Submit a pull request to the `main` branch of the original repository, providing a detailed description of your changes and why they are beneficial.

Please ensure your code adheres to good practices, includes necessary comments, and passes any existing tests. We appreciate your efforts\!

-----

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute the code, provided the original license and copyright notice are included.

See the [LICENSE](https://github.com/shambhavi-maker/Chatbot-from-csv/blob/main/LICENSE) file for full details (*Note: A `LICENSE` file containing the MIT License text is assumed to be present in the repository root.*).

-----

## Acknowledgments

  * Thanks to the creators of **LangChain** for providing a robust and flexible framework for building advanced LLM applications.
  * Appreciation to **Streamlit** for making it incredibly easy to build beautiful and interactive web applications with Python.
  * Gratitude to **OpenAI** and **Google Generative AI** for their cutting-edge and powerful large language models that drive the intelligence of this chatbot.
  * Credit to **ChromaDB** for its efficient and developer-friendly vector store capabilities.
  * Inspired by various open-source Retrieval-Augmented Generation (RAG) implementations and educational resources.
