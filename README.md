# Simple RAG Chat System

A lightweight Retrieval-Augmented Generation (RAG) system that lets you chat with your own data using local AI models. This implementation uses simple keyword matching for retrieval and connects to Ollama for text generation.

## What This Does

- **Load your data**: Reads text files and makes them searchable
- **Find relevant information**: Uses keyword matching to find the most relevant content for your questions
- **Generate answers**: Connects to a local AI model (via Ollama) to generate natural language responses based on the retrieved information

## Prerequisites

### 1. Python

Make sure you have Python 3.7+ installed:

- **Windows**: Download from [python.org](https://python.org) (check "Add Python to PATH" during installation)
- **Mac**: `brew install python3` or download from python.org
- **Linux**: `sudo apt install python3 python3-pip`

### 2. Ollama

Install Ollama for local AI model serving:

1. Go to [ollama.ai](https://ollama.ai)
2. Download and install for your operating system
3. Pull a model (we recommend llama3.2):

   ```bash
   ollama pull llama3.2
   ```

## Quick Start

### 1. Clone this repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Install dependencies

```bash
# Install UV if you don't have it yet
curl -sSf https://install.ultraviolet.rs | sh
# Or on Windows PowerShell
iwr https://install.ultraviolet.rs | iex

# Install dependencies with UV
uv pip install -r requirements.txt
```

*Note: This project only requires the `requests` library - everything else uses Python's built-in modules*

### 3. Run the system

```bash
uv run python src/run_rag.py --web
```

### 4. Start chatting

Once running, you can ask questions like:

- "Who is the CEO of Apple?"
- "What does NVIDIA make?"
- "Which companies are worth over 1 trillion dollars?"

Type `quit` to exit.

## How It Works

### Architecture

```
Your Question → Keyword Search → Find Relevant Data → Send to Ollama → Get Answer
```

### Components

- **Data Storage**: SQLite database stores text chunks and keywords
- **Retrieval**: Simple keyword matching finds relevant information
- **Generation**: Ollama processes retrieved context and generates responses

### Example Flow

1. You ask: "Who runs Microsoft?"
2. System finds chunks containing "microsoft" and "ceo"  
3. Sends Microsoft's information + your question to Ollama
4. Ollama responds: "Satya Nadella is the CEO of Microsoft"

## Customizing Your Data

### Adding Your Own Data

1. Replace `tech_companies.txt` with your own text file
2. Format your data with clear sections (the system will auto-detect natural breaks)
3. Run the system - it will automatically re-index your new data

### Data Format Tips

- **Use clear headings**: Company names, product names, etc.
- **Include keywords**: Make sure important terms appear in your text
- **Natural sections**: The system splits text into logical chunks automatically

## Configuration

### Changing the AI Model

Edit the model name in `run_rag.py`:

```python
# In the chat() method, change:
json={"model": "llama3.2", "prompt": prompt, "stream": False}
# To your preferred model:
json={"model": "llama2", "prompt": prompt, "stream": False}
```

### Adjusting Search Results

Change the number of relevant chunks returned:

```python
# In the search() method, modify n_results:
def search(self, query, n_results=3):  # Change 3 to your preferred number
```

## Troubleshooting

### "Cannot connect to Ollama"

- Make sure Ollama is running: `ollama serve`
- Verify the model is installed: `ollama list`
- Check if the model name matches what you're using in the code

### "No such file or directory" (tech_companies.txt)

- Make sure your data file is in the same directory as `simple_rag.py`
- Check the filename matches exactly (case-sensitive)

### Python not found

- On Windows: Make sure Python is added to PATH during installation
- Try `python3` instead of `python`
- Verify installation: `python --version`

### Poor search results

- Make sure your question contains keywords that appear in your data
- Try rephrasing your question with different terms
- Check that your data file contains the information you're asking about

## Features

### What's Included

- Local data processing (no external APIs needed for search)
- SQLite database for fast keyword lookup
- Integration with any Ollama model
- Simple, readable codebase
- No complex dependencies

## Technical Details

### Dependencies

- **Built-in**: `os`, `re`, `json`, `sqlite3` (no installation needed)
- **External**: `requests` (for Ollama communication)

### Database Schema

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    content TEXT,        -- The actual text content
    keywords TEXT        -- Searchable keywords extracted from content
);
```

### Performance

- **Indexing**: Fast keyword extraction and SQLite storage
- **Search**: O(n) keyword matching across all chunks
- **Memory**: Minimal - only active chunks loaded into memory
- **Storage**: SQLite database file created in project directory

## Contributing

Feel free to submit issues and enhancement requests! This is a learning project, so improvements and educational additions are especially welcome.

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Happy chatting with your data!** 🤖💬
