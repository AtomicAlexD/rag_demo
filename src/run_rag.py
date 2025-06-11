#!/usr/bin/env python3
"""
Flask web interface for Simple RAG system.
Combines the RAG functionality with a web frontend.
"""

import os
import re
import sqlite3
import requests
from typing import List, Dict, Optional, Tuple
from flask import Flask, render_template, request, jsonify


class DatabaseManager:
    """Manages SQLite database operations for storing text chunks and keywords."""
    
    def __init__(self, db_path: str = "simple_rag.db"):
        self.db_path = db_path
        self.conn = None
        self.setup_database()
    
    def setup_database(self):
        """Create database connection and initialize tables."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_keywords 
                ON chunks(keywords)
            """)
            self.conn.commit()
            print(f"Database initialized: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            raise
    
    def store_chunks(self, chunks_with_keywords: List[Tuple[str, str]]) -> int:
        """Store text chunks with their keywords in the database."""
        if not self.conn:
            raise RuntimeError("Database not initialized")
        
        try:
            # Clear existing data
            self.conn.execute("DELETE FROM chunks")
            
            # Insert new chunks
            for i, (content, keywords) in enumerate(chunks_with_keywords):
                self.conn.execute(
                    "INSERT INTO chunks (id, content, keywords) VALUES (?, ?, ?)",
                    (i, content.strip(), keywords)
                )
            
            self.conn.commit()
            return len(chunks_with_keywords)
        
        except sqlite3.Error as e:
            print(f"Error storing chunks: {e}")
            self.conn.rollback()
            raise
    
    def search_chunks(self, query_keywords: str, n_results: int = 3) -> List[Tuple[str, int]]:
        """Search for chunks matching query keywords."""
        if not self.conn:
            raise RuntimeError("Database not initialized")
        
        try:
            cursor = self.conn.execute("SELECT content, keywords FROM chunks")
            results = []
            query_words = set(query_keywords.lower().split())
            
            for content, keywords in cursor.fetchall():
                keyword_words = set(keywords.split())
                score = len(query_words.intersection(keyword_words))
                
                if score > 0:
                    results.append((content, score))
            
            # Sort by score (descending) and return top n_results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:n_results]
        
        except sqlite3.Error as e:
            print(f"Error searching chunks: {e}")
            return []
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in database."""
        if not self.conn:
            return 0
        
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
        except sqlite3.Error:
            return 0
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


class SearchEngine:
    """Handles text processing, chunking, and keyword-based search operations."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_keywords(self, text: str) -> str:
        """Extract keywords from text for search indexing."""
        # Find all words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very short words and common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
            'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # Filter out stop words and short words
        meaningful_words = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        # Return unique words as space-separated string
        return ' '.join(set(meaningful_words))
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for processing."""
        # First, try to split by natural boundaries (companies in this case)
        companies = re.split(
            r'\n(?=[A-Z][a-zA-Z\s&]+(?:Inc\.|Corporation|Platforms|LLC)?\.?\s*\n)', 
            text
        )
        companies = [company.strip() for company in companies if company.strip()]
        
        chunks = []
        
        for company in companies:
            # If company info is too long, split it further
            if len(company) > self.chunk_size:
                chunks.extend(self._split_long_text(company))
            else:
                chunks.append(company)
        
        return chunks
    
    def _split_long_text(self, text: str) -> List[str]:
        """Split long text into smaller chunks with overlap."""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Calculate chunk end position
            chunk_words = []
            current_length = 0
            
            # Add words until we reach chunk_size
            while i < len(words) and current_length < self.chunk_size:
                word = words[i]
                chunk_words.append(word)
                current_length += len(word) + 1  # +1 for space
                i += 1
            
            # Create chunk
            if chunk_words:
                chunks.append(' '.join(chunk_words))
            
            # Move back for overlap (if not at end)
            if i < len(words):
                overlap_words = min(self.overlap // 10, len(chunk_words) // 2)
                i -= overlap_words
        
        return chunks
    
    def load_and_process_file(self, file_path: str) -> List[Tuple[str, str]]:
        """Load text file and process it into chunks with keywords."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")
        
        # Process text into chunks
        chunks = self.chunk_text(text)
        
        # Generate keywords for each chunk
        processed_chunks = []
        for chunk in chunks:
            keywords = self.extract_keywords(chunk)
            processed_chunks.append((chunk, keywords))
        
        return processed_chunks
    
    def search_query_keywords(self, query: str) -> str:
        """Extract keywords from a search query."""
        return self.extract_keywords(query)


class OllamaClient:
    """Client for communicating with local Ollama server."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.generate_url = f"{self.base_url}/api/generate"
    
    def check_connection(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_rag_response(self, query: str, context_chunks: List[str], 
                            model: Optional[str] = None) -> Dict:
        """Generate RAG response given query and context."""
        model_name = model or self.model
        
        # Create prompt
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."
        prompt = f"""You are a helpful assistant. Use the provided context to answer the user's question. If the context doesn't contain enough information to answer the question, say so clearly. Be concise and accurate in your response.

Context:
{context}

Question: {query}

Answer:"""
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": data.get("model", model_name),
                    "tokens": data.get("eval_count", 0),
                    "generation_time": data.get("eval_duration", 0) / 1e9,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "response": "",
                "error": "Request timed out - the model might be too slow or unavailable"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "response": "",
                "error": "Cannot connect to Ollama server. Make sure it's running on localhost:11434"
            }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "error": f"Unexpected error: {str(e)}"
            }


class SimpleRAG:
    """Simple RAG system that combines keyword search with local LLM generation."""
    
    def __init__(self, db_path: str = "simple_rag.db", model: str = "llama3.2"):
        print("Initializing Simple RAG system...")
        
        # Initialize components
        self.database = DatabaseManager(db_path)
        self.search_engine = SearchEngine()
        self.llm_client = OllamaClient(model=model)
        
        print("Simple RAG system initialized successfully!")
    
    def load_data(self, file_path: str) -> Dict:
        """Load and index data from a text file."""
        print(f"Loading data from: {file_path}")
        
        try:
            # Process file into chunks with keywords
            chunks_with_keywords = self.search_engine.load_and_process_file(file_path)
            
            # Store in database
            chunks_stored = self.database.store_chunks(chunks_with_keywords)
            
            result = {
                "success": True,
                "file_path": file_path,
                "chunks_stored": chunks_stored,
                "total_chunks": len(chunks_with_keywords)
            }
            
            print(f"Successfully loaded {chunks_stored} chunks from {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            print(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "file_path": file_path
            }
    
    def search(self, query: str, n_results: int = 3) -> List[str]:
        """Search for relevant text chunks based on query."""
        # Extract keywords from query
        query_keywords = self.search_engine.search_query_keywords(query)
        
        # Search database
        results = self.database.search_chunks(query_keywords, n_results)
        
        # Return just the content (not scores)
        return [content for content, score in results]
    
    def chat(self, query: str, n_results: int = 3, model: Optional[str] = None) -> Dict:
        """Complete RAG pipeline: search + generate response."""
        print(f"Processing query: {query}")
        
        # Search for relevant chunks
        relevant_chunks = self.search(query, n_results)
        
        if not relevant_chunks:
            return {
                "success": False,
                "query": query,
                "response": "I couldn't find any relevant information to answer your question.",
                "chunks_found": 0
            }
        
        print(f"Using {len(relevant_chunks)} relevant chunks for context")
        
        # Generate response
        llm_result = self.llm_client.generate_rag_response(query, relevant_chunks, model)
        
        # Prepare response
        response = {
            "success": llm_result["success"],
            "query": query,
            "response": llm_result["response"],
            "chunks_found": len(relevant_chunks),
            "generation_time": llm_result.get("generation_time", 0),
            "model_used": llm_result.get("model", "unknown")
        }
        
        if not llm_result["success"]:
            response["error"] = llm_result["error"]
        
        return response
    
    def get_status(self) -> Dict:
        """Get system status information."""
        return {
            "database_chunks": self.database.get_chunk_count(),
            "ollama_connected": self.llm_client.check_connection(),
            "model": self.llm_client.model
        }
    
    def close(self):
        """Clean up resources."""
        if self.database:
            self.database.close()


# Global RAG instance
rag = None

# Flask app
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.secret_key = 'your-secret-key-change-this-in-production'

@app.route('/')
def index():
    """Main chat interface."""
    status = rag.get_status() if rag else {"database_chunks": 0, "ollama_connected": False, "model": "none"}
    return render_template('index.html', status=status)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat requests."""
    if not rag:
        return jsonify({"success": False, "error": "RAG system not initialized"})
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"success": False, "error": "No query provided"})
    
    query = data['query'].strip()
    if not query:
        return jsonify({"success": False, "error": "Empty query"})
    
    try:
        result = rag.chat(query)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    if not rag:
        return jsonify({"success": False, "error": "RAG system not initialized"})
    
    try:
        status = rag.get_status()
        return jsonify({"success": True, **status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def create_templates():
    """Create HTML template if it doesn't exist."""
    # Create templates directory relative to where the Flask app is running
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    template_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(template_path):
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple RAG Chat</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .status {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .status.connected { border-left: 4px solid #28a745; }
        .status.disconnected { border-left: 4px solid #dc3545; }
        .chat-container {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            background: #fafafa;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 18px;
            max-width: 80%;
        }
        .user-message {
            background: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: white;
            border: 1px solid #ddd;
        }
        .message-meta {
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        #queryInput {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
        }
        #sendButton {
            padding: 12px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
        }
        #sendButton:hover {
            background: #0056b3;
        }
        #sendButton:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Simple RAG Chat</h1>
            <p>Ask questions about your data</p>
        </div>
        
        <div class="status {{ 'connected' if status.ollama_connected else 'disconnected' }}">
            <strong>System Status:</strong>
            Database: {{ status.database_chunks }} chunks loaded |
            Ollama: {{ 'Connected' if status.ollama_connected else 'Disconnected' }} |
            Model: {{ status.model }}
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                <div>👋 Hello! I'm ready to answer questions about your data. What would you like to know?</div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div>🤔 Thinking...</div>
        </div>
        
        <div class="input-container">
            <input type="text" id="queryInput" placeholder="Ask a question..." maxlength="500">
            <button id="sendButton">Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const queryInput = document.getElementById('queryInput');
        const sendButton = document.getElementById('sendButton');
        const loading = document.getElementById('loading');

        function addMessage(content, isUser = false, meta = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.textContent = content;
            messageDiv.appendChild(contentDiv);
            
            if (meta) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                metaDiv.textContent = meta;
                messageDiv.appendChild(metaDiv);
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = `Error: ${message}`;
            chatContainer.appendChild(errorDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const query = queryInput.value.trim();
            if (!query) return;

            // Add user message
            addMessage(query, true);
            queryInput.value = '';
            
            // Show loading
            sendButton.disabled = true;
            loading.style.display = 'block';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });

                const data = await response.json();
                
                if (data.success) {
                    const meta = `${data.chunks_found} chunks • ${data.generation_time?.toFixed(1)}s • ${data.model_used}`;
                    addMessage(data.response, false, meta);
                } else {
                    showError(data.error || 'Unknown error occurred');
                }
            } catch (error) {
                showError('Failed to connect to server');
            } finally {
                sendButton.disabled = false;
                loading.style.display = 'none';
            }
        }

        sendButton.addEventListener('click', sendMessage);
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Focus input on load
        queryInput.focus();
    </script>
</body>
</html>'''
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Created template: {template_path}")
        return template_path
    
    return template_path
def initialize_rag():
    """Initialize the RAG system with data."""
    global rag
    
    # Look for data file
    possible_paths = [
        "../data/tech_companies.txt",
        "data/tech_companies.txt", 
        "tech_companies.txt",
        "../tech_companies.txt"
    ]
    
    data_file = None
    for path in possible_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if not data_file:
        print("Warning: No data file found. The system will start but won't have any data to search.")
        print("Expected locations: data/tech_companies.txt")
        rag = SimpleRAG()
        return False
    
    # Initialize RAG and load data
    rag = SimpleRAG()
    load_result = rag.load_data(data_file)
    
    if load_result["success"]:
        print(f"✅ Loaded {load_result['chunks_stored']} chunks from {data_file}")
        return True
    else:
        print(f"❌ Failed to load data: {load_result['error']}")
        return False

def main():
    """Main function - can run as CLI or web app."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        # Run web interface
        print("🌐 Starting Flask web interface...")
        create_templates()
        initialize_rag()
        
        print("\n" + "="*50)
        print("🎯 Simple RAG Web Interface")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("="*50 + "\n")
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=True)
        except KeyboardInterrupt:
            print("\n👋 Server stopped!")
        finally:
            if rag:
                rag.close()
    
    else:
        # Run CLI interface (original functionality)
        initialize_rag()
        
        if not rag.llm_client.check_connection():
            print("⚠️  Warning: Cannot connect to Ollama server.")
            print("Make sure Ollama is running: ollama serve")
            print("And that you have a model installed: ollama pull llama3.2")
            print()
        
        # Display system info
        status = rag.get_status()
        print(f"📊 System ready with {status['database_chunks']} chunks loaded")
        print(f"🤖 Using model: {status['model']}")
        print()
        
        # Interactive chat loop
        print("=" * 60)
        print("🎯 Simple RAG Chat System")
        print("Ask questions about your data. Type 'quit', 'exit', or 'q' to stop.")
        print("Type '--web' to start the web interface instead.")
        print("=" * 60)
        print()
        
        try:
            while True:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                try:
                    result = rag.chat(query)
                    
                    if result["success"]:
                        print(f"\n🤖 Bot: {result['response']}")
                        print(f"\n💡 Used {result['chunks_found']} chunks in {result['generation_time']:.1f}s")
                    else:
                        print(f"\n❌ Error: {result.get('error', 'Unknown error occurred')}")
                    
                    print("-" * 60 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Interrupted by user. Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Unexpected error: {e}")
                    print()
        
        finally:
            rag.close()


if __name__ == "__main__":
    main()