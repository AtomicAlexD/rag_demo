import os
import re
import json
import requests
from typing import List, Dict
import sqlite3

# Simple embedding using basic text similarity
class SimpleRAG:
    def __init__(self):
        self.chunks = []
        self.setup_db()
    
    def setup_db(self):
        self.conn = sqlite3.connect("simple_rag.db")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                content TEXT,
                keywords TEXT
            )
        """)
    
    def simple_embed(self, text):
        # Simple keyword-based "embedding"
        words = re.findall(r"\w+", text.lower())
        return " ".join(set(words))
    
    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split by company
        companies = re.split(r"\n(?=[A-Z][a-zA-Z\s]+(?:Inc\.|Corporation|Platforms)?\n)", text)
        
        for i, company in enumerate(companies):
            if company.strip():
                keywords = self.simple_embed(company)
                self.conn.execute(
                    "INSERT OR REPLACE INTO chunks (id, content, keywords) VALUES (?, ?, ?)",
                    (i, company.strip(), keywords)
                )
        self.conn.commit()
        print(f"Loaded {len(companies)} companies")
    
    def search(self, query, n_results=3):
        query_words = set(re.findall(r"\w+", query.lower()))
        
        cursor = self.conn.execute("SELECT content, keywords FROM chunks")
        results = []
        
        for content, keywords in cursor.fetchall():
            keyword_words = set(keywords.split())
            score = len(query_words.intersection(keyword_words))
            if score > 0:
                results.append((content, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:n_results]]
    
    def chat(self, query):
        chunks = self.search(query)
        context = "\n\n".join(chunks)
        
        prompt = f"""Based on this information, answer the question:

{context}

Question: {query}
Answer:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False}
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code}"
        except:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running."

# Main execution
if __name__ == "__main__":
    rag = SimpleRAG()
    rag.load_data("tech_companies.txt")
    
    print("Simple RAG ready! Ask questions about tech companies.")
    print("Type 'quit' to exit.")
    
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["quit", "exit"]:
            break
        if query:
            response = rag.chat(query)
            print(f"Bot: {response}")