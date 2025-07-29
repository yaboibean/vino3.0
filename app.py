from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import faiss
import numpy as np
import pickle
import logging
import os
from dotenv import load_dotenv
from rapidfuzz import fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
load_dotenv()

client = OpenAI()

# Load index + chunk text with error handling
try:
    if not os.path.exists("magazine_index.faiss"):
        logger.error("❌ magazine_index.faiss not found! Run extract_and_index.py first.")
        raise FileNotFoundError("Index file not found")
    
    if not os.path.exists("magazine_chunks.pkl"):
        logger.error("❌ magazine_chunks.pkl not found! Run extract_and_index.py first.")
        raise FileNotFoundError("Chunks file not found")
    
    logger.info("📂 Loading FAISS index...\n✨ Tapping into nearly two decades of curated wine magazine wisdom.\n🍷 Unlocking expert insights, rare stories, and timeless wine knowledge from Sommelier India archives!")
    index = faiss.read_index("magazine_index.faiss")
    logger.info(f"✅ Index loaded with {index.ntotal} vectors")
    
    logger.info("📂 Loading text chunks...")
    with open("magazine_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"✅ Loaded {len(chunks)} text chunks")
    
    if len(chunks) == 0:
        logger.error("❌ No chunks found in pickle file!")
        raise ValueError("Empty chunks file")
    
    # Log first few chunks for debugging
    logger.info("📋 First few chunks preview:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:100].replace('\n', ' ')
        logger.info(f"   Chunk {i}: {preview}...")

except Exception as e:
    logger.error(f"❌ Failed to load index/chunks: {str(e)}")
    index = None
    chunks = []

def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype="float32")

def calculate_recency_bias_fast(chunk):
    """Fast recency bias calculation - optimized for speed"""
    try:
        # Quick regex search for recent years
        import re
        
        # Look for years 2020+ (most relevant)
        recent_years = re.findall(r'\b(202[0-9])\b', chunk)
        if recent_years:
            latest_year = max(int(year) for year in recent_years)
            if latest_year >= 2024:
                return 1.0
            elif latest_year >= 2022:
                return 0.8
            else:
                return 0.6
        
        # Quick fallback - look for any year pattern
        if '202' in chunk:  # Any 2020s mention
            return 0.7
        elif '201' in chunk:  # Any 2010s mention
            return 0.4
        else:
            return 0.3  # Default
            
    except Exception:
        return 0.3  # Default on any error

def calculate_recency_bias(chunk):
    """Calculate recency bias score - higher score for more recent content"""
    try:
        # Extract year from chunk text (look for patterns like "2023", "2024", "2025")
        import re

        # Look for 4-digit years in the chunk
        years = re.findall(r'\b(20[1-2][0-9])\b', chunk)
        
        # Also look for "Issue" numbers and patterns
        issue_patterns = [
            r'Issue\s+(\d+),?\s+20([1-2][0-9])',  # "Issue 1, 2023"
            r'SI Issue\s+(\d+),?\s+20([1-2][0-9])',  # "SI Issue 2, 2024"
            r'20([1-2][0-9])',  # Just year
        ]
        
        latest_year = 2018  # Default fallback year
        issue_number = 1     # Default issue number
        
        # Find the most recent year mentioned
        if years:
            latest_year = max(int(year) for year in years)
        
        # Look for issue numbers
        for pattern in issue_patterns:
            matches = re.findall(pattern, chunk)
            if matches:
                if len(matches[0]) == 2:  # Issue number and year
                    issue_number = int(matches[0][0])
                    year = int('20' + matches[0][1])
                    if year > latest_year:
                        latest_year = year
                break
        
        # Calculate recency score (0-1, higher for more recent)
        current_year = 2025
        year_diff = current_year - latest_year
        
        # Recent content gets higher scores
        if year_diff <= 1:  # 2024-2025
            year_score = 1.0
        elif year_diff <= 2:  # 2023
            year_score = 0.8
        elif year_diff <= 3:  # 2022
            year_score = 0.6
        elif year_diff <= 5:  # 2020-2021
            year_score = 0.4
        else:  # Older than 2020
            year_score = 0.2
        
        # Boost for higher issue numbers (later in year)
        issue_boost = min(issue_number * 0.1, 0.3)
        
        final_score = year_score + issue_boost
        
        logger.debug(f"Recency bias for year {latest_year}, issue {issue_number}: {final_score}")
        return final_score
        
    except Exception as e:
        logger.debug(f"Error calculating recency bias: {str(e)}")
        return 0.3  # Default moderate score

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Check if system is properly initialized
        if index is None or not chunks:
            logger.error("❌ System not properly initialized")
            return jsonify({"error": "System not initialized. Please run extract_and_index.py first."}), 500

        data = request.get_json()
        query = data.get("question", "")
        if not query:
            return jsonify({"error": "No question provided"}), 400

        logger.info(f"❓ Received question: {query}")

        query_embedding = embed_query(query)
        D, I = index.search(np.array([query_embedding]), k=3)
        # Vector search results
        vector_chunks = set()
        for idx in I[0]:
            if idx < len(chunks):
                vector_chunks.add(idx)

        # Keyword and fuzzy search results (optimized)
        keyword_chunks = set()
        query_lower = query.lower()
        # Only check the first 500 chunks for fuzzy match to speed up
        for i, chunk in enumerate(chunks[:500]):
            chunk_lower = chunk.lower()
            # Exact match
            if query_lower in chunk_lower:
                keyword_chunks.add(i)
            # Fuzzy match (lower threshold for speed, e.g. 70)
            elif fuzz.partial_ratio(query_lower, chunk_lower) >= 70:
                keyword_chunks.add(i)

        # Combine results (union)
        combined_idxs = list(vector_chunks | keyword_chunks)
        # If no results, fallback to pure fuzzy keyword search
        if not combined_idxs:
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                if query_lower in chunk_lower or fuzz.partial_ratio(query_lower, chunk_lower) >= 80:
                    combined_idxs.append(i)
        # Fix type mismatch: cast FAISS indices to int
        I0_ints = [int(i) for i in I[0]]
        combined_idxs = sorted(combined_idxs, key=lambda x: I0_ints.index(x) if x in I0_ints else len(D[0]))

        # Get up to 5 relevant chunks (full text, not truncated)
        relevant_chunks = []
        for idx in combined_idxs[:5]:
            chunk = chunks[idx]
            relevant_chunks.append(chunk)

        relevant = "\n\n".join([chunk[:800] + "..." if len(chunk) > 800 else chunk for chunk in relevant_chunks])
        logger.info(f"📝 Using {len(relevant_chunks)} relevant chunks, total length: {len(relevant)} chars")
        if relevant:
            preview = relevant[:200].replace('\n', ' ')
            logger.info(f"📋 Relevant content preview: {preview}...")
        else:
            logger.warning("⚠️  No relevant content found!")

        prompt = f"""You are a helpful wine expert assistant answering questions based on wine magazine content.

Here is relevant context from the wine magazines:
{relevant}

Question: {query}

Instructions:
- Keep responses concise but informative and specific (2-4 paragraphs unless the question is very complex)
- Wherever appropriate use the names of specific wines, regions, and terminology
- Use bullet points for key information
- Try to quote directly from magazines when relevant (use quotation marks)
- If magazines don't contain specific info, state this briefly
- End with source citations: "Sommelier India, <issue number>, <year>"
- Make sure you contain ZERO bold, italics, or other formatting.

Be direct and focused - provide depth without being wordy."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,  # Further reduced for maximum speed
            temperature=0.3
        )

        answer = response.choices[0].message.content
        logger.info(f"✅ Generated answer: {answer[:100]}...")
        
        # Preserve formatting by not modifying the response
        return jsonify({
            "answer": answer,
            "relevant": relevant,
            "debug_chunks": relevant_chunks
        })

    except Exception as e:
        logger.error(f"❌ Error in ask endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/")
def serve_html():
    return send_from_directory(".", "index.html")

@app.route("/debug")
def debug_info():
    """Debug endpoint to check system status"""
    info = {
        "index_loaded": index is not None,
        "chunks_loaded": len(chunks) if chunks else 0,
        "index_vectors": index.ntotal if index else 0,
        "files_exist": {
            "index": os.path.exists("magazine_index.faiss"),
            "chunks": os.path.exists("magazine_chunks.pkl")
        }
    }
    if chunks:
        info["sample_chunk"] = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
    return jsonify(info)

@app.route("/followup", methods=["POST"])
def get_followup_questions():
    try:
        data = request.get_json()
        previous_question = data.get("previous_question", "")
        
        if not previous_question:
            # Return Popular Questions without validation for speed
            return jsonify({
                "title": "Popular Questions",
                "questions": [
                    "What are the best wine pairings for summer dishes?",
                    "How should I store my wine collection properly?",
                    "What's the difference between Old World and New World wines?"
                ]
            })
        
        # Generate more follow-up questions for validation
        prompt = f"""Based on this wine-related question: "{previous_question}"

Generate 10 natural follow-up questions that someone might ask next. Make them specific and relevant to wine knowledge that would likely be covered in wine magazines.

Format as a simple list:
1. [question]
2. [question]
3. [question]
4. [question]
5. [question]
6. [question]
7. [question]
8. [question]
9. [question]
10. [question]"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,  # Increased for more questions
            temperature=0.7
        )

        answer = response.choices[0].message.content.strip()

        # Parse the response to extract questions
        lines = answer.split('\n')
        raw_questions = []
        for line in lines:
            if line.strip() and any(line.startswith(f'{i}.') for i in range(1, 11)):
                question = line.split('.', 1)[1].strip()
                raw_questions.append(question)

        # Validate each question using magazine data
        valid_questions = []
        for q in raw_questions:
            if validate_question_has_good_answer(q):
                valid_questions.append(q)
            if len(valid_questions) >= 3:
                break

        # If we don't have enough valid questions, add fallbacks
        if len(valid_questions) < 3:
            fallback_questions = [
                "Tell me more about wine terminology",
                "What are some wine tasting techniques?",
                "How do wine regions affect flavor?"
            ]
            for fallback in fallback_questions:
                if len(valid_questions) >= 3:
                    break
                if fallback not in valid_questions:
                    valid_questions.append(fallback)

        return jsonify({
            "title": "Follow-up Questions",
            "questions": valid_questions[:3]  # Ensure max 3 questions
        })

    except Exception as e:
        logger.error(f"❌ Error generating follow-up questions: {str(e)}")
        return jsonify({
            "title": "Suggested Questions",
            "questions": [
                "Tell me more about wine styles",
                "What are some wine tasting tips?",
                "How do I choose the right wine?"
            ]
        })

def validate_question_has_good_answer(question):
    """Check if a question has relevant content in the magazine database"""
    try:
        if index is None or not chunks:
            return False
        
        # Generate embedding for the question
        query_embedding = embed_query(question)
        D, I = index.search(np.array([query_embedding]), k=3)
        # Vector search
        vector_match = len(D[0]) > 0 and D[0][0] < 1.2
        # Keyword search
        keyword_match = any(question.lower() in chunk.lower() for chunk in chunks)
        return vector_match or keyword_match
        
    except Exception as e:
        logger.warning(f"⚠️  Error validating question '{question}': {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🍷 Wine Magazine Assistant starting...")
    if index is None or not chunks:
        logger.error("❌ System not properly initialized. Please run extract_and_index.py first!")
    else:
        logger.info("✅ System ready!")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    logger.info("🌐 Running on http://0.0.0.0:8010")