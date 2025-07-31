import fitz  # PyMuPDF

mport openai
from dotenv import load_dotenv
import faiss
import os
import pickle
import numpy as np
import logging
from datetime import datetime
import concurrent.futures
import time
import asyncio
import aiohttp
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment or .env file")
client = openai.OpenAI(api_key=openai_api_key)

def clean_single_chunk(chunk_data):
    """Clean a single chunk - designed for parallel processing"""
    i, chunk = chunk_data
    
    # Extract the tag and content
    lines = chunk.split('\n', 1)
    if len(lines) == 2:
        tag, content = lines
    else:
        tag = f"[Unknown source]"
        content = chunk
    
    # Truncate content for faster processing
    truncated_content = content[:1500]  # Reduced from 3000
    
    prompt = f"""Clean this wine magazine text. Remove OCR errors, fix spacing, preserve wine info. If corrupted, return "CORRUPTED_TEXT".

Text: {truncated_content}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,  # Reduced from 1500
            temperature=0
        )
        
        cleaned_text = response.choices[0].message.content.strip()
        
        if cleaned_text == "CORRUPTED_TEXT":
            return i, chunk
        else:
            return i, f"{tag}\n{cleaned_text}"
            
    except Exception as e:
        logger.warning(f"   ⚠️  Failed to clean chunk {i}: {str(e)}")
        return i, chunk

async def clean_single_chunk_async(session, chunk_data, semaphore):
    """Clean a single chunk asynchronously"""
    async with semaphore:
        i, chunk = chunk_data
        
        # Extract the tag and content
        lines = chunk.split('\n', 1)
        if len(lines) == 2:
            tag, content = lines
        else:
            tag = f"[Unknown source]"
            content = chunk
        
        # Even more truncated for speed
        truncated_content = content[:800]
        
        # Minimal prompt for maximum speed
        prompt = f"Clean wine text, fix OCR errors:\n{truncated_content}"
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": 0
        }
        
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    cleaned_text = result["choices"][0]["message"]["content"].strip()
                    return i, f"{tag}\n{cleaned_text}"
                else:
                    logger.warning(f"   ⚠️  API error for chunk {i}: {response.status}")
                    return i, chunk
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to clean chunk {i}: {str(e)}")
            return i, chunk

async def clean_chunks_async(all_chunks):
    """Clean chunks with maximum async concurrency"""
    logger.info(f"🚀 Starting ultra-fast async cleaning for {len(all_chunks)} chunks...")
    
    # Create chunk data with indices
    chunk_data = [(i+1, chunk) for i, chunk in enumerate(all_chunks)]
    
    # Use high concurrency - adjust based on your rate limits
    semaphore = asyncio.Semaphore(50)  # 50 concurrent requests
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            clean_single_chunk_async(session, chunk_info, semaphore)
            for chunk_info in chunk_data
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        logger.info(f"   ⚡ Async processing completed in {end_time - start_time:.1f}s")
    
    # Sort results by index and extract cleaned chunks
    cleaned_chunks = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"   ❌ Task failed: {result}")
            continue
        try:
            chunk_idx, cleaned_chunk = result
            cleaned_chunks.append((chunk_idx, cleaned_chunk))
        except:
            continue
    
    # Sort by index and return just the chunks
    cleaned_chunks.sort(key=lambda x: x[0])
    final_chunks = [chunk for _, chunk in cleaned_chunks]
    
    logger.info(f"✅ Ultra-fast cleaning completed: {len(final_chunks)} chunks processed")
    return final_chunks

def clean_and_format_chunks(all_chunks):
    """Wrapper to run async cleaning"""
    return asyncio.run(clean_chunks_async(all_chunks))

def extract_text_from_pdf(file_path):
    logger.info(f"📖 Opening PDF: {os.path.basename(file_path)}")
    doc = fitz.open(file_path)
    text_chunks = []
    
    logger.info(f"📄 Processing {len(doc)} pages...")
    for page_num, page in enumerate(doc):
        logger.info(f"   Reading page {page_num + 1}/{len(doc)}")
        text = page.get_text()
        if text.strip():
            tag = f"[Page {page_num + 1} of {os.path.basename(file_path)}]"
            text_chunks.append(f"{tag}\n{text.strip()}")
            logger.info(f"   ✅ Extracted {len(text.strip())} characters from page {page_num + 1}")
        else:
            logger.warning(f"   ⚠️  Page {page_num + 1} appears to be empty")
    
    logger.info(f"✅ Completed extraction: {len(text_chunks)} non-empty pages from {os.path.basename(file_path)}")
    return text_chunks

def embed_texts(texts):
    logger.info(f"🔄 Starting embedding generation for {len(texts)} text chunks")
    embeddings = []
    
    # Larger batches for faster embedding
    batch_size = 100  # Increased from 10
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch = texts[i:i+batch_size]
        logger.info(f"   Batch {batch_num}/{total_batches}: Processing {len(batch)} chunks...")
        
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            batch_embeddings = [d.embedding for d in response.data]
            embeddings.extend(batch_embeddings)
            logger.info(f"   ✅ Batch {batch_num}/{total_batches} completed successfully")
        except Exception as e:
            logger.error(f"   ❌ Error in batch {batch_num}: {str(e)}")
            # Try smaller batches if large batch fails
            if batch_size > 10:
                logger.info(f"   🔄 Retrying with smaller batches...")
                for j in range(0, len(batch), 10):
                    mini_batch = batch[j:j+10]
                    try:
                        response = client.embeddings.create(
                            input=mini_batch,
                            model="text-embedding-3-small"
                        )
                        batch_embeddings = [d.embedding for d in response.data]
                        embeddings.extend(batch_embeddings)
                    except Exception as mini_e:
                        logger.error(f"   ❌ Mini-batch also failed: {mini_e}")
                        raise
            else:
                raise
    
    logger.info(f"✅ Embedding generation completed: {len(embeddings)} embeddings created")
    return embeddings

def build_combined_index(folder_path="magazine_data"):
    logger.info(f"🚀 Starting combined index build from folder: {folder_path}")
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        logger.error(f"❌ Folder '{folder_path}' does not exist")
        return
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    logger.info(f"📂 Found {len(pdf_files)} PDF files: {pdf_files}")
    
    all_chunks = []
    for i, filename in enumerate(pdf_files, 1):
        logger.info(f"🔄 Processing file {i}/{len(pdf_files)}: {filename}")
        file_path = os.path.join(folder_path, filename)
        chunks = extract_text_from_pdf(file_path)
        all_chunks.extend(chunks)
        logger.info(f"📊 Total chunks so far: {len(all_chunks)}")

    if not all_chunks:
        logger.error("❌ No PDF text found.")
        return

    # Clean and format all chunks at the end
    logger.info("🧹 Starting text cleaning and formatting phase...")
    all_chunks = clean_and_format_chunks(all_chunks)

    logger.info(f"🧠 Generating embeddings for {len(all_chunks)} total chunks...")
    embeddings = embed_texts(all_chunks)
    
    logger.info("🔧 Building FAISS index...")
    dim = len(embeddings[0])
    logger.info(f"   Embedding dimension: {dim}")
    index = faiss.IndexFlatL2(dim)
    
    logger.info("   Adding embeddings to index...")
    index.add(np.array(embeddings).astype("float32"))
    logger.info(f"   ✅ Index built with {index.ntotal} vectors")

    logger.info("💾 Saving chunks to pickle file...")
    with open("magazine_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    logger.info("   ✅ Chunks saved to magazine_chunks.pkl")

    logger.info("💾 Saving FAISS index...")
    faiss.write_index(index, "magazine_index.faiss")
    logger.info("   ✅ Index saved to magazine_index.faiss")
    
    logger.info(f"🎉 SUCCESS! Indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs in '{folder_path}'.")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info("=" * 50)
    logger.info("🍷 WINE MAGAZINE INDEXER STARTED")
    logger.info("=" * 50)
    
    build_combined_index()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info("=" * 50)
    logger.info(f"⏱️  Total processing time: {duration}")
    logger.info("🍷 WINE MAGAZINE INDEXER COMPLETED")
    logger.info("=" * 50)
