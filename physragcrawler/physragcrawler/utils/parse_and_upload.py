import json
from typing import List, Dict
import uuid
from openai import OpenAI
from upstash_vector import Index
import os
import logging
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


class DocumentParser:
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.batch_size = 25  # Reduced batch size
        self.max_concurrent_batches = 2  # Reduced concurrent batches

    def _flatten_json(self, json_obj: Dict) -> str:
        """Optimized flattening with string concatenation"""
        parts = []
        if 'title' in json_obj and json_obj['title']:
            parts.append(f"Title: {json_obj['title']}")
        if 'abstract' in json_obj and json_obj['abstract']:
            parts.append(f"Abstract: {json_obj['abstract']}")
        if 'authors' in json_obj and json_obj['authors']:
            parts.append(f"Authors: {json_obj['authors']}")
        if 'comments' in json_obj and json_obj['comments']:
            parts.append(f"Comments: {json_obj['comments']}")
        
        return " ".join(parts)

    def _create_chunks(self, text: str) -> List[str]:
        """Optimized chunking with less string operations"""
        if not text or not isinstance(text, str):
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            if end < text_len:
                # Find last space only in a small window
                window_start = max(end - 50, start)  # Look back max 50 chars
                last_space = text.rfind(' ', window_start, end)
                if last_space != -1:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= text_len:
                break
            start = end - self.overlap_size

        return chunks

    async def _process_batch_async(self, batch_texts: List[str], vector_store: 'UpstashVectorStore', 
                                 source_url: str, current_count: int) -> None:
        """Asynchronous batch processing"""
        try:
            all_chunks = []
            for text in batch_texts:
                chunks = self._create_chunks(text)
                all_chunks.extend(chunks)

            if all_chunks:
                chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
                await vector_store.add_async(
                    ids=chunk_ids,
                    documents=all_chunks,
                    link=source_url
                )
                print(f"Processed {current_count} papers")
                
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")

    async def parse_and_upload(self, json_path: str, vector_store: 'UpstashVectorStore', source_url: str) -> None:
        """Asynchronous parsing and uploading with rate limiting"""
        buffer = []
        processed_count = 0
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                while True:
                    lines = list(islice(f, self.batch_size))
                    if not lines:
                        break

                    batch_texts = []
                    for line in lines:
                        if line.strip():
                            try:
                                json_obj = json.loads(line)
                                paper_text = self._flatten_json(json_obj)
                                if paper_text:
                                    batch_texts.append(paper_text)
                                    processed_count += 1
                            except json.JSONDecodeError:
                                continue

                    if batch_texts:
                        async with semaphore:
                            try:
                                await self._process_batch_async(
                                    batch_texts, 
                                    vector_store, 
                                    source_url, 
                                    processed_count
                                )
                                # Add delay between batches
                                await asyncio.sleep(1)
                            except Exception as e:
                                print(f"Error processing batch: {str(e)}")
                                # Wait longer if we hit rate limits
                                if 'rate_limit' in str(e).lower():
                                    await asyncio.sleep(5)

            print(f"\nProcessing complete:")
            print(f"Total papers processed: {processed_count}")
            return processed_count
            
        except Exception as e:
            print(f"Critical error: {str(e)}")
            raise


    
class UpstashVectorStore:
    def __init__(
            self,
            url: str,
            token: str
    ):
        self.client = OpenAI()
        self.index = Index(url=url, token=token)
        self.embedding_batch_size = 50  # Reduced batch size
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced workers
        self.tokens_per_min_limit = 1000000
        self.current_tokens = 0
        self.last_reset = time.time()

    async def wait_for_rate_limit(self, requested_tokens: int) -> None:
        """Implement rate limiting for OpenAI API"""
        current_time = time.time()
        
        # Reset token count if a minute has passed
        if current_time - self.last_reset >= 60:
            self.current_tokens = 0
            self.last_reset = current_time
        
        # If adding these tokens would exceed the limit, wait
        if self.current_tokens + requested_tokens > self.tokens_per_min_limit:
            wait_time = 60 - (current_time - self.last_reset)
            print(f"Rate limit approached. Waiting {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)
            self.current_tokens = 0
            self.last_reset = time.time()
        
        self.current_tokens += requested_tokens

    async def get_embeddings_with_retry(
            self,
            documents: List[str],
            model: str = "text-embedding-ada-002",
            max_retries: int = 5
    ) -> List[List[float]]:
        """Get embeddings with exponential backoff retry"""
        delay = 1  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Estimate token count (rough approximation)
                estimated_tokens = sum(len(doc.split()) * 1.3 for doc in documents)
                await self.wait_for_rate_limit(int(estimated_tokens))
                
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.embeddings.create(input=documents, model=model)
                )
                return [data.embedding for data in response.data]
                
            except Exception as e:
                if 'rate_limit' in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        print(f"Rate limit hit. Waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                raise

    async def get_embeddings_async(
            self,
            documents: List[str],
            model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """Asynchronous embedding generation with batching"""
        cleaned_documents = [doc.replace("\n", " ").strip() for doc in documents if doc]
        embeddings = []
        
        # Process in smaller batches
        for i in range(0, len(cleaned_documents), self.embedding_batch_size):
            batch = cleaned_documents[i:i + self.embedding_batch_size]
            try:
                batch_embeddings = await self.get_embeddings_with_retry(batch)
                embeddings.extend(batch_embeddings)
                # Add small delay between batches
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                raise

        return embeddings
    

    async def add_async(
            self,
            ids: List[str],
            documents: List[str],
            link: str
    ) -> None:
        """Asynchronous document addition with rate limiting"""
        try:
            if not documents or not ids:
                raise ValueError("Empty documents or ids")
            
            if len(documents) != len(ids):
                raise ValueError(f"Mismatch between documents ({len(documents)}) and ids ({len(ids)})")

            # Get embeddings with rate limiting
            embeddings = await self.get_embeddings_async(documents)
            
            # Create vectors for upserting
            vectors = [
                (id, embedding, {"text": doc, "url": link})
                for id, embedding, doc in zip(ids, embeddings, documents)
            ]
            
            # Process upserts in smaller batches
            upsert_batch_size = 25
            for i in range(0, len(vectors), upsert_batch_size):
                batch = vectors[i:i + upsert_batch_size]
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.index.upsert(vectors=batch)
                    )
                    # Add small delay between upserts
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Error upserting batch {i//upsert_batch_size + 1}: {str(e)}")
                    raise
                
        except Exception as e:
            print(f"Error in async add: {str(e)}")
            raise



# Configure the logger
logging.basicConfig(filename='phys_rag_log.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    # Initialize the parser, and vector store
    parser = DocumentParser()
    vector_store = UpstashVectorStore(url=os.environ.get("UPSTASH_VECTOR_REST_URL"), token=os.environ.get("UPSTASH_VECTOR_REST_TOKEN"))
    
    # Parse and upload a JSON document
    await parser.parse_and_upload(
        json_path="arxiv-metadata-oai-snapshot.json",
        vector_store=vector_store,
        source_url="https://www.kaggle.com/datasets/Cornell-University/arxiv/data"
    )

if __name__ == "__main__":
    asyncio.run(main())


#print(f"Total papers processed: {total_processed}")