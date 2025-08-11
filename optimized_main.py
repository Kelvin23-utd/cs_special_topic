# optimized_main.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
import logging
from typing import List, Dict, Any
import time
import uvloop  # High-performance event loop
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from functools import lru_cache
import hashlib
import json

# Use high-performance event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedJailbreakClassifier:
    def __init__(self, model_name="jackhhao/jailbreak-classifier", max_workers=4):
        self.model_name = model_name
        self.max_workers = max_workers
        self.request_queue = queue.Queue(maxsize=10000)
        self.response_futures = {}
        self.cache = {}
        self.cache_size_limit = 10000
        
        # Load model components separately for better control
        logger.info("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Set model to evaluation mode and optimize
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on GPU")
        else:
            # Optimize for CPU inference
            self.model = torch.jit.optimize_for_inference(self.model)
            logger.info("Model optimized for CPU inference")
        
        # Create thread pool for CPU-bound inference
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Start background processing threads
        self.workers = []
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"Initialized with {max_workers} worker threads")

    def _compute_cache_key(self, text: str) -> str:
        """Generate cache key for input text"""
        return hashlib.md5(text.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def _cached_tokenize(self, text: str):
        """Cache tokenization results"""
        return self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

    def _batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts in a single batch"""
        if not texts:
            return []
        
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Inference with no gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            confidence_scores = torch.max(probabilities, dim=-1)[0]
        
        # Convert to CPU and extract results
        predictions = predictions.cpu().numpy()
        confidence_scores = confidence_scores.cpu().numpy()
        
        results = []
        label_map = {0: "SAFE", 1: "JAILBREAK"}  # Adjust based on your model
        
        for pred, score in zip(predictions, confidence_scores):
            results.append({
                "label": label_map.get(pred, "UNKNOWN"),
                "score": float(score)
            })
        
        return results

    def _worker_loop(self):
        """Background worker for processing requests in batches"""
        batch_size = 32  # Optimal batch size for most models
        batch_timeout = 0.01  # 10ms timeout for batching
        
        while True:
            batch = []
            batch_futures = []
            
            # Collect requests for batching
            start_time = time.time()
            while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                try:
                    request_id, text, future = self.request_queue.get(timeout=0.001)
                    batch.append(text)
                    batch_futures.append((request_id, future))
                except queue.Empty:
                    if batch:
                        break
                    continue
            
            if not batch:
                continue
            
            try:
                # Process batch
                results = self._batch_classify(batch)
                
                # Return results to futures
                for (request_id, future), result in zip(batch_futures, results):
                    if not future.cancelled():
                        future.set_result(result)
                        
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Set error for all futures in batch
                for request_id, future in batch_futures:
                    if not future.cancelled():
                        future.set_exception(e)

    async def classify_async(self, text: str) -> Dict[str, Any]:
        """Async interface for classification"""
        # Check cache first
        cache_key = self._compute_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create future for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_id = id(future)
        
        try:
            # Add to processing queue
            self.request_queue.put((request_id, text, future), timeout=0.1)
            
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=10.0)
            
            # Cache result
            if len(self.cache) < self.cache_size_limit:
                self.cache[cache_key] = result
            
            return result
            
        except asyncio.TimeoutError:
            future.cancel()
            raise HTTPException(status_code=408, detail="Request timeout")
        except queue.Full:
            raise HTTPException(status_code=503, detail="Service overloaded")

# Initialize the optimized classifier
logger.info("Initializing optimized classifier...")
classifier = OptimizedJailbreakClassifier(max_workers=8)  # Adjust based on CPU cores

# FastAPI app with optimizations
app = FastAPI(
    title="High-Performance Jailbreak Classifier",
    docs_url="/docs",
    redoc_url="/redoc"
)

class PromptRequest(BaseModel):
    prompt: str

class BatchPromptRequest(BaseModel):
    prompts: List[str]

class ClassificationResponse(BaseModel):
    prompt: str
    verdict: str
    confidence_score: float
    processing_time_ms: float

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
    total_processing_time_ms: float

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "jailbreak-classifier",
        "version": "2.0-optimized"
    }

@app.get("/metrics")
async def get_metrics():
    """Performance metrics endpoint"""
    return {
        "queue_size": classifier.request_queue.qsize(),
        "cache_size": len(classifier.cache),
        "worker_threads": len(classifier.workers),
        "model_device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_prompt(request: PromptRequest):
    """Single prompt classification endpoint"""
    start_time = time.time()
    
    try:
        result = await classifier.classify_async(request.prompt)
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResponse(
            prompt=request.prompt,
            verdict=result["label"],
            confidence_score=result["score"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchPromptRequest):
    """Batch classification endpoint for even higher throughput"""
    start_time = time.time()
    
    if len(request.prompts) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        # Process all prompts concurrently
        tasks = [classifier.classify_async(prompt) for prompt in request.prompts]
        results = await asyncio.gather(*tasks)
        
        total_processing_time = (time.time() - start_time) * 1000
        
        response_results = []
        for prompt, result in zip(request.prompts, results):
            response_results.append(ClassificationResponse(
                prompt=prompt,
                verdict=result["label"],
                confidence_score=result["score"],
                processing_time_ms=total_processing_time / len(request.prompts)
            ))
        
        return BatchClassificationResponse(
            results=response_results,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 High-performance jailbreak classifier service started")
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Worker threads: {classifier.max_workers}")
    logger.info(f"   - Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"   - Cache limit: {classifier.cache_size_limit}")

if __name__ == "__main__":
    import uvicorn
    
    # High-performance server configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker with internal threading
        loop="uvloop",
        http="httptools",
        access_log=False,  # Disable for performance
        server_header=False,
        date_header=False
    )