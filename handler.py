import base64
import concurrent.futures
import os
import time
from io import BytesIO
from queue import Queue
from typing import cast

import runpod
import requests
import torch
from PIL import Image
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from runpod.serverless.utils.rp_validator import validate

from schemas import INPUT_SCHEMA

# Import the schema (assuming it exists or will be created)
# from schemas import INPUT_SCHEMA


torch.cuda.empty_cache()


class ColPaliModelHandler:
    def __init__(self):
        self.model_name = "vidore/colqwen2.5-v0.2"
        self.gpu_pool = {}
        self.gpu_queue = Queue()
        self.available_devices = []
        self.load_models()

    def load_models(self):
        # Detect all available GPUs
        self.available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not self.available_devices:
            self.available_devices = ['cpu']

        print(f"Available devices: {self.available_devices}")
        print('is_flash_attn_2_available', is_flash_attn_2_available())

        model_dir = f"./models/{self.model_name}"
        # Load the model and processor on each GPU
        for device in self.available_devices:
            model = ColQwen2_5.from_pretrained(
                self.model_name,
                cache_dir=model_dir,
                local_files_only=False,
                trust_remote_code=True,
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ).eval()

            processor = cast(ColQwen2_5_Processor, ColQwen2_5_Processor.from_pretrained(self.model_name))
            self.gpu_pool[device] = {"model": model, "processor": processor}

        print("Model and processor loaded on all GPUs!")

        # Initialize GPU queue
        for device in self.available_devices:
            self.gpu_queue.put(device)

    def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL"""
        response = requests.get(url)
        return Image.open(BytesIO(response.content))

    def load_image_from_base64(self, base64_str: str) -> Image.Image:
        """Load image from base64 string"""
        if base64_str.startswith('data:image'):
            # Remove data:image/png;base64, prefix if present
            base64_str = base64_str.split(',')[1]

        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))

    def load_images(self, image_inputs: list[str]) -> list[Image.Image]:
        """Load images from URLs or base64 strings"""
        images = []
        for img_input in image_inputs:
            if img_input.startswith('http'):
                # It's a URL
                images.append(self.load_image_from_url(img_input))
            elif img_input.startswith('data:image') or len(img_input) > 100:
                # It's likely base64
                images.append(self.load_image_from_base64(img_input))
            else:
                raise ValueError(f"Invalid image input format: {img_input[:50]}...")
        return images

    def scale_image(self, image: Image.Image, new_height: int = 3584) -> Image.Image:
        """Scale image to new height while maintaining aspect ratio"""
        width, height = image.size
        aspect_ratio = width / height
        new_width = int(new_height * aspect_ratio)
        return image.resize((new_width, new_height))

    def p_scale_images(self, images: list[Image.Image], new_height: int = 3584, max_workers: int = None) -> list[
        Image.Image]:
        """Parallel image scaling"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            resized_images = list(executor.map(lambda img: self.scale_image(img, new_height=new_height), images))
        return resized_images

    def run_image_embedding(self, images: list[Image.Image], size=3584, pool_factor=2):
        """Process images and return embeddings"""
        device = self.gpu_queue.get()  # Pick available GPU
        try:
            st = time.time()
            print(f'Running image embedding on device: {device}, size: {size}...')

            # Process images
            batch_images = self.gpu_pool[device]["processor"].process_images(images).to(device)

            with torch.no_grad():
                embedding = self.gpu_pool[device]["model"](**batch_images)

            if pool_factor > 1:
                token_pooler = HierarchicalTokenPooler(pool_factor=pool_factor)
                embedding = token_pooler.pool_embeddings(
                    embedding,
                    padding=True,
                    padding_side=self.gpu_pool[device]["processor"].tokenizer.padding_side,
                ).cpu().float().numpy().tolist()
            else:
                embedding = embedding.cpu().float().numpy().tolist()

            print(f'Finished image embedding on device: {device}, time: {(time.time() - st):2f}s.')
        finally:
            self.gpu_queue.put(device)  # GPU is free again

        return embedding

    def run_query_embedding(self, queries: list[str]):
        """Process queries and return embeddings"""
        device = self.gpu_queue.get()
        try:
            print(f'Running query embedding on device: {device}...')
            st = time.time()

            # Get the model and processor associated with this GPU
            model = self.gpu_pool[device]["model"]
            processor = self.gpu_pool[device]["processor"]

            # Process all queries at once
            batch_queries = processor.process_queries(queries).to(device)

            # Forward passes
            with torch.no_grad():
                query_embeddings = model.forward(**batch_queries)
                # Convert all embeddings to list format
                vectors = [emb.cpu().float().numpy().tolist() for emb in query_embeddings]

            print(f'Finished query embedding on device: {device}, time: {(time.time() - st):2f}s.')
        finally:
            # Return GPU to the queue
            self.gpu_queue.put(device)
            print(f'Done running query embedding on device: {device}...')

        return vectors


# Initialize the model handler
MODELS = ColPaliModelHandler()


@torch.inference_mode()
def process_colpali(job):
    """
    Process ColPali operations - either image embedding or query embedding
    """
    # Debug logging
    import json, pprint

    print("[process_colpali] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)

    job_input = job["input"]

    print("[process_colpali] job['input'] payload:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4, compact=False)

    # Input validation
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
    except Exception as err:
        import traceback
        print("[process_colpali] validate(...) raised an exception:", err, flush=True)
        traceback.print_exc()
        raise

    print("[process_colpali] validate(...) returned:")
    try:
        print(json.dumps(validated_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(validated_input, depth=4, compact=False)

    if "errors" in validated_input:
        return {"error": validated_input["errors"]}

    job_input = validated_input["validated_input"]

    operation_type = job_input["operation_type"]
    size = job_input.get("size", 3584)
    pool_factor = job_input.get("pool_factor", 2)

    try:
        if operation_type == "embed_images":
            if "images" not in job_input or not job_input["images"]:
                return {"error": "Images are required for embed_images operation"}

            # Load images from URLs or base64
            images = MODELS.load_images(job_input["images"])

            # Process images and get embeddings
            embeddings = MODELS.run_image_embedding(images, size=size, pool_factor=pool_factor)

            return {
                "embeddings": embeddings,
                "operation_type": "embed_images",
                "num_images": len(images),
                "size": size,
                "pool_factor": pool_factor
            }

        elif operation_type == "embed_query":
            if "queries" not in job_input or not job_input["queries"]:
                return {"error": "Queries are required for embed_query operation"}

            queries = job_input["queries"]

            # Process queries and get embeddings
            embeddings = MODELS.run_query_embedding(queries)

            return {
                "embeddings": embeddings,
                "operation_type": "embed_query",
                "num_queries": len(queries)
            }

        else:
            return {"error": f"Unknown operation_type: {operation_type}"}

    except Exception as e:
        import traceback
        print(f"Error processing ColPali operation: {str(e)}")
        traceback.print_exc()
        return {"error": f"Processing failed: {str(e)}"}


runpod.serverless.start({"handler": process_colpali})