from io import BytesIO
from queue import Queue
from typing import cast

import requests
import torch
from PIL import Image
# from colpali_engine.models import ColQwen2, ColQwen2Processor
# from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient
from transformers.utils.import_utils import is_flash_attn_2_available

# Model name
# model_name = "vidore/colqwen2-v1.0"
model_name = "vidore/colpali-v1.3"
# model_name = "vidore/colqwen2.5-v0.2"
# model_name = "Metric-AI/ColQwen2.5-3b-multilingual-v1.0"
collection_name = "arxiv_colqwen2_10"
qdrant_client = QdrantClient("192.168.1.90", port=6334, prefer_grpc=True)

# Detect all available GPUs
available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
print(f"Available devices: {available_devices}")

# GPU pool with models and processors
gpu_pool = {}

# Load the model and processor on each GPU
for device in available_devices:
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,  # or "mps" if on Apple Silicon
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()

    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

    gpu_pool[device] = {"model": model, "processor": processor}

print("Model and processor loaded on all GPUs!")

# Queue for tracking free GPUs
gpu_queue = Queue()
for device in available_devices:
    gpu_queue.put(device)


# Function to load images from a URL
def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


# Function to scale image
def scale_image(image: Image.Image, new_height: int = 3584) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height))


# Helper function to run inference on images
async def run_image(image: Image.Image):
    # Get the next available GPU
    device = gpu_queue.get()
    try:
        print('running on device: ', device, '...')
        # Get the model and processor associated with this GPU
        model = gpu_pool[device]["model"]
        processor = gpu_pool[device]["processor"]

        # Preprocess inputs
        batch_images = processor.process_images([scale_image(image, new_height=3584)]).to(device)
        # Forward passes
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            vector = image_embeddings[0].cpu().float().numpy().tolist()
            print('embedding', len(vector))
    finally:
        # Return GPU to the queue
        gpu_queue.put(device)
        print('done running on device: ', device, '...')

    return vector


# Helper function to run inference on text (queries)
def run_query(query: str):
    # Get the next available GPU
    device = gpu_queue.get()
    try:
        print('running on device: ', device, '...')
        # Get the model and processor associated with this GPU
        model = gpu_pool[device]["model"]
        processor = gpu_pool[device]["processor"]

        # Preprocess inputs
        batch_queries = processor.process_queries([query]).to(device)

        # Forward passes
        with torch.no_grad():
            query_embeddings = model.forward(**batch_queries)
            vector = query_embeddings[0].cpu().float().numpy().tolist()
    finally:
        # Return GPU to the queue
        gpu_queue.put(device)
        print('done running on device: ', device, '...')

    return vector
