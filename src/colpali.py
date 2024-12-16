from io import BytesIO
from typing import cast
import requests
from queue import Queue
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from qdrant_client import QdrantClient
import torch

# Model name
model_name = "vidore/colqwen2-v1.0"
collection_name = "arxiv_colqwen2_10"
qdrant_client = QdrantClient("192.168.1.90", port=6334, prefer_grpc=True)

# Detect all available GPUs
available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
print(f"Available devices: {available_devices}")

# GPU pool with models and processors
gpu_pool = {}

# Load the model and processor on each GPU
for device in available_devices:
    model = cast(
        ColQwen2,
        ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    ).eval()

    processor = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))

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
def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)
    return image.resize((new_width, new_height))


# Helper function to run inference on images
def run_image(image: Image.Image):
    # Get the next available GPU
    device = gpu_queue.get()
    try:
        print('running on device: ', device, '...')
        # Get the model and processor associated with this GPU
        model = gpu_pool[device]["model"]
        processor = gpu_pool[device]["processor"]

        # Preprocess inputs
        batch_images = processor.process_images([scale_image(image, new_height=1024)]).to(device)

        # Forward passes
        with torch.no_grad():
            image_embeddings = model.forward(**batch_images)
            vector = image_embeddings[0].cpu().float().numpy().tolist()
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
