from io import BytesIO
from typing import cast
import requests
from PIL.ImageFile import ImageFile
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.utils.torch_utils import get_torch_device
from PIL import Image

import torch
from qdrant_client import QdrantClient

model_name = "vidore/colqwen2-v1.0"

device = get_torch_device("auto")
print(f"Using device: {device}")

# Load the model
model = cast(
    ColQwen2,
    ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ),
).eval()

# Load the processor
processor = cast(ColQwen2Processor, ColQwen2Processor.from_pretrained(model_name))


def load_image_from_url(url: str) -> Image.Image:
    """
    Load a PIL image from a valid URL.
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    # Calculate the scaling factor
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


collection_name = 'arxiv_colqwen2_10'
qdrant_client = QdrantClient("192.168.1.90", port=6334, prefer_grpc=True)


def run_image(image: ImageFile):
    # Preprocess inputs
    batch_images = processor.process_images([scale_image(image, new_height=512)]).to(device)

    # Forward passes
    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)
        vector = image_embeddings[0].cpu().float().numpy().tolist()
    return vector


def run_query(query: str):
    # Preprocess inputs
    batch_queries = processor.process_queries([query]).to(device)

    # Forward passes
    with torch.no_grad():
        query_embeddings = model.forward(**batch_queries)
        vector = query_embeddings[0].cpu().float().numpy().tolist()
    return vector
