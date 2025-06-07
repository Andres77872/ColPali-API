import asyncio
import functools
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from PIL import Image
from fastapi import FastAPI, UploadFile, Request, HTTPException, File, Form
from starlette.middleware.cors import CORSMiddleware

from src.colpali import run_image, run_query

# Start a thread executor at application startup with max_workers = num of GPUs
executor = ThreadPoolExecutor(max_workers=3)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()

    if request.method == 'POST':
        if int(request.headers['content-length']) > 8388608 * 32:
            process_time = time.time() - start_time
            raise HTTPException(status_code=413,
                                detail="Max sie 32 mib",
                                headers={
                                    'X-Process-Time': str(process_time),
                                    'Access-Control-Allow-Origin': '*'
                                })

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/image")
async def emb_image(images: List[UploadFile] = File(...),
                    size: int = File(3584),
                    pool_factor: int = File(2)):
    pil_images = []
    for image in images:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_images.append(pil_image)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor,
                                        functools.partial(run_image,
                                                          pil_images,
                                                          size=size,
                                                          pool_factor=pool_factor))
    return result


@app.post("/image_local")
async def emb_image_local(images: List[str] = Form(),
                          size: int = Form(3584),
                          pool_factor: int = Form(2)):
    pil_images = []
    for image in images:
        pil_image = Image.open(image)
        pil_images.append(pil_image)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor,
                                        functools.partial(run_image,
                                                          pil_images,
                                                          size=size,
                                                          pool_factor=pool_factor))
    return result


@app.post("/text")
async def emb_query(text):
    # Pass the PIL image to run_query
    result = run_query(text)

    return result
