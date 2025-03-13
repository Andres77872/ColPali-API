from fastapi import FastAPI, UploadFile, Request, HTTPException
from PIL import Image
from starlette.middleware.cors import CORSMiddleware

from src.colpali import run_image, run_query
import io
import time

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
        if int(request.headers['content-length']) > 8388608:
            process_time = time.time() - start_time
            raise HTTPException(status_code=413,
                                detail="Max sie 8 mib",
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
async def emb_image(image: UploadFile):
    # Read the contents of the uploaded file
    image_bytes = await image.read()

    # Open the image using PIL
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Pass the PIL image to run_image
    result = await run_image(pil_image)

    return result


@app.post("/text")
async def emb_query(text):

    # Pass the PIL image to run_query
    result = run_query(text)

    return result
