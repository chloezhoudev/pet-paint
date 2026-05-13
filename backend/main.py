from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
import io
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# main.py 所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备和精度
device = os.getenv("DEVICE", "cuda")
dtype = torch.float16 if device == "cuda" else torch.float32

# 加载 SDXL base 模型
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
)

# 加载本地 LoRA 权重
pipe.load_lora_weights(BASE_DIR, weight_name="classipeintxl.safetensors")

pipe = pipe.to(device)

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((1024, 1024))

    result = pipe(
        prompt="oil painting, thick impasto brushstrokes, painterly, warm golden tones, detailed fur texture, fine art",
        image=image,
        strength=0.6,
        guidance_scale=7.5,
    ).images[0]

    output_path = f"/tmp/{uuid.uuid4()}.jpg"
    result.save(output_path)

    return FileResponse(output_path, media_type="image/jpeg")