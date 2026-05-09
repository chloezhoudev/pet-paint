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

# 设备和精度：T4 用 cuda + float16，本地 M1 用 mps + float32
device = os.getenv("DEVICE", "cuda")
dtype = torch.float16 if device == "cuda" else torch.float32

# 加载 SDXL base 模型
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
)

# 加载 LoRA 油画风格权重
pipe.load_lora_weights("classipeintxl.safetensors")

pipe = pipe.to(device)

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    # 1. 读取上传的图片
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((1024, 1024))  # SDXL 原生分辨率

    # 2. 生成油画
    result = pipe(
        prompt="oil painting, thick impasto brushstrokes, painterly, warm golden tones, detailed fur texture, fine art",
        image=image,
        strength=0.6,
        guidance_scale=7.5,
    ).images[0]

    # 3. 保存临时文件
    output_path = f"/tmp/{uuid.uuid4()}.jpg"
    result.save(output_path)

    # 4. 返回图片
    return FileResponse(output_path, media_type="image/jpeg")