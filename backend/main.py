from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import io
import uuid
import os

# 创建 FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型（只加载一次）
model_id = "runwayml/stable-diffusion-v1-5"
device = os.getenv("DEVICE", "mps")
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    safety_checker=None,
)
pipe = pipe.to(device)

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    # 1. 读取上传的图片
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((512, 512))

    # 2. 生成油画
    result = pipe(
        prompt="oil painting, thick impasto brushstrokes, painterly, warm golden tones, detailed fur texture, fine art",
        image=image,
        strength=0.55,
        guidance_scale=7.5,
    ).images[0]

    # 3. 保存结果到临时文件
    output_path = f"/tmp/{uuid.uuid4()}.jpg"
    result.save(output_path)

    # 4. 返回图片
    return FileResponse(output_path, media_type="image/jpeg")