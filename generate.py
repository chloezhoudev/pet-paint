import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 1. 选择模型
model_id = "runwayml/stable-diffusion-v1-5"

# 2. 加载模型
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None,
)

# 3. 使用 M1 GPU
pipe = pipe.to("mps")

# pipe.enable_attention_slicing()

# 4. 读取你的宠物照片
image = Image.open("pet.jpg").convert("RGB")
image = image.resize((512, 512))

# 5. 生成油画风格的图片
result = pipe(
    prompt="oil painting, thick impasto brushstrokes, painterly, warm golden tones, detailed fur texture, fine art",
    image=image,
    strength=0.55,
    guidance_scale=7.5,
).images[0]

# 6. 保存结果
result.save("output.jpg")
print("完成！图片已保存为 output.jpg")