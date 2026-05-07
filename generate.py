import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image

# 1. 加载 ControlNet 模型（Canny 版本，配合 SD v1.5）
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32,
)

# 2. 加载主 pipeline，把 ControlNet 注入进去
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,         # 把 ControlNet 作为结构约束插件传入
    torch_dtype=torch.float32,
    safety_checker=None,
)

# 3. 使用 M1 GPU
pipe = pipe.to("mps")

# 4. 读取宠物照片
image = Image.open("pet.jpg").convert("RGB")
image = image.resize((512, 512))

# 5. 用 Canny 边缘检测生成"结构骨架图"
#    这张图告诉 ControlNet：轮廓必须遵守这些线条
canny = CannyDetector()
control_image = canny(image)       # 输入原图，输出黑白边缘图

# 6. 生成油画（同时传入原图和边缘图）
result = pipe(
    prompt="oil painting, thick impasto brushstrokes, painterly, warm golden tones, detailed fur texture, fine art",
    image=image,                   # 原图：提供颜色和内容
    control_image=control_image,   # 边缘图：提供结构约束
    strength=0.55,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,  # ControlNet 影响强度，1.0=完全遵守，0=忽略
).images[0]

# 7. 保存结果
result.save("output_controlnet.jpg")   # 用新文件名，方便和旧结果对比
print("完成！图片已保存为 output_controlnet.jpg")