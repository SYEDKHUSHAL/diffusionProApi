from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, DiffusionPipeline, StableDiffusionDepth2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline

import torch
# from RealESRGAN import RealESRGAN
from controlnet_aux import OpenposeDetector

stable_diffusion_1_5 = "./models/stable_diffusion_1_5"

device = "cuda"


# real_ESRGAN_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=4)
# real_ESRGAN_model.load_weights('./models/RealESRGAN_x4plus/RealESRGAN_x4plus.pth', download=True)











    





