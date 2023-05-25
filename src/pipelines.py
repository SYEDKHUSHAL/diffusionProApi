from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, DiffusionPipeline, StableDiffusionDepth2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline

import torch
# from RealESRGAN import RealESRGAN
from controlnet_aux import OpenposeDetector


device = "cuda"

# waifu_diffusion_pipe = StableDiffusionPipeline.from_pretrained("./models/waifu-diffusion",
#                                                                 revision="fp32", 
#                                                                 torch_dtype=torch.float16, 
#                                                                 local_files_only = True
#                                                                 ).to(device)


# stable_diffusion_1_5_pipe = StableDiffusionPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                 revision="fp32", 
#                                                 torch_dtype=torch.float16, 
#                                                 local_files_only = True
#                                                 ).to(device)


# stable_diffusion_2_1_pipe = StableDiffusionPipeline.from_pretrained("./models/stable-diffusion-2-1",
#                                                                      revision="fp16" if torch.cuda.is_available() else "fp32",
#                                                                      torch_dtype=torch.float16, 
#                                                                      local_files_only = True,
#                                                                      scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/stable-diffusion-2-1", subfolder="scheduler"),
#                                                                      safety_checker = None,
#                                                                      feature_extractor = None,
#                                                                      ).to(device)


# stable_diffusion_2_1_img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("./models/stable-diffusion-2-1", 
#                                                     torch_dtype=torch.float16, 
#                                                     local_files_only = True,
#                                                     revision="fp16" if torch.cuda.is_available() else "fp32",
#                                                     scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/stable-diffusion-2-1", subfolder="scheduler"),
#                                                     safety_checker = None,
#                                                     feature_extractor = None,
#                                                 ).to(device)



# lina_qurf_pipe = StableDiffusionPipeline.from_pretrained("./models/anything-v3.0", 
#                                                         revision="fp16" if torch.cuda.is_available() else "fp32",
#                                                         torch_dtype=torch.float16,  
#                                                         local_files_only = True,
#                                                         safety_checker = None,
#                                                         feature_extractor = None,
#                                                         scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/anything-v3.0",
#                                                                                                                  subfolder="scheduler"
#                                                                                                                  )
#                                                     ).to(device)
    



lina_qurf_img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("./models/anything-v3.0", 
                                                    torch_dtype=torch.float16, 
                                                    local_files_only = True,
                                                    revision="fp16" if torch.cuda.is_available() else "fp32",
                                                    scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/anything-v3.0", subfolder="scheduler"),
                                                    safety_checker = None,
                                                    feature_extractor = None,
                                                ).to(device)





# real_ESRGAN_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=4)
# real_ESRGAN_model.load_weights('./models/RealESRGAN_x4plus/RealESRGAN_x4plus.pth', download=True)



# openpose_detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


# openpose_controlnet = ControlNetModel.from_pretrained("./models/stable-diffusion-v1-5-controlnet-openpose",
#                                               revision="fp32", 
#                                               torch_dtype=torch.float16, 
#                                               local_files_only = True
#                                               ).to(device)

# openpose_controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                                               controlnet=openpose_controlnet, 
#                                                                               torch_dtype=torch.float16, 
#                                                                               local_files_only = True
#                                                                               ).to(device)


# controlnet_canny = ControlNetModel.from_pretrained("./models/sd-controlnet-canny",
#                                                     revision="fp32", 
#                                                     torch_dtype=torch.float16, 
#                                                     local_files_only = True
#                                                     ).to(device)

# controlnet_canny_pipe = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                                            controlnet=controlnet_canny,
#                                                                             torch_dtype=torch.float16, 
#                                                                             local_files_only = True
#                                                                             ).to(device)




# controlnet_shuffle = ControlNetModel.from_pretrained("./models/control_v11e_sd15_shuffle", torch_dtype=torch.float16)
# pipe_shuffle = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_shuffle, 
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)


# controlnet_scribble = ControlNetModel.from_pretrained("./models/controlnet_scribble", torch_dtype=torch.float16)
# pipe_scribble = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_scribble,
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)

# controlnet_hed = ControlNetModel.from_pretrained("./models/controlnet_hed", torch_dtype=torch.float16)
# pipe_hed = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_hed, 
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)

# controlnet_lineart = ControlNetModel.from_pretrained("./models/controlnet_lineart", torch_dtype=torch.float16)
# pipe_lineart = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_lineart,
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)

# controlnet_softEdge = ControlNetModel.from_pretrained("./models/controlnet_softEdge", torch_dtype=torch.float16)
# pipe_softEdge = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_softEdge,
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)
    

# controlnet_normalBae = ControlNetModel.from_pretrained("./models/controlnet_normalBae", torch_dtype=torch.float16)
# pipe_normalBae = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_normalBae,
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)


# controlnet_mlsd = ControlNetModel.from_pretrained("./models/controlnet_mlsd", torch_dtype=torch.float16)
# pipe_mlsd = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
#                                                         controlnet=controlnet_mlsd,
#                                                         torch_dtype=torch.float16
#                                                         ).to(device)
