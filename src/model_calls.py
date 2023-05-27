from torch import autocast
import json

# import src.pipe_lines as pipe_lines

from src.pipelines import *
from io import BytesIO
import base64 
from PIL import Image
from controlnet_aux import OpenposeDetector, ContentShuffleDetector, PidiNetDetector, HEDdetector, LineartDetector, NormalBaeDetector, MLSDdetector
import cv2
import numpy as np
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, DiffusionPipeline, StableDiffusionDepth2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline

device = "cuda"
stable_diffusion_1_5 = "./models/stable_diffusion_1_5"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def createCannyImage(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save("./images/canny.png")
    return canny_image





def call_waifu_diffusion(prompt, 
                        seed, 
                        negative_prompt, 
                        height, 
                        width, 
                        guidance_scale, 
                        steps):

    waifu_diffusion_pipe = StableDiffusionPipeline.from_pretrained("./models/waifu-diffusion",
                                                                revision="fp32", 
                                                                torch_dtype=torch.float16, 
                                                                local_files_only = True
                                                                ).to(device)


    with autocast(device): 
        image = waifu_diffusion_pipe(
                            prompt=prompt, 
                            num_inference_steps = steps,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt = 4,
                            width=width,
                            height=height,
                            negative_prompt = negative_prompt,
                            generator = torch.Generator(device= 'cpu').manual_seed(seed),
                            )
    dictobj = {}
    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/generated_waifu{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr


async def call_stable_diffusion_1_5(prompt, 
                                    seed, 
                                    negative_prompt, 
                                    height, 
                                    width, 
                                    guidance_scale, 
                                    steps):

    stable_diffusion_1_5_pipe = StableDiffusionPipeline.from_pretrained("./models/stable_diffusion_1_5",
                                                revision="fp32", 
                                                torch_dtype=torch.float16, 
                                                local_files_only = True
                                                ).to(device)
    with autocast(device): 
        image = stable_diffusion_1_5_pipe(prompt=prompt, 
                                        num_inference_steps = steps,
                                        guidance_scale=guidance_scale,
                                        num_images_per_prompt = 4,
                                        width=width,
                                        height=height,
                                        negative_prompt = negative_prompt,
                                        generator = torch.Generator(device= 'cpu').manual_seed(seed),
                                        )


    dictobj = {}
    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/generated1.4.{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr



def call_lina_qurf(prompt, 
                            seed, 
                            negative_prompt, 
                            height, 
                            width, 
                            guidance_scale, 
                            steps
                            ):

    lina_qurf_pipe = StableDiffusionPipeline.from_pretrained("./models/anything-v3.0", 
                                revision="fp16" if torch.cuda.is_available() else "fp32",
                                torch_dtype=torch.float16,  
                                local_files_only = True,
                                safety_checker = None,
                                feature_extractor = None,
                                scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/anything-v3.0",
                                                                                            subfolder="scheduler"
                                                                                            )
                            ).to(device)

    lina_qurf_pipe.enable_attention_slicing()
    # lina_qurf_pipe.enable_model_cpu_offload()
    lina_qurf_pipe.enable_xformers_memory_efficient_attention()

    with autocast(device): 
        image = lina_qurf_pipe(
            prompt=prompt, 
            num_inference_steps = steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt = 4,
            width=width,
            height=height,
            negative_prompt = negative_prompt,
            generator = torch.Generator(device= 'cpu').manual_seed(seed),
            )


        dictobj = {}
        for num_img in range(4):
            gen_image = image.images[num_img]
            gen_image.save(f"./results/generated_linaqurf{num_img}.png")

            buffer = BytesIO()
            gen_image.save(buffer, format="PNG")
            data = base64.b64encode(buffer.getvalue())
            base64_string = data.decode('utf-8')
            
            dictobj.update({
                f"img{num_img}": base64_string,
            })

    imgstr = json.dumps(dictobj)
    return imgstr


def call_lina_qurf_img2img(image,
                            prompt, 
                            seed, 
                            negative_prompt, 
                            guidance_scale, 
                            strength,
                            steps
                            ):
    img = Image.open(BytesIO(image))

    lina_qurf_img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("./models/anything-v3.0", 
                                                    torch_dtype=torch.float16, 
                                                    local_files_only = True,
                                                    revision="fp16" if torch.cuda.is_available() else "fp32",
                                                    scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/anything-v3.0", subfolder="scheduler"),
                                                    safety_checker = None,
                                                    feature_extractor = None,
                                                ).to(device)

    lina_qurf_img2img_pipe.enable_attention_slicing()
    # lina_qurf_img2img_pipe.enable_model_cpu_offload()
    lina_qurf_img2img_pipe.enable_xformers_memory_efficient_attention()

    with autocast(device): 
        result = lina_qurf_img2img_pipe(
            prompt = prompt,
            num_inference_steps = int(steps),
            num_images_per_prompt = 1,
            negative_prompt = negative_prompt,
            image = img,
            strength = strength,
            guidance_scale = guidance_scale,
            generator = torch.Generator(device= 'cpu').manual_seed(seed),
        )

    image = result.images[0]
    image.save("./results/lina_qufr_img2imgv2.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return imgstr

def call_stable_diffusion_2_1(prompt, 
                                    seed, 
                                    negative_prompt, 
                                    height, 
                                    width, 
                                    guidance_scale, 
                                    steps
                                    ):
    stable_diffusion_2_1_pipe.enable_attention_slicing()
    # stable_diffusion_2_1_pipe.enable_model_cpu_offload()
    stable_diffusion_2_1_pipe.enable_xformers_memory_efficient_attention()

    stable_diffusion_2_1_pipe = StableDiffusionPipeline.from_pretrained("./models/stable-diffusion-2-1",
                                                                     revision="fp16" if torch.cuda.is_available() else "fp32",
                                                                     torch_dtype=torch.float16, 
                                                                     local_files_only = True,
                                                                     scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/stable-diffusion-2-1", subfolder="scheduler"),
                                                                     safety_checker = None,
                                                                     feature_extractor = None,
                                                                     ).to(device)


    with autocast(device): 
        image = stable_diffusion_2_1_pipe(prompt,
                                           negative_prompt=negative_prompt,
                                           num_inference_steps=steps,
                                           num_images_per_prompt = 4,
                                           guidance_scale=guidance_scale,
                                           height= height,
                                           width=width,
                                           generator = torch.Generator(device= 'cpu').manual_seed(seed),
                                           )

    dictobj = {}
    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/generatedv2.{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr

def call_stable_diffusion_2_1_img2img(image, 
                                    prompt, 
                                    seed, 
                                    negative_prompt, 
                                    guidance_scale, 
                                    strength,
                                    steps
                                    ):
    image = Image.open(BytesIO(image))

    stable_diffusion_2_1_img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("./models/stable-diffusion-2-1", 
                                                    torch_dtype=torch.float16, 
                                                    local_files_only = True,
                                                    revision="fp16" if torch.cuda.is_available() else "fp32",
                                                    scheduler = DPMSolverMultistepScheduler.from_pretrained("./models/stable-diffusion-2-1", subfolder="scheduler"),
                                                    safety_checker = None,
                                                    feature_extractor = None,
                                                ).to(device)



    stable_diffusion_2_1_img2img_pipe.enable_attention_slicing()
    # stable_diffusion_2_1_img2img_pipe.enable_model_cpu_offload()
    stable_diffusion_2_1_img2img_pipe.enable_xformers_memory_efficient_attention()

    with autocast(device): 
        result = stable_diffusion_2_1_img2img_pipe(
            prompt = prompt,
            num_inference_steps = steps,
            num_images_per_prompt = 1,
            negative_prompt = negative_prompt,
            image = image,
            strength = strength,
            guidance_scale = guidance_scale,
            generator = torch.Generator(device= 'cpu').manual_seed(seed),
        )
    image = result.images[0]
    image.save("./results/img2imgv2.png")

    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return imgstr

#control nets

def call_controlnet_canny(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    canny_image = createCannyImage(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    canny_base64 = base64.b64encode(buffer.getvalue())
    canny_string = canny_base64.decode('utf-8')
    prompt = prompt

    controlnet_canny = ControlNetModel.from_pretrained("./models/sd-controlnet-canny",
                                                    revision="fp32", 
                                                    torch_dtype=torch.float16, 
                                                    local_files_only = True
                                                    ).to(device)

    controlnet_canny_pipe = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
                                                                            controlnet=controlnet_canny,
                                                                                torch_dtype=torch.float16, 
                                                                                local_files_only = True
                                                                                ).to(device)


    controlnet_canny_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_canny_pipe.scheduler.config)
    # controlnet_canny_pipe.enable_model_cpu_offload()
    controlnet_canny_pipe.enable_xformers_memory_efficient_attention()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with autocast(device): 
        output = controlnet_canny_pipe(
            prompt,
            canny_image,
            num_images_per_prompt = 4,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps,
            height= height,
            width=width,
            guidance_scale=guidance_scale

    )

    dictobj = {}
    dictobj.update({
        "preImg": canny_string,
    })
    for num_img in range(4):
        gen_image = output.images[num_img]
        gen_image.save(f"./results/canny{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr


# def call_real_ESRGAN_model(image):
#     image = Image.open(BytesIO(image))
#     sr_image = real_ESRGAN_model.predict(image)
#     sr_image.save('results/upscaledv1.png')
#     buffer = BytesIO()
#     sr_image.save(buffer, format="PNG")
#     imgstr = base64.b64encode(buffer.getvalue())



def call_openpose_controlnet(image, 
                                    prompt, 
                                    seed, 
                                    negative_prompt, 
                                    height, 
                                    width, 
                                    guidance_scale, 
                                    steps):
    image = Image.open(BytesIO(image))
    pose = openpose_detector_model(image)
    pose.save('./images/generatedPose.png')
    buffer = BytesIO()
    pose.save(buffer, format="PNG")
    pose_base64 = base64.b64encode(buffer.getvalue())
    pose_string = pose_base64.decode('utf-8')


    openpose_detector_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


    openpose_controlnet = ControlNetModel.from_pretrained("./models/stable-diffusion-v1-5-controlnet-openpose",
                                                revision="fp32", 
                                                torch_dtype=torch.float16, 
                                                local_files_only = True
                                                ).to(device)

    openpose_controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
                                                                                controlnet=openpose_controlnet, 
                                                                                torch_dtype=torch.float16, 
                                                                                local_files_only = True
                                                                            ).to(device)
    

    generator = torch.Generator(device="cpu").manual_seed(seed)
    openpose_controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(openpose_controlnet_pipe.scheduler.config)
    openpose_controlnet_pipe.enable_xformers_memory_efficient_attention()

    with autocast(device): 
        output = openpose_controlnet_pipe(
            prompt,
            pose,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=steps,
            num_images_per_prompt = 4,
            height=height,
            width=width,
            guidance_scale=guidance_scale
        )

    dictobj = {}
    dictobj.update({
        "preImg": pose_string,
    })
    for num_img in range(4):
        gen_image = output.images[num_img]
        gen_image.save(f"./results/poseafterprocessing{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr



def call_controlet_shuffle(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    
    image = Image.open(BytesIO(image))
    prompt = prompt
    processor = ContentShuffleDetector()

    control_image = processor(image)
    control_image.save("./images/shuffle.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })

    controlnet_shuffle = ControlNetModel.from_pretrained("./models/control_v11e_sd15_shuffle", torch_dtype=torch.float16)
    pipe_shuffle = StableDiffusionControlNetPipeline.from_pretrained("./models/stable_diffusion_1_5",
                                                            controlnet=controlnet_shuffle, 
                                                            torch_dtype=torch.float16
                                                            ).to(device)

    pipe_shuffle.scheduler = UniPCMultistepScheduler.from_config(pipe_shuffle.scheduler.config)
    # pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(seed)
    image = pipe_shuffle(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=4,
                        num_inference_steps=steps,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )


    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/shuffled{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr


def call_controlet_scribble(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    prompt = prompt
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')

    control_image = processor(image, scribble=True)
    control_image.save("./images/scribble.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })

    controlnet_scribble = ControlNetModel.from_pretrained("./models/controlnet_scribble", torch_dtype=torch.float16)
    pipe_scribble = StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_1_5,
                                                        controlnet=controlnet_scribble,
                                                        torch_dtype=torch.float16
                                                        ).to(device)


    pipe_scribble.scheduler = UniPCMultistepScheduler.from_config(pipe_scribble.scheduler.config)
    # pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(seed)
    image = pipe_scribble(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        num_images_per_prompt=4,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )

    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/scribbled{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr


def call_controlet_hed(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    prompt = prompt
    hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

    control_image = hed(image)
    control_image.save("./images/hed.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })

    controlnet_hed = ControlNetModel.from_pretrained("./models/controlnet_hed", torch_dtype=torch.float16)
    pipe_hed = StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_1_5,
                                                            controlnet=controlnet_hed, 
                                                            torch_dtype=torch.float16
                                                            ).to(device)

    pipe_hed.scheduler = UniPCMultistepScheduler.from_config(pipe_hed.scheduler.config)
    # pipe.enable_model_cpu_offload()
    pipe_hed.enable_xformers_memory_efficient_attention()
    generator = torch.manual_seed(seed)
    image = pipe_hed(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=4,
                        num_inference_steps=steps,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )

    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/hedResult{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr

def call_controlet_lineart(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    # image = image.resize((512, 512))
    prompt = prompt
    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)
    control_image.save("./images/lineart.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })
    controlnet_lineart = ControlNetModel.from_pretrained("./models/controlnet_lineart", torch_dtype=torch.float16)
    pipe_lineart = StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_1_5,
                                                            controlnet=controlnet_lineart,
                                                            torch_dtype=torch.float16
                                                            ).to(device)
    

    pipe_lineart.scheduler = UniPCMultistepScheduler.from_config(pipe_lineart.scheduler.config)
    # pipe.enable_model_cpu_offload()
    pipe_lineart.enable_xformers_memory_efficient_attention()
    generator = torch.manual_seed(seed)
    image = pipe_lineart(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt = 4,
                        num_inference_steps=steps,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )

    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/lineartResult{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr


def call_controlet_softEdge(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    prompt = prompt
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')

    control_image = processor(image)
    control_image.save("./images/softEdge.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })

    controlnet_softEdge = ControlNetModel.from_pretrained("./models/controlnet_softEdge", torch_dtype=torch.float16)
    pipe_softEdge = StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_1_5,
                                                            controlnet=controlnet_softEdge,
                                                            torch_dtype=torch.float16
                                                            ).to(device)
    

    pipe_softEdge.scheduler = UniPCMultistepScheduler.from_config(pipe_softEdge.scheduler.config)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    generator = torch.manual_seed(seed)
    image = pipe_softEdge(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        num_images_per_prompt=4,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )

    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/softedgeResult{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr




def call_controlet_normalBae(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    prompt = prompt
    processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

    control_image = processor(image)
    control_image.save("./images/normalBae.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })

    controlnet_normalBae = ControlNetModel.from_pretrained("./models/controlnet_normalBae", torch_dtype=torch.float16)
    pipe_normalBae = StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_1_5,
                                                            controlnet=controlnet_normalBae,
                                                            torch_dtype=torch.float16
                                                            ).to(device)

    pipe_normalBae.scheduler = UniPCMultistepScheduler.from_config(pipe_normalBae.scheduler.config)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    generator = torch.manual_seed(seed)
    image = pipe_normalBae(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        num_images_per_prompt=4,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )

    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/normalBae{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr



def call_controlet_mlsd(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 guidance_scale, 
                                 steps):
    image = Image.open(BytesIO(image))
    prompt = prompt
    processor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

    control_image = processor(image)
    control_image.save("./images/mlsd.png")

    buffer = BytesIO()
    control_image.save(buffer, format="PNG")
    control_image_base64 = base64.b64encode(buffer.getvalue())
    control_image_string = control_image_base64.decode('utf-8')

    dictobj = {}
    dictobj.update({
        "preImg": control_image_string,
    })

    controlnet_mlsd = ControlNetModel.from_pretrained("./models/controlnet_mlsd", torch_dtype=torch.float16)
    pipe_mlsd = StableDiffusionControlNetPipeline.from_pretrained(stable_diffusion_1_5,
                                                            controlnet=controlnet_mlsd,
                                                            torch_dtype=torch.float16
                                                            ).to(device)
    

    pipe_mlsd.scheduler = UniPCMultistepScheduler.from_config(pipe_mlsd.scheduler.config)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    generator = torch.manual_seed(seed)
    image = pipe_mlsd(prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        num_images_per_prompt=4,
                        generator=generator,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale
                        )

    for num_img in range(4):
        gen_image = image.images[num_img]
        gen_image.save(f"./results/mlsd{num_img}.png")

        buffer = BytesIO()
        gen_image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue())
        base64_string = data.decode('utf-8')
        
        dictobj.update({
            f"img{num_img}": base64_string,
        })

    imgstr = json.dumps(dictobj)
    return imgstr
