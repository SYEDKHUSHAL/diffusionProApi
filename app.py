from fastapi import FastAPI, Response, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import torch

from src.model_calls import *







app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)



@app.get("/waifu")
def generate(prompt:str,
                negative_prompt: str="", 
                guidance_scale: str="8.5",
                seed: int=0,
                height: int=None,
                width: int=None, 
                steps: int=30): 
    imgstr = call_waifu_diffusion(prompt, 
                                    seed, 
                                    negative_prompt, 
                                    height, 
                                    width, 
                                    float(guidance_scale), 
                                    steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")


@app.get("/generate_1_5")
async def generate_1_5(prompt:str,
                        negative_prompt: str="", 
                        guidance_scale: str="8.5",
                        seed: int=0,
                        height: int=None,
                        width: int=None, 
                        steps: int=30): 
    data = await call_stable_diffusion_1_5(prompt, 
                                            seed, 
                                            negative_prompt, 
                                            height, 
                                            width, 
                                            float(guidance_scale), 
                                            steps)
    torch.cuda.empty_cache()
    return Response(content=data, media_type="application/json")



@app.get("/text2img_v2")
def text2img_v2(prompt:str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30): 
    imgstr = call_stable_diffusion_2_1(prompt, 
                                    seed, 
                                    negative_prompt, 
                                    height, 
                                    width, 
                                    float(guidance_scale), 
                                    steps
                                    )
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")


@app.post("/img2img_v2")
def img2img_v2(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    strength: str="0.8",
                    seed: int=0,
                    steps: int=30, 
                    file: UploadFile= File(...)): 
    image = file.file.read()
    imgstr = call_stable_diffusion_2_1_img2img(image, 
                                    prompt, 
                                    seed, 
                                    negative_prompt, 
                                    float(guidance_scale), 
                                    float(strength),
                                    steps
                                    )
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")


@app.get("/text2img_LinaQurf")
def text2img_LinaQurf(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30): 
    
    imgstr = call_lina_qurf(prompt, 
                            seed, 
                            negative_prompt, 
                            height, 
                            width, 
                            float(guidance_scale), 
                            steps
                            )
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")


@app.post("/img2imgLinaQurf")
def img2imgLinaQurf(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    strength: str="0.8",
                    seed: int=0,
                    steps: int=30, 
                    file: UploadFile= File(...)): 
    image = file.file.read()
    imgstr = call_lina_qurf_img2img(image,
                            prompt, 
                            seed, 
                            negative_prompt, 
                            float(guidance_scale), 
                            float(strength),
                            steps
                            )
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")



#controlnets

# @app.post("/upsample")
# def upsample(file: UploadFile = File(...)): 
#     image = file.file.read()
#     imgstr = call_real_ESRGAN_model(image)
    
#     return Response(content=imgstr, media_type="application/json")

@app.post("/canny_img")
def canny_img(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)): 
    image = file.file.read()
    imgstr = call_controlnet_canny(image, 
                                 prompt, 
                                 seed, 
                                 negative_prompt, 
                                 height, 
                                 width, 
                                 float(guidance_scale), 
                                 steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")


@app.post("/create_from_raw_pose")
def create_from_raw_pose(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)): 
    image = file.file.read()
    imgstr = call_openpose_controlnet(image, 
                                    prompt, 
                                    seed, 
                                    negative_prompt, 
                                    height, 
                                    width, 
                                    float(guidance_scale), 
                                    steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")

@app.post("/shuffle")
def shuffle(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):  

    image = file.file.read()
    imgstr = call_controlet_shuffle(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")

@app.post("/scribble")
def scribble(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):   
    
    image = file.file.read()
    imgstr = call_controlet_scribble(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")


@app.post("/hed")
def hed(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):  

    image = file.file.read()
    imgstr = call_controlet_hed(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")

@app.post("/lineart")
def lineart(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):  

    image = file.file.read()
    imgstr = call_controlet_lineart(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")

@app.post("/softedge")
def softedge(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):  

    image = file.file.read()
    imgstr = call_controlet_softEdge(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")

@app.post("/normal_bae")
def normal_bae(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=33,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):  

    image = file.file.read()
    imgstr = call_controlet_normalBae(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")

@app.post("/mlsd")
def mlsd(prompt: str,
                    negative_prompt: str="", 
                    guidance_scale: str="8.5",
                    seed: int=0,
                    height: int=None,
                    width: int=None, 
                    steps: int=30, 
                    file: UploadFile= File(...)):  

    image = file.file.read()
    imgstr = call_controlet_mlsd(image, 
                                          prompt, 
                                          seed, 
                                          negative_prompt, 
                                          height, 
                                          width, 
                                          float(guidance_scale), 
                                          steps)
    torch.cuda.empty_cache()
    return Response(content=imgstr, media_type="application/json")



