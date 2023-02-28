from t2i_adapters import patch_pipe, Adapter
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import numpy as np


if __name__ == "__main__":
    device = "cuda:0"
    
     # 0. Define model
    model_id = "runwayml/stable-diffusion-v1-5"
   
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        device
    )
    patch_pipe(pipe)
    # 1. Define Adapter feature extractor
    
    for ext_type, prompt in [("depth", "antique house"), ("seg", "motorcycle"), ("keypose", "elon musk"), ("sketch", "robot owl")]:
        adapter = Adapter.from_pretrained(ext_type).to(device)
        
        # 2. Prepare Condition via adapter.
        cond_img = Image.open(f"./contents/examples/{ext_type}_0.png")
                        
        if ext_type == "sketch":
            cond_img = cond_img.convert("L")
            cond_img = np.array(cond_img) / 255.0
            cond_img = torch.from_numpy(cond_img).unsqueeze(0).unsqueeze(0).to(device)
            cond_img = (cond_img > 0.5).float()
            
        else:
            cond_img = cond_img.convert("RGB")
            cond_img = np.array(cond_img) / 255.0
            
            cond_img = torch.from_numpy(cond_img).permute(2, 0, 1).unsqueeze(0).to(device).float()
            
        with torch.no_grad():
            adapter_features = adapter(cond_img)
            
        pipe.unet.set_adapter_features(adapter_features)
    
        pipe.safety_checker = None
        neg_prompt = "out of frame, duplicate, watermark "
        torch.manual_seed(1)
        n = 1
        imgs = pipe(
            [prompt] * n,
            negative_prompt=[neg_prompt] * n,
            num_inference_steps=50,
            guidance_scale=8.5,
            height=cond_img.shape[2],
            width=cond_img.shape[3],
        ).images

        imgs[0].save(f"./contents/{ext_type}_out.png")

        