from t2i_adapters import patch_pipe, Adapter, sketch_extracter
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import numpy as np


if __name__ == "__main__":
    device = "cuda:0"
    
     # 0. Define model
    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "Linaqruf/anything-v3.0"
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        device
    )
    
    patch_pipe(pipe)
    from lora_diffusion import LoRAManager
    manager = LoRAManager(["./contents/lora_krk.safetensors"], pipe)
    # 1. Define Adapter feature extractor
    manager.tune([.9])
    for ext_type, prompt in [("keypose", "a photo of <s0-0><s0-1> sitting down")]:
        adapter = Adapter.from_pretrained(ext_type).to(device)
        
        # 2. Prepare Condition via adapter.
        cond_img = Image.open(f"./contents/examples/{ext_type}_1.png")
                        
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
            guidance_scale=4.5,
            height=cond_img.shape[2],
            width=cond_img.shape[3],
        ).images

        imgs[0].save(f"./contents/{ext_type}_out_lora.png")

        