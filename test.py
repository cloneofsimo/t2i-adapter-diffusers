from t2i_adapters import T2IAdapterUNet2DConditionModel, Adapter, sketch_extracter
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import numpy as np


if __name__ == "__main__":
    
    # 1. Define Adapter feature extractor
    adapter = Adapter.from_pretrained("sketch").to("cuda:1")
    
    # 2. Prepare Condition via adapter.
    edge = np.array(Image.open("./contents/examples/sketch_0.png").resize((512, 512)).convert("L"))/ 255.0
    edge = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to("cuda:1")
    edge = (edge > 0.5).float()
    with torch.no_grad():
        adapter_features = adapter(edge)
    
    
    
    model_id = "runwayml/stable-diffusion-v1-5"
    unet2 = UNet2DConditionModel.from_pretrained(model_id, subfolder = "unet")
    
    a_unet = T2IAdapterUNet2DConditionModel.from_config(
        unet2.config
    )
    a_unet.to("cuda:1").to(torch.float16)
    
    
    a_unet.set_adapter_features(adapter_features)
    
    
    inf = a_unet.load_state_dict(unet2.state_dict(), strict = False)
    a_unet.to("cuda:1").to(torch.float16)
    print(f"DONE : info : {inf}")
    
        
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        "cuda:1"
    )

    pipe.unet = a_unet
    
    
    
    prompt = "cute robot owl in style of van gogh"
    pipe.safety_checker = None
    neg_prompt = "out of frame, duplicate, watermark "
    torch.manual_seed(1)
    n = 1
    imgs = pipe(
        [prompt] * n,
        negative_prompt=[neg_prompt] * n,
        num_inference_steps=50,
        guidance_scale=8.5,
        height=512,
        width=512,
    ).images

    imgs[0].save("./contents/owl_output.png")

    