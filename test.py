from t2i_unet import T2IAdapterUNet2DConditionModel, Adapter, sketch_extracter
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image
import numpy as np


if __name__ == "__main__":
    
    # 1. Define Adapter feature extractor
    adapter = Adapter.from_pretrained("sketch").to("cuda:1")
    
    # 2. Prepare Condition via adapter.
    edge = np.array(Image.open("./contents/dog_edge.png").resize((512, 512)).convert("L"))/ 255.0
    edge = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to("cuda:1")
    edge = (edge > 0.5).float()
    with torch.no_grad():
        adapter_features = adapter(edge)
    
    
    
    model_id = "runwayml/stable-diffusion-v1-5"

    a_unet = T2IAdapterUNet2DConditionModel.from_config(
        {
            "act_fn": "silu",
            "attention_head_dim": 8,
            "block_out_channels": [320, 640, 1280, 1280],
            "center_input_sample": False,
            "cross_attention_dim": 768,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ],
            "downsample_padding": 1,
            "dual_cross_attention": False,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "out_channels": 4,
            "sample_size": 64,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ],
            "upcast_attention": False,
            "use_linear_projection": False,
        }
    )
    a_unet.to("cuda:1").to(torch.float16)
    
    
    a_unet.set_adapter_features(adapter_features)
    
    unet2 = UNet2DConditionModel.from_pretrained(model_id, subfolder = "unet")
    
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

    