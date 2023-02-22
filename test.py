from t2i_unet import T2IAdapterUNet2DConditionModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch


if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"

    x = T2IAdapterUNet2DConditionModel.from_config(
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
    x.to("cuda:1").to(torch.float16)
    unet2 = UNet2DConditionModel.from_pretrained(model_id, subfolder = "unet")

    inf = x.load_state_dict(unet2.state_dict(), strict = False)

    print(f"DONE : info : {inf}")
    
        
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        "cuda:1"
    )

    pipe.unet = x
    prompt = "cute doggo"
    pipe.safety_checker = None
    neg_prompt = "out of frame, duplicate, watermark "
    torch.manual_seed(2)
    n = 1
    imgs = pipe(
        [prompt] * n,
        negative_prompt=[neg_prompt] * n,
        num_inference_steps=50,
        guidance_scale=4.5,
        height=512,
        width=512,
    ).images

    imgs[0].save("./contents/test.png")

    