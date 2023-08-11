from share import *
import config

import cv2
import einops
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldmXL.model import create_model, load_state_dict
import cv2

from scripts.streamlit_helpers import *

SAVE_PATH = "outputs/demo/txt2img/"

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
}


if __name__ == "__main__":

    model = create_model('./models/cldm_xl.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sdxl_ini.ckpt', location='cuda'))
    model = model.cuda()  # if you do not have enough V memory, you can comment this line load container, Unet, control_model step by step

    is_legacy = False
    W = 1024
    H = 1024
    C = 4
    F = 8

    seed_everything(42)

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }

    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        prompt="cute dog",
        negative_prompt="",
    )

    sampler, num_rows, num_cols = init_sampling(stage2strength=None)
    model.sampler = sampler

    num_samples = num_rows * num_cols

    apply_hed = HEDdetector()
    detect_map = cv2.imread("/home/happen/Downloads/drawingboard_M1690594457616.png")
    detect_map = cv2.resize(detect_map, (W, H))
    detect_map = apply_hed(detect_map)
    detect_map = HWC3(detect_map)
    control = torch.from_numpy(detect_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    hint = einops.rearrange(control, 'b h w c -> b c h w').clone()



    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        [num_samples],
    )

    force_uc_zero_embeddings = ["txt"] if not is_legacy else []


    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=force_uc_zero_embeddings,
    )

    del model.conditioner

    for k in c:
        if not k == "crossattn":
            print("ks is not crossattn")
            c[k], uc[k] = map(
                lambda y: y[k][: math.prod([num_samples])].to("cuda"), (c, uc)
            )

    shape = (math.prod([num_samples]), C, H // F, W // F)

    with torch.no_grad():
            with model.ema_scope():
                samples = model.sample(c, uc, batch_size= 1, hint = hint, shape = shape)
                del model.control_model
                samples_copy = samples.clone()
                samples_copy = einops.rearrange(samples_copy, 'b c h w -> b h w c')
                samples_copy = samples_copy.cpu().numpy()
                samples_copy = samples_copy * 255.0
                samples_copy = samples_copy.astype(np.uint8)
                cv2.imwrite("test_out.png", samples_copy[0])
                samples = model.decode_first_stage(samples)

    samples = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)

    for sample in samples:
        sample = 255.0 * einops.rearrange(sample.cpu().numpy(), "c h w -> h w c")
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        cv2.imwrite("test.png", sample)





