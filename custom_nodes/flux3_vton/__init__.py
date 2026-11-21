import os
import sys
import numpy as np
import torch
from PIL import Image

# Search paths for official BFL Klein modules
for candidate in [
    "/comfyui/flux-3-image-klein/src",
    "/workspace/flux-3-image-klein/src",
    "/runpod-volume/flux-3-image-klein/src"
]:
    if os.path.exists(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    import transformers
    from transformers import AutoConfig, AutoModelForVision2Seq
    from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    class Qwen3VLConfig(Qwen2_5_VLConfig):
        model_type = 'qwen3_vl'
        def __init__(self, **kwargs):
            text_cfg = kwargs.get('text_config', {})
            if isinstance(text_cfg, dict) and text_cfg:
                try:
                    kwargs['text_config'] = Qwen2Config(**text_cfg)
                except Exception:
                    pass
            super().__init__(**kwargs)

    class Qwen3VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
        config_class = Qwen3VLConfig
        def can_generate(self):
            return False

    from transformers.generation.configuration_utils import GenerationConfig
    orig_from_model_config = getattr(GenerationConfig, 'from_model_config', None)
    def safe_from_model_config(model_config, **kwargs):
        try:
            if orig_from_model_config:
                return orig_from_model_config(model_config, **kwargs)
        except Exception:
            pass
        return GenerationConfig()
    GenerationConfig.from_model_config = safe_from_model_config

    AutoConfig.register('qwen3_vl', Qwen3VLConfig)
    AutoModelForVision2Seq.register(Qwen3VLConfig, Qwen3VLForConditionalGeneration)
    transformers.Qwen3VLConfig = Qwen3VLConfig
    transformers.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    sys.modules['transformers'].Qwen3VLConfig = Qwen3VLConfig
    sys.modules['transformers'].Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    print('[BFL Node] Successfully registered Qwen3VL polyfills in transformers!')
except Exception as e:
    print('[BFL Node] Qwen3-VL registration error:', e)

_CACHED_MODELS = None
def get_models():
    global _CACHED_MODELS
    if _CACHED_MODELS is None:
        dit = None
        candidates = [
            "/comfyui/models/unet",
            "/comfyui/models/checkpoints",
            "/runpod-volume/weights/flux-3-image-klein",
            "/runpod-volume/weights",
            "/workspace/weights/flux-3-image-klein",
            "/workspace/weights",
            "/comfyui/weights"
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                for fname in ["model.safetensors", "flux1-dev.safetensors", "flux1-schnell.safetensors"]:
                    if os.path.exists(os.path.join(candidate, fname)):
                        dit = candidate
                        break
            if dit:
                break
        
        if dit is None:
            try:
                from flux3.pipeline import dit_path
                dit = str(dit_path("distilled"))
            except Exception as e:
                print("[BFL Node] dit_path fallback:", e)
                dit = "/comfyui/models/unet"
            
        print(f"[BFL Node] Initializing Models with dit={dit}...")
        try:
            from flux3.generate import Models
            _CACHED_MODELS = Models({"distilled": dit}, None, 0.35)
            print("[BFL Node] Models loaded successfully into GPU VRAM!")
        except Exception as e:
            print("[BFL Node] Failed to instantiate Models:", e)
            raise e
    return _CACHED_MODELS

def tensor_to_pil(t):
    arr = np.clip(255.0 * t[0].cpu().numpy(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil_to_tensor(p):
    arr = np.array(p).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

class BFLFlux3NativeVTON:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 8, "min": 4, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_vton"
    CATEGORY = "FLUX3_Native"

    def generate_vton(self, model_image, garment_image, prompt, steps, seed):
        try:
            from flux3.generate import generate
            models = get_models()
            p_pil = tensor_to_pil(model_image)
            g_pil = tensor_to_pil(garment_image)
            print(f"[BFL Node] Starting generate: steps={steps}, seed={seed}...")
            image, _ = generate(
                models=models, model="distilled", prompt=prompt,
                images=[p_pil, g_pil], boxes=None, width=None, height=None,
                steps=steps, guidance=None, seed=seed, pup=False, safety=False
            )
            print("[BFL Node] Generation completed successfully!")
            return (pil_to_tensor(image),)
        except Exception as e:
            print("[BFL Node] Error during generation:", e)
            import traceback
            traceback.print_exc()
            return (model_image,)

NODE_CLASS_MAPPINGS = {"BFLFlux3NativeVTON": BFLFlux3NativeVTON}
NODE_DISPLAY_NAME_MAPPINGS = {"BFLFlux3NativeVTON": "FLUX 3 Klein Native VTON (Official BFL)"}
