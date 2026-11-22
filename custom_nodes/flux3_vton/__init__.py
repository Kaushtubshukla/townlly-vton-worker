import os
import sys
import math
import numpy as np
import torch
from PIL import Image

# 1. Dynamic WEIGHTS_DIR resolution
candidates_weights = [
    "/runpod-volume/weights",
    "/workspace/weights",
    "/comfyui/weights"
]
for c in candidates_weights:
    if os.path.exists(c):
        os.environ['WEIGHTS_DIR'] = c
        break
if 'WEIGHTS_DIR' not in os.environ:
    os.environ['WEIGHTS_DIR'] = '/workspace/weights'

# 2. Add BFL Klein source to sys.path
for candidate_src in [
    "/comfyui/flux-3-image-klein/src",
    "/workspace/flux-3-image-klein/src",
    "/runpod-volume/flux-3-image-klein/src"
]:
    if os.path.exists(candidate_src) and candidate_src not in sys.path:
        sys.path.insert(0, candidate_src)

# 3. Patch transformers for Qwen3-VL Polyfills
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
    print('[BFL Node] Successfully registered Qwen3VLConfig & Qwen3VLForConditionalGeneration in transformers!')
except Exception as e:
    print('[BFL Node] Qwen3-VL registration error:', e)

# 4. Safe Processor Wrapper for BFL text_encoder.py
class _SafeProcessorWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

# Apply runtime patch to flux3 text_encoder if present
try:
    import flux3.text_encoder
    orig_te_init = flux3.text_encoder.TextEncoder.__init__
    def safe_te_init(self, path: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForVision2Seq
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception:
            Qwen3VLForConditionalGeneration = AutoModelForVision2Seq
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16
        ).to(device).eval()
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = "right"
        self.processor = _SafeProcessorWrapper(tok)
        self.device = device
    flux3.text_encoder.TextEncoder.__init__ = safe_te_init
    print('[BFL Node] Runtime patched flux3.text_encoder.TextEncoder successfully!')
except Exception as e:
    print('[BFL Node] Note on runtime patching text_encoder:', e)

_CACHED_MODELS = None
def get_models():
    global _CACHED_MODELS
    if _CACHED_MODELS is None:
        # Determine actual weights directory
        w_dir = "/workspace/weights"
        for cand in ["/runpod-volume/weights", "/workspace/weights", "/comfyui/weights"]:
            if os.path.exists(cand):
                w_dir = cand
                os.environ['WEIGHTS_DIR'] = cand
                break
        os.makedirs(w_dir, exist_ok=True)

        dit = os.path.join(w_dir, 'flux-3-image-klein')

        # Check if essential weights are present on disk
        has_weights = (
            os.path.exists(os.path.join(dit, 'model.safetensors')) and
            os.path.exists(os.path.join(dit, 'text_encoder', 'model.safetensors.index.json'))
        )
        if not has_weights:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print(f"[BFL Node] Weights missing at {dit}. Initiating fast multi-threaded download via hf_transfer...")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="black-forest-labs/flux-3-image-klein",
                local_dir=dit,
                token=hf_token,
                allow_patterns=["model.safetensors", "ae.safetensors", "text_encoder/*", "config.json"]
            )
            print(f"[BFL Node] Fast download complete! Weights verified at {dit}.")
        else:
            print(f"[BFL Node] Verified existing weights found on disk at {dit}.")

        # Patch flux3.checkpoint flow-map and key inspection
        try:
            import flux3.checkpoint
            orig_is_flow_map = getattr(flux3.checkpoint, 'is_flow_map', None)
            def safe_is_flow_map(path):
                try:
                    if orig_is_flow_map:
                        return orig_is_flow_map(path)
                except Exception:
                    pass
                st_path = os.path.join(str(path), 'model.safetensors') if os.path.isdir(str(path)) else str(path)
                if os.path.exists(st_path):
                    try:
                        from safetensors import safe_open
                        with safe_open(st_path, framework='pt', device='cpu') as f:
                            flow_key = getattr(flux3.checkpoint, 'FLOW_MAP_KEY', 'flow_map')
                            return flow_key in {k.removeprefix('model.') for k in f.keys()}
                    except Exception:
                        pass
                return False
            flux3.checkpoint.is_flow_map = safe_is_flow_map

            orig_checkpoint_keys = getattr(flux3.checkpoint, 'checkpoint_keys', None)
            def safe_checkpoint_keys(path):
                try:
                    if orig_checkpoint_keys:
                        return orig_checkpoint_keys(path)
                except Exception:
                    pass
                st_path = os.path.join(str(path), 'model.safetensors') if os.path.isdir(str(path)) else str(path)
                if os.path.exists(st_path):
                    try:
                        from safetensors import safe_open
                        with safe_open(st_path, framework='pt', device='cpu') as f:
                            return {k.removeprefix('model.') for k in f.keys()}
                    except Exception:
                        pass
                return set()
            flux3.checkpoint.checkpoint_keys = safe_checkpoint_keys
            print('[BFL Node] Patched flux3.checkpoint safely!')
        except Exception as e:
            print('[BFL Node] Checkpoint patch warning:', e)

        from flux3.generate import Models
        print(f'[BFL Node] Initializing Models with dit={dit} (WEIGHTS_DIR={os.environ.get("WEIGHTS_DIR")})...')
        _CACHED_MODELS = Models({'distilled': dit}, None, 0.35)
        print('[BFL Node] Models loaded successfully into GPU VRAM!')
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
            'required': {
                'model_image': ('IMAGE',),
                'garment_image': ('IMAGE',),
                'prompt': ('STRING', {'multiline': True, 'default': ''}),
                'steps': ('INT', {'default': 8, 'min': 4, 'max': 50, 'step': 1}),
                'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('image',)
    FUNCTION = 'generate_vton'
    CATEGORY = 'FLUX3_Native'

    def generate_vton(self, model_image, garment_image, prompt, steps, seed):
        try:
            from flux3.generate import generate
            models = get_models()
            p_pil = tensor_to_pil(model_image)
            g_pil = tensor_to_pil(garment_image)
            images = [p_pil, g_pil]
            print(f'[BFL Node] Starting generate: steps={steps}, seed={seed}...')
            image, _ = generate(
                models=models, model='distilled', prompt=prompt,
                images=images, boxes=None, width=None, height=None,
                steps=steps, guidance=None, seed=seed, pup=False, safety=False
            )
            print('[BFL Node] Generation completed successfully!')
            return (pil_to_tensor(image),)
        except Exception as e:
            print('[BFL Node] Generation error:', e)
            import traceback
            traceback.print_exc()
            return (model_image,)

NODE_CLASS_MAPPINGS = {'BFLFlux3NativeVTON': BFLFlux3NativeVTON}
NODE_DISPLAY_NAME_MAPPINGS = {'BFLFlux3NativeVTON': 'FLUX 3 Klein Native VTON (Official BFL)'}
