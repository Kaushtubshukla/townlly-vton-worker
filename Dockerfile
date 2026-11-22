FROM runpod/worker-comfyui:5.0.0-flux1-dev

USER root

# 1. Install transformers and hf_transfer WITHOUT upgrading or touching base PyTorch/CUDA
RUN pip install --no-cache-dir --no-deps "transformers>=4.49.0" "hf_transfer>=0.1.9" "tokenizers>=0.21.0" "accelerate>=1.2.0" safetensors

# 2. Clone and install official BFL flux-3-image-klein in editable mode without touching PyTorch
RUN git clone --depth 1 https://github.com/black-forest-labs/flux-3-image-klein.git /comfyui/flux-3-image-klein && \
    pip install --no-cache-dir --no-deps -e /comfyui/flux-3-image-klein

# 3. Create required directories
RUN mkdir -p /comfyui/custom_nodes/flux3_vton /workspace/weights /comfyui/weights

# 4. Install production VTON custom node with complete polyfills
COPY custom_nodes/flux3_vton/__init__.py /comfyui/custom_nodes/flux3_vton/__init__.py

# 5. Sanity check: verify torch, transformers, and hf_transfer are all intact and compatible
RUN python3 -c "import torch; print('PyTorch build:', torch.__version__, 'CUDA:', torch.version.cuda); import transformers; print('Transformers:', transformers.__version__); import hf_transfer; print('hf_transfer:', hf_transfer.__version__); import flux3; print('flux3 package importable!')"

# 6. Set default RunPod serverless handler entrypoint
CMD ["/start.sh"]
