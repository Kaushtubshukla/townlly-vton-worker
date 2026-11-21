FROM runpod/worker-comfyui:5.0.0-flux1-dev

USER root

# 1. Upgrade transformers and install PyTorch/HuggingFace dependencies
RUN pip install --no-cache-dir --upgrade "transformers>=4.49.0" "tokenizers>=0.21.0" "accelerate>=1.2.0" safetensors

# 2. Clone and install official BFL flux-3-image-klein
RUN git clone --depth 1 https://github.com/black-forest-labs/flux-3-image-klein.git /comfyui/flux-3-image-klein && \
    pip install --no-cache-dir -e /comfyui/flux-3-image-klein

# 3. Create required directories
RUN mkdir -p /comfyui/custom_nodes/flux3_vton /workspace/weights /comfyui/weights

# 4. Install production VTON custom node
COPY custom_nodes/flux3_vton/__init__.py /comfyui/custom_nodes/flux3_vton/__init__.py

# 5. Set default RunPod serverless handler entrypoint
CMD ["/start.sh"]
