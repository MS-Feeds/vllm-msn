# 1. Base Image: Pre-loaded with perfectly matched PyTorch, vLLM, and Flash Attention
FROM vllm/vllm-openai:gemma4

ARG DEBIAN_FRONTEND=noninteractive

# ---- 2. Safe OS Environment Variables ----
ENV NCCL_DEBUG="INFO"
ENV NCCL_SOCKET_IFNAME="eth0"
ENV NCCL_IB_PCI_RELAXED_ORDERING="1"
ENV GLOO_SOCKET_IFNAME="eth0"
ENV MASTER_ADDR="node-0"
ENV MASTER_PORT="9500"
ENV LANG="en_US.UTF-8"
ENV LC_ALL="en_US.UTF-8"
ENV TZ="US/Pacific"

# ---- 3. Azure ML Specific Tools & Multimodal OS Deps ----
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    openssh-server \
    openssh-client \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir jupyter ipykernel pandas && \
    mkdir -p /var/run/sshd && \
    python3 -m ipykernel install --user

# ---- 4. Setup Azure Workspace ----
WORKDIR /workspace
COPY run.gemma4.sh /workspace/run.gemma4.sh
RUN chmod +x /workspace/run.gemma4.sh

COPY run.gemma4.quant.sh /workspace/run.gemma4.quant.sh
RUN chmod +x /workspace/run.gemma4.quant.sh

COPY run.qwen36.sh /workspace/run.qwen36.sh
RUN chmod +x /workspace/run.qwen36.sh

COPY run.gemma4.full.sh /workspace/run.gemma4.full.sh
RUN chmod +x /workspace/run.gemma4.full.sh

COPY tool_chat_template_gemma4.jinja /workspace/tool_chat_template_gemma4.jinja
RUN chmod +x /workspace/tool_chat_template_gemma4.jinja

EXPOSE 8000 6066 8080 5001 8883 8889

CMD ["bash"]