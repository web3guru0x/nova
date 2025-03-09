# Imagine de bază cu Ubuntu și CUDA 12.4, optimizată pentru runtime
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Setăm variabilele NVIDIA pentru performanță maximă
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Activează optimizări NVIDIA pentru compilare JIT pe GPU
ENV NVIDIA_NVJITLINK_OVERRIDE=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_SHM_DISABLE=1
ENV NCCL_DEBUG=INFO

# Instalăm pachetele necesare
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y \
    software-properties-common \
    nano \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Bucharest /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    curl git wget unzip vim nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setăm Python implicit la 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && ln -sf /usr/bin/python3.12 /usr/bin/python

# Instalăm pip și dependențele Python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Creăm un director de lucru
WORKDIR /workspace

# Clonăm codul sursă Nova
RUN git clone https://github.com/metanova-labs/nova.git && cd nova

# Setăm un `.env` default pentru siguranță
RUN echo "WALLET_NAME=main\nWALLET_HOTKEY=default\nSUBTENSOR_NETWORK=finney\nRUN_MODE=miner\nDEVICE_OVERRIDE=cuda" > /workspace/nova/.env

# Instalăm dependențele Nova (inclusiv PyTorch și Bittensor)
RUN cd /workspace/nova && export PATH=$HOME/.local/bin:$PATH && bash install_deps_cu124.sh || echo "Install script failed, continuing..."

# Setăm un proces activ pentru a preveni shutdown instant
ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]

# CMD implicit pentru debugging
CMD ["nvidia-smi", "/bin/bash"]
