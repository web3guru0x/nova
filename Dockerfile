# Imagine de bază cu Ubuntu și CUDA 12.4
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Setăm variabila pentru CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Instalăm pachetele necesare
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    tzdata \
    && ln -fs /usr/share/zoneinfo/Europe/Bucharest /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    curl git wget unzip vim \
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

# Copiem fișierul .env
COPY .env /workspace/nova/.env

# Instalăm dependențele Nova (inclusiv PyTorch și Bittensor)
RUN cd /workspace/nova && export PATH=$HOME/.local/bin:$PATH && bash install_deps_cu124.sh || echo "Install script failed, continuing..."

# Setăm punctul de intrare
ENTRYPOINT ["/bin/bash", "-c", "tail -f /dev/null"]
CMD ["/bin/bash"]


