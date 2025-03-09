export PATH=$HOME/.local/bin:$PATH
source ~/.profile

bash install_deps_cu124.sh

source .venv/bin/activate

wget -O PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

apt update && apt install -y tmux

tmux new -s miner 'python3 /workspace/nova/neurons/miner.py --wallet.name miner --wallet.hotkey default --network ws://178.63.86.244:9944 --logging.debug 2>&1 | tee miner.log'


python3 neurons/miner.py --wallet.name miner --wallet.hotkey default --logging.debug --network ws://178.63.86.244:9944


GITHUB TOKEN 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