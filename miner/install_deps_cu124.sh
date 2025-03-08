# install uv:
wget -qO- https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv && source .venv/bin/activate \
	&& uv pip install -r requirements/requirements_cu124.txt \
	&& uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
	&& uv pip install torch-geometric==2.6.1 \
	&& uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html