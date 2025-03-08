import os
import math
import random
import argparse
import asyncio
from typing import cast
from types import SimpleNamespace
import sys

import torch
import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper

class Miner:
    def __init__(self):
        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'

        # 🔹 Optimizare memorie VRAM pentru H200
        self.total_mem = torch.cuda.get_device_properties(0).total_memory // 1e9  # în GB
        self.chunk_size = int(4096 + (self.total_mem - 80) * 40)  # Ajustează chunk size
        self.batch_size = int(1024 + (self.total_mem - 80) * 10)  # Ajustează batch size
        self.tolerance = 3

        self.config = self.get_config()
        node = SubstrateInterface(url=self.config.network)
        self.epoch_length = node.query("SubtensorModule", "Tempo", [self.config.netuid]).value
        self.setup_logging()
        self.current_block = 0
        self.current_challenge_protein = None
        self.last_challenge_protein = None
        self.psichic_wrapper = PsichicWrapper()
        self.candidate_product = None
        self.candidate_product_score = 0
        self.best_score = 0
        self.last_submitted_product = None
        self.shared_lock = asyncio.Lock()
        self.inference_task = None
        self.shutdown_event = asyncio.Event()

        # 🔹 Optimizare Tensor Cores și CUDA
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--network', default='wss://archive.chain.opentensor.ai:443', help='Network to use')
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'miner',
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network}")

    async def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(config=self.config)
        async with bt.async_subtensor(network=self.config.network) as subtensor:
            self.subtensor = subtensor
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            await self.metagraph.sync()

    def stream_random_chunk_from_dataset(self):
        files = list_repo_files(self.hugging_face_dataset_repo, repo_type='dataset')
        files = [file for file in files if file.endswith('.csv')]
        random_file = random.choice(files)
        dataset_dict = load_dataset(self.hugging_face_dataset_repo,
                                    data_files={'train': random_file},
                                    streaming=True)
        dataset = dataset_dict['train']
        return dataset.batch(self.chunk_size)

    async def run_psichic_model_loop(self):
        dataset = self.stream_random_chunk_from_dataset()
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    df = pd.DataFrame.from_dict(chunk)
                    df['product_smiles'] = df['product_smiles'].apply(lambda x: x.replace('"', ''))
                    bt.logging.debug(f'Running inference...')
                    
                    # 🔹 Optimizează inferența cu FP8 pe H200
                    input_data = torch.tensor(df['product_smiles'].tolist()).to(torch.float8_e4m3fn)

                    # 🔹 Folosește CUDA Graphs pentru execuție rapidă
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        chunk_psichic_scores = self.psichic_wrapper.run_validation(input_data)

                    chunk_psichic_scores = chunk_psichic_scores.sort_values(by=self.psichic_result_column_name, ascending=False).reset_index(drop=True)
                    if chunk_psichic_scores[self.psichic_result_column_name].iloc[0] > self.best_score:
                        async with self.shared_lock:
                            candidate_molecule = chunk_psichic_scores['Ligand'].iloc[0]
                            self.best_score = chunk_psichic_scores[self.psichic_result_column_name].iloc[0]
                            self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]
                            bt.logging.info(f"New best score: {self.best_score}, New candidate product: {self.candidate_product}")
                        await asyncio.sleep(1)
                    await asyncio.sleep(3)

            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
                self.shutdown_event.set()

    async def run(self):
        bt.logging.info("Starting miner loop.")
        await self.setup_bittensor_objects()
        self.inference_task = asyncio.create_task(self.run_psichic_model_loop())

        while True:
            try:
                await asyncio.sleep(1)

            except RuntimeError as e:
                bt.logging.error(e)

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting miner.")
                exit()

if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())
