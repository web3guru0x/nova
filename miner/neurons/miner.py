import os
import math
import random
import argparse
import asyncio
from typing import cast
from types import SimpleNamespace
import sys
import torch
import traceback

import bittensor as bt
from bittensor.core.chain_data.utils import decode_metadata
from bittensor.core.errors import MetadataError
from substrateinterface import SubstrateInterface
from datasets import load_dataset
from huggingface_hub import list_repo_files
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Import our GPU utilities
from gpu_utils import setup_gpu_for_h200, empty_gpu_cache, gpu_timer, get_gpu_memory_usage, optimize_memory_for_inference
from my_utils import get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper

class Miner:
    def __init__(self):
        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'
        self.chunk_size = 256  # Increased chunk size for GPU processing
        self.tolerance = 3

        # Setup GPU optimizations for H200 SXM
        if not setup_gpu_for_h200():
            bt.logging.error("GPU setup failed - check your CUDA installation")
            sys.exit(1)
            
        bt.logging.info(f"GPU memory before initialization: {get_gpu_memory_usage()}")
        
        # Optimize memory allocation for inference workloads
        optimize_memory_for_inference(max_batch_size=self.chunk_size)
        
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
        
        # Log GPU information
        bt.logging.info(f"GPU memory after initialization: {get_gpu_memory_usage()}")

    def get_config(self):
        # Set up the configuration parser.
        parser = argparse.ArgumentParser()
        # Adds override arguments for network.
        parser.add_argument('--network', default='wss://archive.chain.opentensor.ai:443', help='Network to use')
        # Adds override arguments for network and netuid.
        parser.add_argument('--netuid', type=int, default=68, help="The chain subnet uid.")
        # Adds subtensor specific arguments.
        bt.subtensor.add_args(parser)
        # Adds logging specific arguments.
        bt.logging.add_args(parser)
        # Adds wallet specific arguments.
        bt.wallet.add_args(parser)
        # Parse the config.
        config = bt.config(parser)
        # Set up logging directory.
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                'miner',
            )
        )
        # Ensure the logging directory exists.
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        # Set up logging.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")
        bt.logging.info(self.config)

    async def setup_bittensor_objects(self):
        # Build Bittensor validator objects.
        bt.logging.info("Setting up Bittensor objects.")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        async with bt.async_subtensor(network=self.config.network) as subtensor:
            self.subtensor = subtensor
            bt.logging.info(f"Subtensor: {self.subtensor}")

            # Initialize and sync metagraph
            self.metagraph = await self.subtensor.metagraph(self.config.netuid)
            await self.metagraph.sync()
            bt.logging.info(f"Metagraph synced: {self.metagraph}")
            
            # Log stake distribution
            stakes = self.metagraph.S.tolist()
            sorted_stakes = sorted([(i, s) for i, s in enumerate(stakes)], key=lambda x: x[1], reverse=True)
            bt.logging.info("Top 5 validators by stake:")
            for uid, stake in sorted_stakes[:5]:
                bt.logging.info(f"UID: {uid}, Stake: {stake}")

    async def get_commitments(self, metagraph, block_hash: str) -> dict:
        """
        Retrieve commitments for all miners on a given subnet (netuid) at a specific block.

        Args:
            subtensor: The subtensor client object.
            netuid (int): The network ID.
            block (int, optional): The block number to query. Defaults to None.

        Returns:
            dict: A mapping from hotkey to a SimpleNamespace containing uid, hotkey,
                block, and decoded commitment data.
        """

        # Gather commitment queries for all validators (hotkeys) concurrently.
        commits = await asyncio.gather(*[
            self.subtensor.substrate.query(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.config.netuid, hotkey],
                block_hash=block_hash,
            ) for hotkey in metagraph.hotkeys
        ])

        # Process the results and build a dictionary with additional metadata.
        result = {}
        for uid, hotkey in enumerate(metagraph.hotkeys):
            commit = cast(dict, commits[uid])
            if commit:
                result[hotkey] = SimpleNamespace(
                    uid=uid,
                    hotkey=hotkey,
                    block=commit['block'],
                    data=decode_metadata(commit)
                )
        return result

    def stream_random_chunk_from_dataset(self):
        # Streams a random chunk from the dataset repo on huggingface.
        files = list_repo_files(self.hugging_face_dataset_repo, repo_type='dataset')
        files = [file for file in files if file.endswith('.csv')]
        random_file = random.choice(files)
        bt.logging.info(f"Loading dataset chunk from: {random_file}")
        dataset_dict = load_dataset(self.hugging_face_dataset_repo,
                                    data_files={'train': random_file},
                                    streaming=True,
                                    )
        dataset = dataset_dict['train']
        batched = dataset.batch(self.chunk_size)
        return batched
    
    async def get_protein_from_epoch_start(self, epoch_start: int):
        """
        Picks the highest-stake protein from the window [epoch_start .. epoch_start + tolerance].
        """
        final_block = epoch_start + self.tolerance
        current_block = await self.subtensor.get_current_block()

        if final_block > current_block:
            while (await self.subtensor.get_current_block()) < final_block:
                await asyncio.sleep(12)

        block_hash = await self.subtensor.determine_block_hash(final_block)
        commits = await self.get_commitments(self.metagraph, block_hash)

        # Filter to keep only commits that occurred after or at epoch start
        fresh_commits = {
            hotkey: commit
            for hotkey, commit in commits.items()
            if commit.block >= epoch_start
        }
        if not fresh_commits:
            bt.logging.info(f"No commits found in block window [{epoch_start}, {final_block}].")
            return None

        highest_stake_commit = max(
            fresh_commits.values(),
            key=lambda c: self.metagraph.S[c.uid],
            default=None
        )
        return highest_stake_commit.data if highest_stake_commit else None

    async def run_psichic_model_loop(self):
        """
        Continuously runs the PSICHIC model on batches of molecules from the dataset.
        Optimized for GPU processing with H200 SXM.
        """
        dataset = self.stream_random_chunk_from_dataset()
        
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Create CUDA streams for overlapped execution
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    # Record start time
                    start_event.record()
                    
                    # Preprocess data in stream1
                    with torch.cuda.stream(stream1):
                        df = pd.DataFrame.from_dict(chunk)
                        df['product_name'] = df['product_name'].apply(lambda x: x.replace('"', ''))
                        df['product_smiles'] = df['product_smiles'].apply(lambda x: x.replace('"', ''))
                    
                    # Wait for preprocessing to complete
                    torch.cuda.current_stream().wait_stream(stream1)
                    
                    # Run inference with mixed precision in stream2
                    with torch.cuda.stream(stream2):
                        with torch.cuda.amp.autocast():
                            bt.logging.debug(f'Running inference on batch of {len(df)} molecules')
                            chunk_psichic_scores = self.psichic_wrapper.run_validation(df['product_smiles'].tolist())
                    
                    # Wait for inference to complete
                    torch.cuda.current_stream().wait_stream(stream2)
                    
                    # Record end time and calculate duration
                    end_event.record()
                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
                    bt.logging.info(f"Processed {len(df)} molecules in {duration:.2f} seconds ({len(df)/duration:.2f} molecules/sec)")
                    
                    # Process results
                    chunk_psichic_scores = chunk_psichic_scores.sort_values(by=self.psichic_result_column_name, ascending=False).reset_index(drop=True)
                    if chunk_psichic_scores[self.psichic_result_column_name].iloc[0] > self.best_score:
                        async with self.shared_lock:
                            candidate_molecule = chunk_psichic_scores['Ligand'].iloc[0]
                            self.best_score = chunk_psichic_scores[self.psichic_result_column_name].iloc[0]
                            self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]
                            bt.logging.info(f"New best score: {self.best_score}, New candidate product: {self.candidate_product}")
                            
                            # Log GPU memory stats when we find a better candidate
                            bt.logging.debug(f"GPU memory usage: {get_gpu_memory_usage()}")
                    
                    # Prevent GPU overloading by occasional synchronization
                    if random.random() < 0.05:  # 5% chance to clean up
                        empty_gpu_cache()
                        
                    await asyncio.sleep(1)  # Reduced sleep time for faster throughput

            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
                bt.logging.error(traceback.format_exc())
                empty_gpu_cache()  # Clear GPU memory on error
                self.shutdown_event.set()

    async def run(self):
        # The Main Mining Loop.
        bt.logging.info("Starting miner loop.")
        await self.setup_bittensor_objects()

        # Startup case: In case we start mid-epoch get most recent protein and start inference
        current_block = await self.subtensor.get_current_block()
        last_boundary = (current_block // self.epoch_length) * self.epoch_length
        start_protein = await self.get_protein_from_epoch_start(last_boundary)
        if start_protein:
            self.current_challenge_protein = start_protein
            self.last_challenge_protein = start_protein
            bt.logging.info(f"Startup protein: {start_protein}")

            protein_sequence = get_sequence_from_protein_code(start_protein)
            try:
                # Clear CUDA cache before model initialization
                empty_gpu_cache()
                
                # Initialize model with GPU timing
                with gpu_timer("Model initialization"):
                    self.psichic_wrapper.run_challenge_start(protein_sequence)
                
                bt.logging.info(f"Initialized model for {start_protein}")
                bt.logging.info(f"GPU memory after model init: {get_gpu_memory_usage()}")
            except Exception as e:
                bt.logging.error(f"Error initializing model: {e}")
                bt.logging.error(traceback.format_exc())

            try:
                self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                bt.logging.debug("Inference started on startup protein.")
            except Exception as e:
                bt.logging.error(f"Error starting inference: {e}")
                bt.logging.error(traceback.format_exc())

        while True:
            try:
                current_block = await self.subtensor.get_current_block()
                # If we are at the epoch boundary, wait for the tolerance blocks to find a new protein
                if current_block % self.epoch_length == 0:
                    bt.logging.info(f"Epoch boundary at block {current_block}, waiting {self.tolerance} blocks.")
                    new_protein = await self.get_protein_from_epoch_start(current_block)
                    if new_protein and new_protein != self.last_challenge_protein:
                        self.current_challenge_protein = new_protein
                        self.last_challenge_protein = new_protein
                        bt.logging.info(f"New protein: {new_protein}")

                    # If old task still running, set shutdown event
                    if self.inference_task:
                        if not self.inference_task.done():
                            self.shutdown_event.set()
                            bt.logging.debug(f"Shutdown event set for old inference task.")

                            # reset old values for best score, etc
                            self.candidate_product = None
                            self.candidate_product_score = 0
                            self.best_score = 0
                            self.last_submitted_product = None
                            self.shutdown_event = asyncio.Event()

                    # Clean up GPU memory before loading new model
                    empty_gpu_cache()
                    bt.logging.info(f"GPU memory before new model: {get_gpu_memory_usage()}")
                    
                    # Get protein sequence from uniprot
                    protein_sequence = get_sequence_from_protein_code(self.current_challenge_protein)

                    # Initialize PSICHIC for new protein
                    bt.logging.info(f'Initializing model for protein sequence: {protein_sequence}')
                    try:
                        # Initialize model with GPU timing
                        with gpu_timer("Model initialization"):
                            self.psichic_wrapper.run_challenge_start(protein_sequence)
                        
                        bt.logging.info('Model initialized successfully.')
                        bt.logging.info(f"GPU memory after model init: {get_gpu_memory_usage()}")
                    except Exception as e:
                        try:
                            # Download model if needed
                            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
                            
                            # Clear GPU cache again
                            empty_gpu_cache()
                            
                            # Try initialization again
                            with gpu_timer("Model initialization retry"):
                                self.psichic_wrapper.run_challenge_start(protein_sequence)
                            
                            bt.logging.info('Model initialized successfully on retry.')
                            bt.logging.info(f"GPU memory after model init: {get_gpu_memory_usage()}")
                        except Exception as e:
                            bt.logging.error(f'Error initializing model: {e}')
                            bt.logging.error(traceback.format_exc())

                    # Start inference loop
                    try:
                        self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                        bt.logging.debug(f'Inference task started successfully')
                    except Exception as e:
                        bt.logging.error(f'Error initializing inference: {e}')
                        bt.logging.error(traceback.format_exc())

                # Check if candidate product has changed
                async with self.shared_lock:
                    if self.candidate_product:
                        if self.candidate_product != self.last_submitted_product:
                            current_product_to_submit = self.candidate_product
                            current_product_score = self.best_score
                            try:
                                await self.subtensor.set_commitment(
                                    wallet=self.wallet,
                                    netuid=self.config.netuid,
                                    data=current_product_to_submit
                                    )
                                self.last_submitted_product = current_product_to_submit
                                bt.logging.info(f'Submitted product: {current_product_to_submit} with score: {current_product_score}')

                            except MetadataError as e:
                                bt.logging.info(f'Too soon to commit again, will keep looking for better candidates.')
                            except Exception as e:
                                bt.logging.error(e)
                await asyncio.sleep(1)

                # Periodically update our knowledge of the network graph.
                if self.current_block % 60 == 0:
                    await self.metagraph.sync()
                    log = (
                        f'Block: {self.metagraph.block.item()} | '
                        f'Number of nodes: {self.metagraph.n} | '
                        f'Current epoch: {self.metagraph.block.item() // self.epoch_length}'
                    )
                    bt.logging.info(log)
                    
                    # Periodically clean CUDA cache
                    if self.current_block % 360 == 0:
                        bt.logging.info("Performing periodic GPU memory cleanup")
                        empty_gpu_cache()
                        bt.logging.info(f"GPU memory after cleanup: {get_gpu_memory_usage()}")
                        
                self.current_block += 1

            except RuntimeError as e:
                bt.logging.error(e)
                bt.logging.error(traceback.format_exc())
                # Clear CUDA cache on error
                empty_gpu_cache()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting miner.")
                # Clean up GPU resources before exit
                empty_gpu_cache()
                exit()

# Run the miner.
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())