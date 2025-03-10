import os
import math
import random
import argparse
import asyncio
from typing import cast
from types import SimpleNamespace
import sys
import time

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
        self.chunk_size = 4096
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

        This method streams random chunks of molecule data from a Hugging Face dataset,
        processes them through the PSICHIC model to predict binding affinities, and updates
        the best candidate when a higher scoring molecule is found. Runs in a separate thread
        until the shutdown event is triggered.
        """
        bt.logging.info("Starting PSICHIC model loop")
        start_time_total = time.time()
        
        dataset_start = time.time()
        dataset = self.stream_random_chunk_from_dataset()
        dataset_time = time.time() - dataset_start
        bt.logging.info(f"⏱️ Dataset initialization took {dataset_time:.2f}s")
        
        chunk_count = 0
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    chunk_count += 1
                    bt.logging.info(f"Processing chunk #{chunk_count}")
                    chunk_start = time.time()
                    
                    # Step 1: DataFrame creation and cleaning
                    df_start = time.time()
                    df = pd.DataFrame.from_dict(chunk)
                    df['product_name'] = df['product_name'].str.replace('"', '')
                    df['product_smiles'] = df['product_smiles'].str.replace('"', '')
                    df_time = time.time() - df_start
                    bt.logging.info(f"⏱️ DataFrame creation: {df_time:.2f}s for {len(df)} molecules")
                    
                    # Step 2: PSICHIC model inference
                    inference_start = time.time()
                    bt.logging.debug(f'Running inference...')
                    chunk_psichic_scores = self.psichic_wrapper.run_validation(df['product_smiles'].tolist())
                    inference_time = time.time() - inference_start
                    bt.logging.info(f"⏱️ PSICHIC inference: {inference_time:.2f}s ({inference_time/len(df):.4f}s per molecule)")
                    
                    # Step 3: Process results
                    processing_start = time.time()
                    chunk_psichic_scores = chunk_psichic_scores.sort_values(
                        by=self.psichic_result_column_name, 
                        ascending=False
                    ).reset_index(drop=True)
                    
                    if not chunk_psichic_scores.empty and chunk_psichic_scores[self.psichic_result_column_name].iloc[0] > self.best_score:
                        update_start = time.time()
                        async with self.shared_lock:
                            candidate_molecule = chunk_psichic_scores['Ligand'].iloc[0]
                            self.best_score = chunk_psichic_scores[self.psichic_result_column_name].iloc[0]
                            self.candidate_product = df.loc[df['product_smiles'] == candidate_molecule, 'product_name'].iloc[0]
                            bt.logging.info(f"🏆 New best score: {self.best_score}, New candidate product: {self.candidate_product}")
                        update_time = time.time() - update_start
                        bt.logging.info(f"⏱️ Best score update: {update_time:.2f}s")
                    
                    processing_time = time.time() - processing_start
                    bt.logging.info(f"⏱️ Results processing: {processing_time:.2f}s")
                    
                    # Step 4: Total time for this chunk
                    chunk_time = time.time() - chunk_start
                    bt.logging.info(f"⏱️ TOTAL CHUNK PROCESSING TIME: {chunk_time:.2f}s")
                    
                    # Memory stats if available
                    try:
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                            bt.logging.info(f"🧠 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    except:
                        pass

            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
                self.shutdown_event.set()

        total_time = time.time() - start_time_total
        bt.logging.info(f"⏱️ PSICHIC model loop completed in {total_time:.2f}s, processed {chunk_count} chunks")

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
                self.psichic_wrapper.run_challenge_start(protein_sequence)
                bt.logging.info(f"Initialized model for {start_protein}")
            except Exception as e:
                bt.logging.error(f"Error initializing model: {e}")

            try:
                self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                bt.logging.debug("Inference started on startup protein.")
            except Exception as e:
                bt.logging.error(f"Error starting inference: {e}")


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

                    # Get protein sequence from uniprot
                    protein_sequence = get_sequence_from_protein_code(self.current_challenge_protein)

                    # Initialize PSICHIC for new protein
                    bt.logging.info(f'Initializing model for protein sequence: {protein_sequence}')
                    try:
                        self.psichic_wrapper.run_challenge_start(protein_sequence)
                        bt.logging.info('Model initialized successfully.')
                    except Exception as e:
                        try:
                            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
                            self.psichic_wrapper.run_challenge_start(protein_sequence)
                            bt.logging.info('Model initialized successfully.')
                        except Exception as e:
                            bt.logging.error(f'Error initializing model: {e}')

                    # Start inference loop
                    try:
                        self.inference_task = asyncio.create_task(self.run_psichic_model_loop())
                        bt.logging.debug(f'Inference task started successfully')
                    except Exception as e:
                        bt.logging.error(f'Error initializing inference: {e}')


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
                self.current_block += 1

            except RuntimeError as e:
                bt.logging.error(e)
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting miner.")
                exit()

# Run the miner.
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())