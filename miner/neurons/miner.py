import os
import math
import random
import argparse
import asyncio
from typing import cast
from types import SimpleNamespace
import sys
import time
import torch
import warnings
from rdkit import RDLogger
import gc
RDLogger.DisableLog('rdApp.*')

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

# Activează fallback pentru torch._dynamo errors la nivel global
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True

class Miner:
    def __init__(self):
        self.hugging_face_dataset_repo = 'Metanova/SAVI-2020'
        self.psichic_result_column_name = 'predicted_binding_affinity'
        self.chunk_size = 4096  # Increased from 4096 to process more molecules at once
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
        
        # Initialize molecule fingerprint cache for similarity-based lookup
        self.molecule_cache = {}
        self.fingerprint_similarity_threshold = self.runtime_config.SIMILARITY_THRESHOLD
        self.parallel_processing = self.runtime_config.PARALLEL_BATCH_PROCESSING
        self.runtime_config = self.psichic_wrapper.runtime_config

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

    def compute_fingerprint(self, smiles):
        """Calculate molecular fingerprint for similarity comparison"""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return None

    def similarity(self, fp1, fp2):
        """Calculate Tanimoto similarity between fingerprints"""
        from rdkit import DataStructs
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def get_cached_or_compute_score(self, smiles):
        """Check cache for similar molecules before computing score"""
        try:
            fp = self.compute_fingerprint(smiles)
            if fp is None:
                return None
                
            # Check cache for similar molecules
            for cached_smiles, (cached_fp, score) in self.molecule_cache.items():
                if self.similarity(fp, cached_fp) > self.fingerprint_similarity_threshold:
                    bt.logging.debug(f"Cache hit for similar molecule to {smiles}")
                    return score
                    
            # If not found in cache, return None to indicate computation needed
            return None
        except Exception as e:
            bt.logging.warning(f"Error in fingerprint calculation: {e}")
            return None

    def process_batch_efficiently(self, df, mini_batch_size=2048):  # Mărit dramatic
        """Process molecules in efficient mini-batches with caching"""
        # Preprocess DataFrame
        df['product_name'] = df['product_name'].str.replace('"', '')
        df['product_smiles'] = df['product_smiles'].str.replace('"', '')
        
        # Filter out molecules already in cache or with known similar structures
        to_process = []
        cached_scores = []
        indices = []
        
        # Procesare mai rapidă a cache-ului
        bt.logging.info(f"Verificare cache pentru {len(df)} molecule...")
        cache_start = time.time()
        
        for idx, row in df.iterrows():
            if pd.isna(row['product_smiles']):
                continue
                
            cached_score = self.get_cached_or_compute_score(row['product_smiles'])
            if cached_score is not None:
                cached_scores.append((idx, cached_score))
            else:
                to_process.append(row['product_smiles'])
                indices.append(idx)
        
        cache_time = time.time() - cache_start
        bt.logging.info(f"Verificare cache: {cache_time:.2f}s, Cache hits: {len(cached_scores)}, De procesat: {len(to_process)}")
        
        # Procesare paralelă pentru batch-uri mari
        if self.parallel_processing and len(to_process) > mini_batch_size:
            from concurrent.futures import ThreadPoolExecutor
            
            def process_batch(batch_idx):
                start_idx = batch_idx * mini_batch_size
                end_idx = min(start_idx + mini_batch_size, len(to_process))
                batch_smiles = to_process[start_idx:end_idx]
                if not batch_smiles:
                    return None
                    
                try:
                    batch_start = time.time()
                    batch_results = self.psichic_wrapper.run_validation(batch_smiles)
                    batch_time = time.time() - batch_start
                    bt.logging.info(f"Batch {batch_idx}: {len(batch_smiles)} molecule în {batch_time:.2f}s ({batch_time/len(batch_smiles):.5f}s per moleculă)")
                    return (batch_idx, batch_results, batch_smiles)
                except Exception as e:
                    bt.logging.error(f"Eroare la procesarea batch-ului {batch_idx}: {e}")
                    return None
            
            num_batches = (len(to_process) + mini_batch_size - 1) // mini_batch_size
            bt.logging.info(f"Procesare paralelă: {num_batches} batch-uri cu {mini_batch_size} molecule per batch")
            
            all_results = []
            with ThreadPoolExecutor(max_workers=self.runtime_config.MAX_PARALLEL_WORKERS) as executor:
                futures = [executor.submit(process_batch, i) for i in range(num_batches)]
                for future in futures:
                    result = future.result()
                    if result is not None:
                        batch_idx, batch_df, batch_smiles = result
                        all_results.append(batch_df)
                        
                        # Update cache with new results
                        for j, smiles in enumerate(batch_smiles):
                            if j < len(batch_df) and self.psichic_result_column_name in batch_df.columns:
                                score = batch_df.iloc[j][self.psichic_result_column_name]
                                fp = self.compute_fingerprint(smiles)
                                if fp is not None:
                                    self.molecule_cache[smiles] = (fp, score)
        else:
            # Procesare originală pentru batch-uri mici
            all_results = []
            for i in range(0, len(to_process), mini_batch_size):
                batch_smiles = to_process[i:i+mini_batch_size]
                if not batch_smiles:
                    continue
                    
                try:
                    batch_start = time.time()
                    batch_results = self.psichic_wrapper.run_validation(batch_smiles)
                    batch_time = time.time() - batch_start
                    bt.logging.info(f"Batch {i//mini_batch_size}: {len(batch_smiles)} molecule în {batch_time:.2f}s ({batch_time/len(batch_smiles):.5f}s per moleculă)")
                    
                    # Update cache
                    for j, smiles in enumerate(batch_smiles):
                        if j < len(batch_results) and self.psichic_result_column_name in batch_results.columns:
                            score = batch_results.iloc[j][self.psichic_result_column_name]
                            fp = self.compute_fingerprint(smiles)
                            if fp is not None:
                                self.molecule_cache[smiles] = (fp, score)
                    
                    all_results.append(batch_results)
                except Exception as e:
                    bt.logging.error(f"Eroare la procesarea batch-ului {i//mini_batch_size}: {e}")
        
        # Combine results and add cached results
        if all_results:
            results_df = pd.concat(all_results, ignore_index=True)
            bt.logging.info(f"Total rezultate procesate: {len(results_df)}")
        else:
            results_df = pd.DataFrame(columns=['Ligand', self.psichic_result_column_name])
            bt.logging.warning("Nu s-au găsit rezultate noi!")
        
        # Add cached results
        for idx, score in cached_scores:
            try:
                cached_row = pd.DataFrame({
                    'Ligand': [df.iloc[idx]['product_smiles']], 
                    self.psichic_result_column_name: [score]
                })
                results_df = pd.concat([results_df, cached_row], ignore_index=True)
            except Exception as e:
                bt.logging.error(f"Eroare la adăugarea rezultatului din cache pentru idx {idx}: {e}")
        
        # Maintain cache size
        if len(self.molecule_cache) > self.runtime_config.MOLECULE_CACHE_SIZE:
            remove_count = len(self.molecule_cache) - (self.runtime_config.MOLECULE_CACHE_SIZE // 2)
            keys_to_remove = list(self.molecule_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.molecule_cache[key]
        
        bt.logging.info(f"Rezultate totale după adăugarea cache: {len(results_df)}")
        return results_df

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
        
        # Try to enable CUDA graphs if available (PyTorch 2.0+)
        use_cuda_graphs = torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables')
        if use_cuda_graphs:
            bt.logging.info("CUDA graphs support detected and enabled")
        
        # Dezactivăm torch.compile pentru a evita problemele de compatibilitate
        use_torch_compile = False
        bt.logging.info("torch.compile dezactivat pentru compatibilitate")
        
        dataset_start = time.time()
        dataset = self.stream_random_chunk_from_dataset()
        dataset_time = time.time() - dataset_start
        bt.logging.info(f"⏱️ Dataset initialization took {dataset_time:.2f}s")
        
        # Clear CUDA cache before starting inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Use mixed precision if available
        autocast_enabled = torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast')
        
        chunk_count = 0
        gc.disable()
        while not self.shutdown_event.is_set():
            try:
                for chunk in dataset:
                    chunk_count += 1
                    bt.logging.info(f"Processing chunk #{chunk_count}")
                    chunk_start = time.time()
                    
                    # Step 1: DataFrame creation
                    df_start = time.time()
                    df = pd.DataFrame.from_dict(chunk)
                    df_time = time.time() - df_start
                    bt.logging.info(f"⏱️ DataFrame creation: {df_time:.2f}s for {len(df)} molecules")
                    
                    # Step 2: PSICHIC model inference with optimizations
                    inference_start = time.time()
                    bt.logging.debug(f'Running inference...')
                    
                    # Process in efficient batches with caching
                    if autocast_enabled:
                        with torch.amp.autocast(device_type='cuda', enabled=True):
                            chunk_psichic_scores = self.process_batch_efficiently(df)
                    else:
                        chunk_psichic_scores = self.process_batch_efficiently(df)
                        
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
                    
                    # Memory stats and management
                    try:
                        if torch.cuda.is_available():
                            # Free memory more aggressively
                            torch.cuda.empty_cache()
                            
                            # Log memory usage
                            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                            bt.logging.info(f"🧠 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    except Exception as e:
                        bt.logging.warning(f"Error checking GPU memory: {e}")

            except Exception as e:
                bt.logging.error(f"Error running PSICHIC model: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
                self.shutdown_event.set()

        gc.enable()
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
                import traceback
                traceback.print_exc()

            except KeyboardInterrupt:
                bt.logging.success("Keyboard interrupt detected. Exiting miner.")
                exit()

# Run the miner.
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.run())