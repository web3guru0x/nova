import asyncio
import math
import os
import sys
import argparse
import traceback
from typing import cast
from types import SimpleNamespace
import bittensor as bt
import torch
from substrateinterface import SubstrateInterface
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

# Import GPU utilities
from gpu_utils import setup_gpu_for_h200, empty_gpu_cache, gpu_timer, get_gpu_memory_usage, optimize_memory_for_inference
from my_utils import get_smiles, get_random_protein, get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper
from bittensor.core.chain_data.utils import decode_metadata

# Initialize PSICHIC wrapper once for the module
psichic = PsichicWrapper()

def get_config():
    """
    Parse command-line arguments to set up the configuration for the wallet
    and subtensor client.
    """
    load_dotenv()
    parser = argparse.ArgumentParser('Nova')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK")
    node = SubstrateInterface(url=config.network)
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value

    return config

async def check_registration(wallet, subtensor, netuid):
    """
    Confirm that the wallet hotkey is in the metagraph for the specified netuid.
    Logs an error and exits if it's not registered. Warns if stake is less than 1000.
    """
    metagraph = await subtensor.metagraph(netuid=netuid)
    my_hotkey_ss58 = wallet.hotkey.ss58_address

    if my_hotkey_ss58 not in metagraph.hotkeys:
        bt.logging.error(f"Hotkey {my_hotkey_ss58} is not registered on netuid {netuid}.")
        bt.logging.error("Are you sure you've registered and staked?")
        sys.exit(1) 
    
    uid = metagraph.hotkeys.index(my_hotkey_ss58)
    myStake = metagraph.S[uid]
    bt.logging.info(f"Hotkey {my_hotkey_ss58} found with UID={uid} and stake={myStake}")

    if (myStake < 1000):
        bt.logging.warning(f"Hotkey has less than 1000 stake, unable to validate")

def run_model(protein: str, molecule: str) -> float:
    """
    Given a protein sequence (protein) and a molecule identifier (molecule),
    retrieves its SMILES string, then uses the PsichicWrapper to produce
    a predicted binding score. Returns 0.0 if SMILES not found or if
    there's any issue with scoring.
    """
    # Use GPU timer for performance monitoring
    with gpu_timer(f"Score {molecule[:10]}"):
        # Use automatic mixed precision for faster computation
        with torch.cuda.amp.autocast():
            try:
                smiles = get_smiles(molecule)
                if not smiles:
                    bt.logging.debug(f"Could not retrieve SMILES for '{molecule}', returning score of 0.0.")
                    return 0.0

                results_df = psichic.run_validation([smiles])  # returns a DataFrame
                if results_df.empty:
                    bt.logging.warning("Psichic returned an empty DataFrame, returning 0.0.")
                    return 0.0

                predicted_score = results_df.iloc[0]['predicted_binding_affinity']
                if predicted_score is None:
                    bt.logging.warning("No 'predicted_binding_affinity' found, returning 0.0.")
                    return 0.0

                return float(predicted_score)
            except Exception as e:
                bt.logging.error(f"Error running model: {e}")
                bt.logging.error(traceback.format_exc())
                empty_gpu_cache()  # Clean up GPU memory on error
                return 0.0


async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int) -> dict:
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
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
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


async def main(config):
    """
    Main routine that continuously checks for the end of an epoch to perform:
        - Setting a new commitment.
        - Retrieving past commitments.
        - Selecting the best protein/molecule pairing based on stakes and scores.
        - Setting new weights accordingly.

    Args:
        config: Configuration object for subtensor and wallet.
    """
    # Setup GPU for optimal performance
    if not setup_gpu_for_h200():
        bt.logging.error("GPU setup failed - check your CUDA installation")
        sys.exit(1)
        
    bt.logging.info(f"GPU memory before initialization: {get_gpu_memory_usage()}")
    
    # Optimize memory allocation for inference workloads
    optimize_memory_for_inference(max_batch_size=64)  # Smaller batch size for validator

    wallet = bt.wallet(config=config)

    # Initialize the asynchronous subtensor client.
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()

    # Check if the hotkey is registered and has at least 1000 stake.
    await check_registration(wallet, subtensor, config.netuid)

    tolerance = 3 # block tolerance window for validators to commit protein

    while True:
        try:
            # Fetch the current metagraph for the given subnet (netuid 68).
            metagraph = await subtensor.metagraph(config.netuid)
            bt.logging.debug(f'Found {metagraph.n} nodes in network')
            current_block = await subtensor.get_current_block()

            # Check if the current block marks the end of an epoch (using a 360-block interval).
            if current_block % config.epoch_length == 0:
                # Clean GPU memory at start of epoch
                empty_gpu_cache()
                bt.logging.info(f"GPU memory at epoch start: {get_gpu_memory_usage()}")

                try:
                    # Set the next commitment target protein.
                    protein = get_random_protein()
                    await subtensor.set_commitment(
                        wallet=wallet,
                        netuid=config.netuid,
                        data=protein
                    )
                    bt.logging.info(f'Committed successfully: {protein}')
                except Exception as e:
                    bt.logging.error(f'Error: {e}')

                # Retrieve commitments from the previous epoch.
                prev_epoch = current_block - config.epoch_length
                best_stake = -math.inf
                current_protein = None

                block_to_check = prev_epoch
                block_hash_to_check = await subtensor.determine_block_hash(block_to_check + tolerance)  
                epoch_metagraph = await subtensor.metagraph(config.netuid, block=block_to_check + tolerance)
                epoch_commitments = await get_commitments(subtensor, epoch_metagraph, block_hash_to_check, netuid=config.netuid)
                epoch_commitments = {k: v for k, v in epoch_commitments.items() if current_block - v.block <= (config.epoch_length + tolerance)}
                high_stake_protein_commitment = max(
                    epoch_commitments.values(),
                    key=lambda commit: epoch_metagraph.S[commit.uid],
                    default=None
                )
                if not high_stake_protein_commitment:
                    bt.logging.error("Error getting current protein commitment.")
                    current_protein = None
                    continue

                current_protein = high_stake_protein_commitment.data
                bt.logging.info(f"Current protein: {current_protein}")

                protein_sequence = get_sequence_from_protein_code(current_protein)

                # Initialize PSICHIC for new protein
                bt.logging.info(f'Initializing model for protein sequence: {protein_sequence}')
                try:
                    # Initialize model with GPU timing
                    with gpu_timer("Model initialization"):
                        psichic.run_challenge_start(protein_sequence)
                    
                    bt.logging.info('Model initialized successfully.')
                    bt.logging.info(f"GPU memory after model init: {get_gpu_memory_usage()}")
                except Exception as e:
                    try:
                        os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
                        
                        # Clear GPU cache and try again
                        empty_gpu_cache()
                        
                        with gpu_timer("Model initialization retry"):
                            psichic.run_challenge_start(protein_sequence)
                        
                        bt.logging.info('Model initialized successfully on retry.')
                        bt.logging.info(f"GPU memory after model init: {get_gpu_memory_usage()}")
                    except Exception as e:
                        bt.logging.error(f'Error initializing model: {e}')
                        bt.logging.error(traceback.format_exc())
                        empty_gpu_cache()

                # Retrieve the latest commitments (current epoch).
                current_block_hash = await subtensor.determine_block_hash(current_block)
                current_commitments = await get_commitments(subtensor, metagraph, current_block_hash, netuid=config.netuid)

                # Create CUDA events for timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                # Identify the best molecule based on the scoring function.
                best_score = -math.inf
                total_commits = 0
                best_molecule = None
                
                # Process commitments in batches to avoid GPU memory pressure
                batch_size = 16
                commitment_items = list(current_commitments.items())
                
                # Create stream for overlapping processing
                stream = torch.cuda.Stream()
                
                bt.logging.info(f"Processing {len(commitment_items)} commitments in batches of {batch_size}")
                
                for i in range(0, len(commitment_items), batch_size):
                    batch = commitment_items[i:i+batch_size]
                    
                    # Track GPU memory usage for batches
                    if i % (batch_size * 4) == 0:
                        bt.logging.debug(f"Processing batch {i}-{i+batch_size}, GPU memory: {get_gpu_memory_usage()}")
                    
                    with torch.cuda.stream(stream):
                        for hotkey, commit in batch:
                            if current_block - commit.block <= config.epoch_length:
                                total_commits += 1
                                molecule = commit.data
                                score = run_model(protein=current_protein, molecule=molecule)
                                
                                # If the score is higher, or equal but the block is earlier, update the best.
                                if (score > best_score) or (score == best_score and best_molecule is not None and commit.block < best_molecule.block):
                                    best_score = score
                                    best_molecule = commit
                                    bt.logging.info(f"New best molecule found: {molecule[:20]}... with score {score}")
                    
                    # Synchronize after each batch
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    
                    # Occasional memory cleanup
                    if i % (batch_size * 8) == 0:
                        empty_gpu_cache()

                end_event.record()
                torch.cuda.synchronize()
                process_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
                bt.logging.info(f"Processed {total_commits} commits in {process_time:.2f} seconds ({total_commits/process_time:.2f} commits/sec)")

                # Ensure a best molecule was found before setting weights.
                if best_molecule is not None:
                    try:
                        # Create weights where the best molecule's UID receives full weight.
                        weights = [0.0 for i in range(metagraph.n)]
                        weights[best_molecule.uid] = 1.0
                        bt.logging.info(f"Setting weight 1.0 for UID {best_molecule.uid} with molecule {best_molecule.data[:20]}...")
                        
                        uids = list(range(metagraph.n))
                        result, message = await subtensor.set_weights(
                            wallet=wallet,
                            uids=uids,
                            weights=weights,
                            netuid=config.netuid,
                            wait_for_inclusion=True,
                            )
                        if result:
                            bt.logging.info(f"Weights set successfully")
                        else:
                            bt.logging.error(f"Error setting weights: {message}")
                    except Exception as e:
                        bt.logging.error(f"Error setting weights: {e}")
                        bt.logging.error(traceback.format_exc())
                else:
                    bt.logging.info("No valid molecule commitment found for current epoch.")

                # Cleanup GPU memory after processing
                empty_gpu_cache()
                bt.logging.info(f"GPU memory after epoch processing: {get_gpu_memory_usage()}")
                
                # Sleep briefly to prevent busy-waiting
                await asyncio.sleep(1)
                
            # Keep validator alive
            elif current_block % (config.epoch_length/2) == 0:
                subtensor = bt.async_subtensor(network=config.network)
                await subtensor.initialize()
                bt.logging.info("Validator reset subtensor connection.")
                
                # Periodically clean GPU memory
                empty_gpu_cache()
                bt.logging.info(f"GPU memory at midpoint: {get_gpu_memory_usage()}")
                
                await asyncio.sleep(12) # Sleep for 1 block to avoid unnecessary re-connection
                
            else:
                blocks_remaining = config.epoch_length - (current_block % config.epoch_length)
                if blocks_remaining % 60 == 0 or blocks_remaining < 10:
                    bt.logging.info(f"Waiting for epoch to end... {blocks_remaining} blocks remaining.")
                await asyncio.sleep(1)
                
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            bt.logging.error(traceback.format_exc())
            # Clean up GPU memory on error
            empty_gpu_cache()
            await asyncio.sleep(5)


if __name__ == "__main__":
    config = get_config()
    asyncio.run(main(config))