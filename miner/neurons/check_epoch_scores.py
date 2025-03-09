import asyncio
import os
import sys
import argparse
import math
from types import SimpleNamespace
from typing import cast

import bittensor as bt
from dotenv import load_dotenv
from substrateinterface import SubstrateInterface

# Importă utilitarele și wrapper-ul PSICHIC
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from my_utils import get_smiles, get_sequence_from_protein_code
from PSICHIC.wrapper import PsichicWrapper
from bittensor.core.chain_data.utils import decode_metadata

# Inițializează wrapper-ul PSICHIC
psichic = PsichicWrapper()

def get_config():
    load_dotenv()
    parser = argparse.ArgumentParser('NOVA Score Checker')
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)

    config = bt.config(parser)
    config.netuid = 68
    config.network = os.environ.get("SUBTENSOR_NETWORK", "wss://archive.chain.opentensor.ai:443")
    node = SubstrateInterface(url=config.network)
    config.epoch_length = node.query("SubtensorModule", "Tempo", [config.netuid]).value

    return config

async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int) -> dict:
    """Recuperează commitment-urile tuturor minerilor de pe subnet."""
    commits = await asyncio.gather(*[
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        ) for hotkey in metagraph.hotkeys
    ])

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

def run_model(protein: str, product_name: str) -> float:
    """Rulează modelul PSICHIC pentru a obține un scor."""
    try:
        # Obține SMILES de la API
        smiles = get_smiles(product_name)
        if not smiles:
            print(f"Nu s-a putut obține SMILES pentru '{product_name}'")
            return 0.0

        results_df = psichic.run_validation([smiles])
        if results_df.empty:
            return 0.0
        predicted_score = results_df.iloc[0]['predicted_binding_affinity']
        return float(predicted_score) if predicted_score is not None else 0.0
    except Exception as e:
        print(f"Eroare la rularea modelului pentru {product_name}: {e}")
        return 0.0

async def check_all_scores():
    """Verifică scorurile tuturor minerilor din epoca curentă."""
    config = get_config()
    
    # Inițializează subtensor-ul
    subtensor = bt.async_subtensor(network=config.network)
    await subtensor.initialize()
    
    # Obține metagraph-ul curent
    metagraph = await subtensor.metagraph(config.netuid)
    current_block = await subtensor.get_current_block()
    
    # Calculează începutul epocii curente
    current_epoch_start = (current_block // config.epoch_length) * config.epoch_length
    print(f"Blocul curent: {current_block}, Începutul epocii: {current_epoch_start}")
    
    # Obține proteina curentă (din epoca anterioară)
    prev_epoch_start = current_epoch_start - config.epoch_length
    tolerance = 3  # Fereastra de toleranță pentru commitmments
    
    block_hash_to_check = await subtensor.determine_block_hash(prev_epoch_start + tolerance)
    prev_metagraph = await subtensor.metagraph(config.netuid, block=prev_epoch_start + tolerance)
    
    prev_commitments = await get_commitments(subtensor, prev_metagraph, block_hash_to_check, netuid=config.netuid)
    
    # Găsește commitment-ul cu stake-ul cel mai mare
    high_stake_protein_commitment = max(
        prev_commitments.values(),
        key=lambda commit: prev_metagraph.S[commit.uid],
        default=None
    )
    
    if not high_stake_protein_commitment:
        print("Nu s-a găsit nicio proteină pentru epoca curentă.")
        return
    
    current_protein = high_stake_protein_commitment.data
    print(f"Proteina curentă: {current_protein}")
    
    # Inițializează modelul PSICHIC cu secvența proteinei
    protein_sequence = get_sequence_from_protein_code(current_protein)
    print(f"Inițializez modelul pentru secvența: {protein_sequence[:50]}...")
    
    try:
        psichic.run_challenge_start(protein_sequence)
        print("Model inițializat cu succes.")
    except Exception as e:
        print(f"Eroare la inițializarea modelului: {e}")
        try:
            os.system(f"wget -O {os.path.join(BASE_DIR, 'PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt')} https://huggingface.co/Metanova/PSICHIC/resolve/main/model.pt")
            psichic.run_challenge_start(protein_sequence)
            print("Model inițializat cu succes după descărcarea modelului.")
        except Exception as e:
            print(f"Eroare la inițializarea modelului: {e}")
            return
    
    # Obține commitment-urile din epoca curentă
    current_block_hash = await subtensor.determine_block_hash(current_block)
    current_commitments = await get_commitments(subtensor, metagraph, current_block_hash, netuid=config.netuid)
    
    # Filtrează commitment-urile pentru a păstra doar pe cele din epoca curentă
    current_epoch_commitments = {
        hotkey: commit for hotkey, commit in current_commitments.items()
        if commit.block >= current_epoch_start
    }
    
    print(f"Număr total de commitment-uri în epoca curentă: {len(current_epoch_commitments)}")
    
    # Evaluează fiecare commitment și afișează scorurile
    scores = []
    for hotkey, commit in current_epoch_commitments.items():
        product_name = commit.data
        uid = commit.uid
        # Rulăm modelul cu numele produsului
        score = run_model(protein=current_protein, product_name=product_name)
        stake = metagraph.S[uid].item()
        scores.append((uid, hotkey, product_name, score, stake, commit.block))
    
    # Sortează rezultatele după scor (descendent)
    scores.sort(key=lambda x: x[3], reverse=True)
    
    print("\n=== SCORURI ÎN EPOCA CURENTĂ ===")
    print("UID | Hotkey (trunchiat) | Moleculă | Scor | Stake | Bloc")
    print("-" * 80)
    
    for uid, hotkey, molecule, score, stake, block in scores:
        # Trunchează hotkey-ul pentru afișare
        short_hotkey = hotkey[:10] + "..." + hotkey[-5:]
        print(f"{uid:3d} | {short_hotkey} | {molecule[:20]}... | {score:.4f} | {stake:.2f} | {block}")
    
    if scores:
        best_uid, best_hotkey, best_molecule, best_score, best_stake, best_block = scores[0]
        print("\n=== CEL MAI BUN SCOR ===")
        print(f"UID: {best_uid}")
        print(f"Hotkey: {best_hotkey}")
        print(f"Moleculă: {best_molecule}")
        print(f"Scor: {best_score:.6f}")
        print(f"Stake: {best_stake:.2f}")
        print(f"Bloc: {best_block}")
    else:
        print("\nNu s-au găsit commitment-uri valide în epoca curentă.")

if __name__ == "__main__":
    asyncio.run(check_all_scores())