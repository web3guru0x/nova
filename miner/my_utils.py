import requests
import os
import json
from dotenv import load_dotenv
import bittensor as bt

#load_dotenv(override=True)

def get_smiles(product_name):

    api_key = os.environ.get("VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set.")

    url = f"https://8vzqr9wt22.execute-api.us-east-1.amazonaws.com/dev/smiles/{product_name}"

    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)

    data = response.json()

    return data.get("smiles")

def get_random_protein():
    api_key = os.environ.get("VALIDATOR_API_KEY")
    if not api_key:
        raise ValueError("validator_api_key environment variable not set contact nova team for api key.")

    url = "https://rvhs77j663.execute-api.us-east-1.amazonaws.com/prod/random-protein-of-interest"
    headers = {"x-api-key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} {response.text}")

    data = response.json()

    if "body" in data:
        inner_body_str = data["body"]  # e.g. '{"uniprot_code": "A0S183", "protein_sequence": "..."}'
        try:
            inner_data = json.loads(inner_body_str) 
            return inner_data.get("uniprot_code")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse body as JSON: {e}")
    else:
        bt.logging.error("Unexpected API response structure.")

def get_sequence_from_protein_code(protein_code:str) -> str:

    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url)

    if response.status_code != 200:
        return None
    else:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        return amino_acid_sequence
