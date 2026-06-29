import os
from datasets import load_dataset
import pandas as pd
import json
from datetime import datetime

# MVP Scope: Ingest a LeRobot v3 format dataset from Hugging Face Hub

def fetch_lerobot_dataset_metadata(repo_id="lerobot/aloha_mobile_cabinet"):
    """
    Fetches the metadata and info for a given dataset on HF Hub.
    """
    print(f"Fetching dataset info for: {repo_id}")
    
    try:
        # Load the dataset in streaming mode to avoid downloading massive video files
        # We only care about the tabular/proprioception data right now
        dataset = load_dataset(repo_id, streaming=True)
        print("Successfully connected to dataset.")
        
        # Get one episode to inspect structure
        if 'train' in dataset:
            iterator = iter(dataset['train'])
            first_row = next(iterator)
            print("\nDataset Schema (First Row Keys):")
            for key in first_row.keys():
                print(f"- {key}: {type(first_row[key])}")
                
            return True
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return False

if __name__ == "__main__":
    fetch_lerobot_dataset_metadata()
