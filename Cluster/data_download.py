from datasets import load_dataset, DownloadConfig, load_from_disk
import os

DATA_PATH = './Cluster/raw_data/wikitext2-raw'

config = DownloadConfig(resume_download=True) 
dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', download_config=config)
dataset.save_to_disk(DATA_PATH)

# dataset = load_from_disk(DATA_PATH)
print(dataset)
