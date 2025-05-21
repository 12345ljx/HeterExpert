# example code to download a dataset

from datasets import load_dataset, DownloadConfig

config = DownloadConfig(resume_download=True) 
dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', download_config=config)
dataset.save_to_disk('./data/wikitext2-raw')

print(dataset)

