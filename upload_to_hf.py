"""
Upload trained model artifacts to Hugging Face Hub.
Called by GitHub Actions after training completes.
"""
import os, sys
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN', '')
if not token:
    print("HF_TOKEN not set, skipping upload")
    sys.exit(0)

api = HfApi()
who = api.whoami(token=token)
HF_USERNAME = who['name']
REPO_NAME = 'imdb-sentiment-tfidf-nn'
repo_id = f'{HF_USERNAME}/{REPO_NAME}'

api.create_repo(repo_id, token=token, exist_ok=True, private=False)
print(f'HF repo: https://huggingface.co/{repo_id}')

artifacts = [
    'model/model.pt',
    'model/config.json',
    'model/metrics.json',
    'model/tfidf_vectorizer.pkl',
]

for fpath in artifacts:
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fpath,
            repo_id=repo_id,
            token=token,
        )
        print(f'  Uploaded {fpath}')
    else:
        print(f'  WARNING: {fpath} not found, skipping')

print(f'Done: https://huggingface.co/{repo_id}')
