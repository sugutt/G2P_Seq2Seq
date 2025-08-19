import argparse
import os
from huggingface_hub import HfApi, HfFolder, Repository, upload_file

# Usage:
# python upload_to_huggingface.py --model_path models/g2p_model.pt --repo_id <username>/<repo_name> --token <hf_token>

def main(model_path, repo_id, token, filename=None):
    if filename is None:
        filename = os.path.basename(model_path)
    print(f"Uploading {model_path} to Hugging Face Hub repo {repo_id} as {filename}...")
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=filename,
        repo_id=repo_id,
        token=token,
    )
    print("Upload complete! View your model at https://huggingface.co/" + repo_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model file to Hugging Face Hub.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file (e.g., models/g2p_model.pt)')
    parser.add_argument('--repo_id', type=str, required=True, help='Hugging Face repo id (e.g., username/repo_name)')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face access token')
    parser.add_argument('--filename', type=str, default=None, help='Filename to use in the repo (optional)')
    args = parser.parse_args()
    main(args.model_path, args.repo_id, args.token, args.filename)
