import os
import shutil
import argparse

# This script copies a trained model to a model_snapshots/ directory for versioning and sharing on GitHub (for small models only).
# For large models, use Hugging Face Hub or other model hosting services.

def main(model_path, snapshot_name):
    os.makedirs('model_snapshots', exist_ok=True)
    dest_path = os.path.join('model_snapshots', snapshot_name)
    shutil.copy2(model_path, dest_path)
    print(f"Model copied to {dest_path}. You can now add and commit this file to GitHub if it is small enough.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy a trained model to model_snapshots/ for GitHub versioning.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (e.g., models/g2p_model.pt)')
    parser.add_argument('--snapshot_name', type=str, default='g2p_model.pt', help='Filename for the snapshot in model_snapshots/')
    args = parser.parse_args()
    main(args.model_path, args.snapshot_name)
