# G2P_Seq2Seq

This project implements a Grapheme-to-Phoneme (G2P) sequence-to-sequence model using PyTorch.

## Project Structure

- `src/` - Source code for data preparation, training, and evaluation
- `data/` - Raw data files (not tracked by git)
- `processed_data/` - Processed data for training/validation/testing (not tracked by git)
- `models/` - Saved model checkpoints (not tracked by git)
- `outputs/` - Output files, logs, or predictions (not tracked by git)
- `notebooks/` - Jupyter notebooks for exploration
- `requirements.txt` - Python dependencies
- `setup.py` - Project setup and folder initialization

## Setup Instructions

1. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   ```
2. Install dependencies and initialize folders:
   ```
   pip install -r requirements.txt
   python setup.py develop
   ```
3. Prepare your data:
   - Place your raw CSV in `data/` (e.g., `data/combined_data.csv`).
   - Run the data preparation script:
     ```
     python src/prepare_data.py
     ```
4. Train the model:
   ```
   python src/train.py
   ```

## Notes
- All intermediate and output data folders are ignored by git (see `.gitignore`).
- Customize hyperparameters and paths using command-line arguments for each script.
