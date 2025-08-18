import pandas as pd
import json
from sklearn.model_selection import train_test_split
import argparse
import os

# Define special tokens to be used in the vocabulary
PAD_TOKEN = '<PAD>'  # Token for padding sequences to the same length
SOS_TOKEN = '<SOS>'  # "Start of Sequence" token
EOS_TOKEN = '<EOS>'  # "End of Sequence" token
UNK_TOKEN = '<UNK>'  # Token for unknown characters (not in vocabulary)

class Vocabulary:
    """
    A class to create and manage the vocabulary for both graphemes (source) and phonemes (target).
    It handles the mapping between tokens and numerical indices.
    """
    def __init__(self, name):
        self.name = name
        self.token2index = {}
        self.index2token = {}
        self.n_tokens = 0
        self._add_special_tokens()

    def _add_special_tokens(self):
        """Adds the predefined special tokens to the vocabulary."""
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self.add_token(token)

    def add_token(self, token):
        """Adds a new token to the vocabulary if it doesn't already exist."""
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1

    def add_sequence(self, sequence):
        """Iterates through a sequence (word or phoneme string) and adds each token."""
        for token in sequence:
            self.add_token(token)

    def to_index(self, token):
        """Returns the index for a given token, defaulting to the UNK_TOKEN index."""
        return self.token2index.get(token, self.token2index[UNK_TOKEN])

    def to_token(self, index):
        """Returns the token for a given index."""
        return self.index2token.get(index, UNK_TOKEN)

    def save(self, filename):
        """Saves the vocabulary mappings to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            data = {
                'name': self.name,
                'token2index': self.token2index,
                'index2token': {int(k): v for k, v in self.index2token.items()},
                'n_tokens': self.n_tokens
            }
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary '{self.name}' saved to {filename}")

def process_data(input_path, output_dir):
    """
    Main function to load, process, and save the G2P data.
    """
    # --- 1. Load Data ---
    # Resolve input path relative to project root (one directory up from script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    abs_input_path = input_path
    if not os.path.isabs(input_path):
        abs_input_path = os.path.join(project_root, input_path)
    print(f"Loading data from {abs_input_path}...")
    try:
        df = pd.read_csv(abs_input_path)
    except FileNotFoundError:
        print(f"Error: The file {abs_input_path} was not found.")
        return

    # Ensure required columns exist
    if 'headword' not in df.columns or 'normalized_ipa' not in df.columns:
        print("Error: CSV must contain 'headword' and 'normalized_ipa' columns.")
        return

    # Drop rows with missing values in our key columns
    df.dropna(subset=['headword', 'normalized_ipa'], inplace=True)
    
    # Convert to lowercase to ensure consistency
    df['headword'] = df['headword'].str.lower()
    df['normalized_ipa'] = df['normalized_ipa'].str.lower()


    # --- 2. Create Vocabularies ---
    print("Building vocabularies...")
    # Source vocabulary (graphemes)
    source_vocab = Vocabulary('grapheme')
    # Target vocabulary (phonemes)
    target_vocab = Vocabulary('phoneme')

    # The source sequence is the characters of the headword
    # The target sequence is the characters of the normalized IPA
    source_sequences = [list(word) for word in df['headword']]
    target_sequences = [list(ipa) for ipa in df['normalized_ipa']]

    for seq in source_sequences:
        source_vocab.add_sequence(seq)
    for seq in target_sequences:
        target_vocab.add_sequence(seq)

    print(f"Source (grapheme) vocabulary size: {source_vocab.n_tokens}")
    print(f"Target (phoneme) vocabulary size: {target_vocab.n_tokens}")


    # --- 3. Split Data ---
    print("Splitting data into training, validation, and test sets...")
    # Pair up source and target sequences
    data_pairs = list(zip(source_sequences, target_sequences))
    
    # First, split into training (80%) and a temporary set (20%)
    train_pairs, temp_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
    # Then, split the temporary set into validation (10%) and test (10%)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

    print(f"Training examples: {len(train_pairs)}")
    print(f"Validation examples: {len(val_pairs)}")
    print(f"Test examples: {len(test_pairs)}")

    # --- 4. Save Processed Data ---

    # Resolve output_dir relative to project root
    abs_output_dir = output_dir
    if not os.path.isabs(output_dir):
        abs_output_dir = os.path.join(project_root, output_dir)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)
        print(f"Created output directory: {abs_output_dir}")

    # Save vocabularies
    source_vocab.save(os.path.join(abs_output_dir, 'source_vocab.json'))
    target_vocab.save(os.path.join(abs_output_dir, 'target_vocab.json'))

    # Save datasets
    # We save them as simple JSON files for easy loading later
    with open(os.path.join(abs_output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=4)
    with open(os.path.join(abs_output_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_pairs, f, ensure_ascii=False, indent=4)
    with open(os.path.join(abs_output_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=4)

    print(f"\nData processing complete. All files saved in '{abs_output_dir}' directory.")


if __name__ == '__main__':
    # --- 5. Command-Line Interface ---
    parser = argparse.ArgumentParser(description="Prepare G2P data for seq2seq modeling.")
    parser.add_argument('--input', type=str, default=r'data\\combined_data.csv',
                        help="Path to the input CSV file (use double backslashes for Windows paths, e.g., data\\combined_data.csv).")
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help="Directory to save the processed files.")
    args = parser.parse_args()
    process_data(args.input, args.output_dir)