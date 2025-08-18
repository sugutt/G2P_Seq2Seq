import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
from tqdm import tqdm
import jiwer
import wandb

# --- 1. Model Architecture (MUST be identical to train.py and inference.py) ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden.squeeze(0), cell.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device


class Vocabulary:
    """A simple wrapper to load the vocabulary from a JSON file."""
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token2index = data['token2index']
        self.index2token = {int(k): v for k, v in data['index2token'].items()}
        self.pad_idx = self.token2index['<PAD>']
        self.sos_idx = self.token2index['<SOS>']
        self.eos_idx = self.token2index['<EOS>']
        self.unk_idx = self.token2index['<UNK>']

class G2PTestDataset:
    """Simple loader for the test dataset."""
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        return "".join(source), "".join(target)

# --- 2. Evaluation Function ---

def run_evaluation(model, test_dataset, source_vocab, target_vocab, device, max_len=50):
    """
    Performs inference on the entire test set and calculates metrics.
    """
    model.eval()

    ground_truths = []
    predictions = []

    print("Running evaluation on the test set...")
    for source_word, target_phoneme in tqdm(test_dataset):
        ground_truths.append(target_phoneme)

        # Preprocess the input word
        tokens = [token for token in source_word.lower()]
        tokens = [source_vocab.index2token[source_vocab.sos_idx]] + tokens + [source_vocab.index2token[source_vocab.eos_idx]]
        src_indexes = [source_vocab.to_index(token) for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_len = torch.LongTensor([len(src_indexes)]).to(device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

        trg_indexes = [target_vocab.sos_idx]
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == target_vocab.eos_idx:
                break
        
        trg_tokens = [target_vocab.to_token(i) for i in trg_indexes]
        prediction = "".join(trg_tokens[1:-1]) # Remove <SOS> and <EOS>
        predictions.append(prediction)

    # Calculate metrics using jiwer
    error_metrics = jiwer.compute_measures(ground_truths, predictions)
    wer = error_metrics['wer']
    cer = error_metrics['cer']
    
    return wer, cer, ground_truths, predictions


# --- 3. Main Execution Block ---

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Initialize W&B ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args) # Log all argparse arguments
    )

    # Load vocabularies
    print("Loading vocabularies...")
    source_vocab = Vocabulary(os.path.join(args.data_dir, 'source_vocab.json'))
    target_vocab = Vocabulary(os.path.join(args.data_dir, 'target_vocab.json'))
    INPUT_DIM = len(source_vocab.token2index)
    OUTPUT_DIM = len(target_vocab.token2index)

    # Load test data
    print(f"Loading test data from {args.test_data_path}...")
    test_dataset = G2PTestDataset(args.test_data_path)

    # Instantiate model
    attn = Attention(args.enc_hid_dim, args.dec_hid_dim)
    enc = Encoder(INPUT_DIM, args.emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dropout)
    dec = Decoder(OUTPUT_DIM, args.emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    # Load the trained model weights
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Run evaluation
    wer, cer, ground_truths, predictions = run_evaluation(model, test_dataset, source_vocab, target_vocab, device)

    print("\n--- Evaluation Metrics ---")
    print(f"Word Error Rate (WER): {wer:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print("--------------------------\n")
    
    # --- Log metrics to W&B ---
    wandb.log({
        "test_wer": wer,
        "test_cer": cer,
    })

    # --- Create and log a W&B Table of predictions ---
    original_words = [item[0] for item in test_dataset.pairs]
    num_examples = min(200, len(original_words)) # Limit table size
    
    prediction_table = wandb.Table(columns=["Word", "Ground Truth", "Prediction", "Correct"])
    for i in range(num_examples):
        is_correct = "✅" if ground_truths[i] == predictions[i] else "❌"
        prediction_table.add_data(
            original_words[i], 
            ground_truths[i], 
            predictions[i],
            is_correct
        )
    
    wandb.log({"test_predictions": prediction_table})
    
    print(f"Evaluation complete. Results logged to W&B run: {wandb.run.get_url()}")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained G2P Seq2Seq model.')
    
    # Path arguments
    parser.add_argument('--model_path', type=str, default='g2p_model.pt', help='Path to the saved .pt model file.')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Directory for vocabulary files.')
    parser.add_argument('--test_data_path', type=str, default='processed_data/test.json', help='Path to the test data JSON file.')

    # Model hyperparameters (must match the trained model)
    parser.add_argument('--emb_dim', type=int, default=256, help='Dimension of character embeddings.')
    parser.add_argument('--enc_hid_dim', type=int, default=512, help='Dimension of encoder hidden state.')
    parser.add_argument('--dec_hid_dim', type=int, default=512, help='Dimension of decoder hidden state.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    
    # W&B arguments
    parser.add_argument('--wandb_project', type=str, default='g2p-seq2seq', help='W&B project name.')
    parser.add_argument('--wandb_run_name', type=str, default='g2p-evaluation', help='W&B run name.')

    args = parser.parse_args()
    main(args)