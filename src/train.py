import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import json
import os
import argparse
import random
import time
import math

# --- 1. Setup & Configuration ---

# For reproducibility
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Vocabulary:
    """A simple wrapper to load the vocabulary from the JSON file."""
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.token2index = data['token2index']
        self.index2token = {int(k): v for k, v in data['index2token'].items()}
        self.n_tokens = data['n_tokens']
        self.pad_idx = self.token2index['<PAD>']
        self.sos_idx = self.token2index['<SOS>']
        self.eos_idx = self.token2index['<EOS>']
        self.unk_idx = self.token2index['<UNK>']

    def to_index(self, token):
        """Returns the index for a given token, defaulting to the UNK_TOKEN index."""
        return self.token2index.get(token, self.unk_idx)

    def to_token(self, index):
        """Returns the token for a given index."""
        return self.index2token.get(index, '<UNK>')

# --- 2. Dataset and Dataloader ---

class G2PDataset(Dataset):
    """Custom PyTorch Dataset for G2P data."""
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Returns a single pair of [source, target] sequences
        return self.pairs[idx]

def collate_fn(batch, source_vocab, target_vocab, device):
    """
    Custom collate function to process batches of sequences.
    - Converts tokens to indices.
    - Adds SOS/EOS tokens.
    - Pads sequences to the same length within a batch.
    - Moves tensors to the specified device.
    """
    sources, targets = zip(*batch)

    # Convert source sequences to indexed tensors
    source_tensors = []
    source_lengths = []
    for seq in sources:
        indexed_seq = [source_vocab.sos_idx] + [source_vocab.to_index(token) for token in seq] + [source_vocab.eos_idx]
        source_tensors.append(torch.tensor(indexed_seq, dtype=torch.long))
        source_lengths.append(len(indexed_seq))

    # Convert target sequences to indexed tensors
    target_tensors = []
    for seq in targets:
        indexed_seq = [target_vocab.sos_idx] + [target_vocab.to_index(token) for token in seq] + [target_vocab.eos_idx]
        target_tensors.append(torch.tensor(indexed_seq, dtype=torch.long))

    # Pad sequences
    padded_sources = nn.utils.rnn.pad_sequence(source_tensors, batch_first=True, padding_value=source_vocab.pad_idx)
    padded_targets = nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=target_vocab.pad_idx)

    return (
        padded_sources.to(device),
        torch.tensor(source_lengths, dtype=torch.long).to(device), # For packing sequences
        padded_targets.to(device)
    )

# --- 3. Model Architecture (Seq2Seq with Attention) ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [batch size, src len]
        embedded = self.dropout(self.embedding(src)) # [batch size, src len, emb dim]
        
        # Pack sequence to handle padding efficiently
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs = [batch size, src len, enc_hid_dim * 2]
        
        # The decoder is not bidirectional, so we need to bridge the hidden state dimensions
        # hidden/cell = [2, batch size, enc_hid_dim] -> [batch size, dec_hid_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec_hid_dim]
        # encoder_outputs = [batch size, src len, enc_hid_dim * 2]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch size, src len, dec_hid_dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2) # [batch size, src len]
        
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
        # input = [batch size]
        # hidden/cell = [batch size, dec_hid_dim]
        # encoder_outputs = [batch size, src len, enc_hid_dim * 2]
        input = input.unsqueeze(0) # [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs).unsqueeze(1) # [batch size, 1, src len]
        
        weighted = torch.bmm(a, encoder_outputs) # [batch size, 1, enc_hid_dim * 2]
        weighted = weighted.permute(1, 0, 2) # [1, batch size, enc_hid_dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim=2) # [1, batch size, (enc_hid_dim*2)+emb_dim]
        
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

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        
        input = trg[:, 0] # Start with <SOS> token
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            input = trg[:, t] if teacher_force else top1
            
        return outputs.permute(1, 0, 2) # [batch size, trg len, output dim]

# --- 4. Training and Evaluation ---

def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        src, src_len, trg = batch
        
        optimizer.zero_grad()
        
        output = model(src, src_len, trg)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim) # Ignore <SOS> token
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src, src_len, trg = batch
            
            output = model(src, src_len, trg, 0) # Turn off teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# --- 5. Main Execution Block ---

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabularies
    print("Loading vocabularies...")
    source_vocab = Vocabulary(os.path.join(args.data_dir, 'source_vocab.json'))
    target_vocab = Vocabulary(os.path.join(args.data_dir, 'target_vocab.json'))
    
    # Create datasets
    print("Loading data...")
    train_dataset = G2PDataset(os.path.join(args.data_dir, 'train.json'))
    val_dataset = G2PDataset(os.path.join(args.data_dir, 'val.json'))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, source_vocab, target_vocab, device))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, source_vocab, target_vocab, device))

    # Initialize model components
    print("Initializing model...")
    attn = Attention(args.enc_hid_dim, args.dec_hid_dim)
    enc = Encoder(source_vocab.n_tokens, args.emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dropout)
    dec = Decoder(target_vocab.n_tokens, args.emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dropout, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(initialize_weights)

    # Optimizer and Loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_idx)

    best_valid_loss = float('inf')
    
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print("\nStarting training...")

    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.clip)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | **Model Saved**")
        else:
             print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")

        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a G2P Seq2Seq model.')
    
    # Data and Model Paths
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Directory containing processed data files.')
    parser.add_argument('--model_save_path', type=str, default='models/g2p_model.pt', help='Path to save the best model.')
    
    # Model Hyperparameters
    parser.add_argument('--emb_dim', type=int, default=256, help='Dimension of character embeddings.')
    parser.add_argument('--enc_hid_dim', type=int, default=512, help='Dimension of encoder hidden state.')
    parser.add_argument('--dec_hid_dim', type=int, default=512, help='Dimension of decoder hidden state.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping value.')
    
    args = parser.parse_args()
    main(args)