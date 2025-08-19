import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse

# --- 1. Model Architecture (MUST be identical to train.py) ---
# It's crucial to have the same model definitions to load the saved state_dict.

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

    # No need for the training-specific forward method here
    # We will create a new method for inference.


class Vocabulary:
    """A simple wrapper to load the vocabulary from a JSON file."""
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

# --- 2. Prediction Function ---

def predict(model, word, source_vocab, target_vocab, device, max_len=50):
    """
    Performs inference on a single word.
    """
    model.eval()  # Set the model to evaluation mode

    # 1. Preprocess the input word
    word = word.lower()
    tokens = [token for token in word]
    
    # Add Start of Sequence (SOS) and End of Sequence (EOS) tokens
    tokens = [source_vocab.index2token[source_vocab.sos_idx]] + tokens + [source_vocab.index2token[source_vocab.eos_idx]]
    
    # Convert tokens to numerical indices
    src_indexes = [source_vocab.to_index(token) for token in tokens]
    
    # Convert to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        # 2. Pass the source through the encoder
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

    # 3. Perform decoding
    # Start with the SOS token
    trg_indexes = [target_vocab.sos_idx]
    
    for i in range(max_len):
        # Get the last predicted token
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        
        # Get the token with the highest probability
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        # Stop if the End of Sequence (EOS) token is predicted
        if pred_token == target_vocab.eos_idx:
            break
            
    # 4. Convert output indices back to phoneme tokens
    trg_tokens = [target_vocab.to_token(i) for i in trg_indexes]
    
    # Return the result, skipping the initial <SOS> token
    return "".join(trg_tokens[1:-1]) # Remove <SOS> and <EOS>


# --- 3. Main Execution Block ---

def main(args):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabularies
    print("Loading vocabularies...")
    if not os.path.exists(args.source_vocab_path) or not os.path.exists(args.target_vocab_path):
        print(f"Error: Vocabulary files not found in '{os.path.dirname(args.source_vocab_path)}'. Please run prepare_data.py first.")
        return
        
    source_vocab = Vocabulary(args.source_vocab_path)
    target_vocab = Vocabulary(args.target_vocab_path)

    # Instantiate model components from saved hyperparameters
    attn = Attention(args.enc_hid_dim, args.dec_hid_dim)
    enc = Encoder(source_vocab.n_tokens, args.emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dropout)
    dec = Decoder(target_vocab.n_tokens, args.emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.dropout, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)

    # Load the trained model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found. Please run train.py first.")
        return
        
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Get word from user
    if args.word:
        word_to_predict = args.word
    else:
        # Interactive mode if no word is provided
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            word_to_predict = input("Enter a word to transcribe > ")
            if word_to_predict.lower() == 'quit':
                break
            prediction = predict(model, word_to_predict, source_vocab, target_vocab, device)
            print(f"'{word_to_predict}' -> '{prediction}'\n")
        return

    # Perform prediction
    prediction = predict(model, word_to_predict, source_vocab, target_vocab, device)

    # Print result
    print("\n--- G2P Prediction ---")
    print(f"Grapheme (Input):  '{word_to_predict}'")
    print(f"Phoneme (Output):  '{prediction}'")
    print("----------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform G2P inference using a trained Seq2Seq model.')
    
    # Path arguments
    parser.add_argument('--model_path', type=str, default='g2p_model.pt', help='Path to the saved .pt model file.')
    parser.add_argument('--source_vocab_path', type=str, default='processed_data/source_vocab.json', help='Path to source vocabulary.')
    parser.add_argument('--target_vocab_path', type=str, default='processed_data/target_vocab.json', help='Path to target vocabulary.')
    
    # Input word
    parser.add_argument('--word', type=str, help='A single word to transcribe. If not provided, enters interactive mode.')
    
    # Model hyperparameters (must match the trained model)
    parser.add_argument('--emb_dim', type=int, default=256, help='Dimension of character embeddings.')
    parser.add_argument('--enc_hid_dim', type=int, default=512, help='Dimension of encoder hidden state.')
    parser.add_argument('--dec_hid_dim', type=int, default=512, help='Dimension of decoder hidden state.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')

    args = parser.parse_args()
    main(args)