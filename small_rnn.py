import torch
import torch.nn as nn
import torch.optim as optim
import h5py

import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = len x batch size
        embedded = self.dropout(self.embedding(src))
        # embedded = len x batch size x emb dim
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = len x batch size x hid dim * n directions
        # hidden = n layers * n directions x batch size x hid dim
        # cell = n layers * n directions x batch size x hid dim
        return hidden, cell

# decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden, cell):
        # inp = batch size
        # hidden = n layers * n directions x batch size x hid dim = n layers x batch size x hid dim
        # cell = n layers * n directions x batch size x hid dim = n layers x batch size x hid dim

        inp = inp.unsqueeze(0)
        # inp = 1 x batch size

        embedded = self.dropout(self.embedding(inp))
        # embedded = 1 x batch size x emb dim

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = len x batch size x hid dim * n directions = 1 x batch size x hid dim
        # hidden = n layers * n directions x batch size x hid dim = n layers x batch size x hid dim
        # cell = n layers * n directions x batch size x hid dim = n layers x batch size x hid dim

        prediction = self.fc_out(output.squeeze(0))

        # prediction = batch size x output dim

        return prediction, hidden, cell


# seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
                "Hidden dimensions of the encoder and decoder most be equal!"
        assert encoder.n_layers == decoder.n_layers, \
                "Number of layers of the encoder and decoder must be equal!"

        def forward(self, src, trg, teacher_forcing_ratio = 0.5):
            # src = len x batch size
            # trg = len x batch size
        
            trg_len, batch_size = trg.shape 
            trg_vocab_size = self.decoder.output_dim

            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
            hidden, cell = self.encoder(src)

            inp = trg[0, :]
            
            for t in range(1, trg_len):
                output, hidden, cell = self.decoder(inp, hidden, cell)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top = output.argmax(1)
                inp = trg[t] if teacher_force else top

            return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

if __name__ == "__main__":
    # input = number of inputs possible
    # input_dim == len(src.vocab)
    # output = number of outputs possible
    # output_dim == len(trg.vocab)
    INPUT_DIM = 512
    OUTPUT_DIM = 512
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 10
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    print("hello")
    with h5py.File('fixed_dat_nctrl', 'r') as f:
        print("checking dat")
        print(f)
        print(f['0'])

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    print(model)
