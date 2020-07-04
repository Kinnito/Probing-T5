import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import argparse

import random
import numpy as np

from tqdm import tqdm, trange
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset

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

def train(model, train, dev):
    train_dataloader = DataLoader(train, shuffle=False, batch_size=args.batch)

    num_trained_epochs = args.epochs

    global_step = 0
    epochs_trained = 0
    tr_loss = 0.0

    params = [p for n,p in model.named_parameters()]

    model.zero_grad()
    optimizer = AdamW(params, lr=args.lr)
    train_iterator = trange(epochs_trained, int(num_trained_epochs), desc="Epoch")

    for idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2]

            print(intputs)
            print(attention_mask)
            print(labels)
            print(inputs.shape)
            print(attention_mask.shape)
            print(labels.shape)
            exit()

# dataset to hold ids, masks, labels
class Dataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        mask = self.masks[index]
        label = self.labels[index]

        return id_, mask, label

# gets the data from h5py form into numpy format
def getData(h5):
    raise NotImplementedException

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

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

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)

    # need to convert these h5py into something kinda useful
    # f_train = [hf[ref] for ref in n1[:]]

    f_train = h5py.File('atten_train', 'r')
    f_dev = h5py.File('atten_dev', 'r')
    f_test = h5py.File('atten_test', 'r')

    f_train_atten = h5py.File('atten_train_atten', 'r')
    f_dev_atten = h5py.File('atten_dev_atten', 'r')
    f_test_atten = h5py.File('atten_test_atten', 'r')

    f_train_label = h5py.File('atten_train_label', 'r')
    f_dev_label = h5py.File('atten_dev_label', 'r')
    f_test_label = h5py.File('atten_test_label', 'r')

    #train_data = np.fromfile('atten_train', dtype=float)
    train_data = [f_train[key][:] for key in f_train.keys()]
    print(train_data)
    print(train_data.shape)
    #dev_data = [f_dev[key][()] for key in f_dev.keys()]
    #test_data = [f_dev[key][()] for key in f_test.keys()]



    train_dataset = Dataset(f_train, f_train_atten, f_train_label)
    dev_dataset = Dataset(f_dev, f_dev_atten, f_dev_label)
    test_dataset = Dataset(f_test, f_test_atten, f_test_label)

    train(model, train_dataset, dev_dataset)


    print(model)
