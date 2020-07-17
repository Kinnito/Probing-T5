import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import argparse
import pdb

import random
import numpy as np

from tqdm import tqdm, trange
# from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter

# export CUDA_VISIBLE_DEVICES=N
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

writer = SummaryWriter()

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
        # removed the self.embedding part
        #print("checking source:", src)
        embedded = self.dropout(src)
        #print("checking embedded shape:", embedded.shape)
        # embedded = len x batch size x emb dim
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = len x batch size x hid dim * n directions
        # hidden = n layers * n directions x batch size x hid dim
        # cell = n layers * n directions x batch size x hid dim

        #print("hidden at the end of forward:", hidden.shape)
        #print("cell at the end of forward:", cell.shape)
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
        
        #print("embedded:", embedded)
        #print("hidden:", hidden)
        #print("cell:", cell)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        #print("after decoder output")

        # output = len x batch size x hid dim * n directions = 1 x batch size x hid dim
        # hidden = n layers * n directions x batch size x hid dim = n layers x batch size x hid dim
        # cell = n layers * n directions x batch size x hid dim = n layers x batch size x hid dim

        #print("output shape:", output.squeeze(0).shape)
        #pdb.set_trace()
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
        #print("inside of seq2seq forward checking src:", src.shape)
        #print("inside of seq2seq forward checking trg:", trg.shape)
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

def train(model, train, label):
    num_trained_epochs = args.epochs

    global_step = 0
    epochs_trained = 0
    tr_loss = 0.0
    
    params = [p for n,p in model.named_parameters()]

    model.zero_grad()
    #optimizer = AdamW(params, lr=args.lr)
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_iterator = trange(epochs_trained, int(num_trained_epochs), desc="Epoch")

    for idx, _ in enumerate(train_iterator):
        # get the data now
        train_keys = list(train.keys())
        label_keys = list(label.keys())
        for step in tqdm(range(len(list(train.keys())))):
            #print("starting the loop")
        #for step, (d_key, l_key) in enumerate(zip(train.keys(), label.keys())):
            model.train()
            inputs = torch.tensor(train[train_keys[step]][:]).to(device).squeeze()
            labels = torch.tensor(label[label_keys[step]][:]).to(device).squeeze()

            #print("before permute")
            inputs = inputs.permute(1, 0, 2)
            labels = labels.permute(1, 0)

            #print("shape of inputs before put into model:", inputs.shape)
            #print("shape of labels before put into model:", labels.shape)
            #print("after permute")
            model.zero_grad()
            output = model(inputs, labels)

            #print("getting the outputs")
            output = output[1:].view(-1, output.shape[-1])
            labels = labels[1:].reshape(-1)

            #print("after reshape")
            #print("shape of output:", output.shape)
            #print("shape of labels:", labels.shape)
            loss = criterion(output, labels)
            #print("after getting loss")
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) # make it something else

            optimizer.step()

            tr_loss = loss.item()
            global_step += 1
            
            #print("loss value:", tr_loss / global_step)
            writer.add_scalar('train loss', tr_loss / global_step, global_step)
        torch.save(model, "test_seq/model_" + str(idx))

    writer.close()



def evaluate(model, test, label):
    eval_loss = 0.0
    correct = 0
    total = 0
    nb_eval_step = 0
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_keys = list(test.keys())
    label_keys = list(label.keys())

    for step in tqdm(range(len(list(test.keys())))):
        inputs = torch.tensor(test[test_keys[step]][:]).to(device).squeeze()
        labels = torch.tensor(label[label_keys[step]][:]).to(device).squeeze()

        inputs = inputs.permute(1, 0, 2)
        labels = labels.permute(1, 0)
    #for step, (d_key, l_key) in enumerate(zip(test.keys(), label.keys())):
        with torch.no_grad():
            output = model(inputs, labels, 0)
            output = output[1:].view(-1, output.shape[-1])
            #print("shape of output:", output.shape)
            #print("shape of label:", labels.shape)
            labels = labels[1:].reshape(-1)

            loss = criterion(output, labels)
            eval_loss += loss.item()

            ypred = torch.max(output.cpu(), dim=1)[1]

            #print("labels:", labels.cpu())
            #print("ypred:",ypred)
            
            correct += sum(y_t==y_p for y_t, y_p in zip(labels.cpu(), ypred))
            total += len(labels)

            #print("here's correct:", correct)
            #print("here's total:", total)

            nb_eval_step += 1

            writer.add_scalar('test loss', eval_loss / nb_eval_step, nb_eval_step)
            writer.add_scalar('test accuracy', correct / float(total), nb_eval_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train', dest='mode', action='store_true', help='choose training')
    parser.add_argument('--eval', dest='mode', action='store_false', help='choose evalution')

    args = parser.parse_args()

    print("Starting running the file")
    if args.mode:
        print("Starting training")
        f_train = h5py.File('rnn_atten_test/atten_train', 'r')
        f_dev = h5py.File('rnn_atten_test/atten_dev', 'r')

        f_train_label = h5py.File('rnn_atten_test/atten_train_label', 'r')
        f_dev_label = h5py.File('rnn_atten_test/atten_dev_label', 'r')

        # input = number of inputs possible
        # output = number of outputs possible
        # intput can be some kinda large size probably
        INPUT_DIM = 100000
        # output needs to be the vocab size LOL but idk how to get this size
        OUTPUT_DIM = 100000
        ENC_EMB_DIM = 768
        DEC_EMB_DIM = 256
        HID_DIM = 10
        N_LAYERS = 2
        ENC_DROPOUT = 0.5
        DEC_DROPOUT = 0.5

        # for training
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

        model = Seq2Seq(enc, dec, device).to(device)
        model.apply(init_weights)

        train(model, f_train, f_train_label)
   
    else:
        print("Starting evaluation")
        f_test = h5py.File('rnn_atten_test/atten_test', 'r')
        f_test_label = h5py.File('rnn_atten_test/atten_test_label', 'r')

        model = torch.load('test_seq/model_1', map_location='cpu')
        model.to(device)
        evaluate(model, f_test, f_test_label)
