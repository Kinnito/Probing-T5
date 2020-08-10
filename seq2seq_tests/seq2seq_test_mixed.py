import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import argparse
import pdb
import time
import math

import random
import numpy as np

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.dataset import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

# export CUDA_VISIBLE_DEVICES=N
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

writer = SummaryWriter('../runs')

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

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.ids) for cur_dataset in dataset.datasets])

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)
    
    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size

        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        sampler_iterators[i] = sampelrs_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    final_samples_list.extend(cur_samples)


        return iter(final_samples_list)

# question - do I need to permute?
def train(model, train_dat_pos, train_label_pos, dev_dat_pos, dev_label_pos, train_dat_ner, train_label_ner, dev_dat_ner, dev_label_ner):
    print("at the beginning of train")
    train_data_pos = POSDataset(train_dat_pos, train_label_pos)
    train_data_ner = NERDataset(train_dat_ner, train_label_ner)

    concat_dataset = ConcatDataset([train_data_pos, train_data_ner])
    #train_sampler = RandomSampler(train_data)

    # add a batch size here?
    train_dataloader = DataLoader(concat_dataset, sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=1), batch_size=1, shuffle=False)

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
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()

            #print("batch:", batch)
            #print("batch shape:", len(batch))
            #print("first batch shape:", batch[0].shape)
            #print("second batch shape:", batch[1].shape)

            inputs = torch.tensor(batch[0]).to(device).squeeze()
            labels = torch.tensor(batch[1]).to(device).squeeze()
            #print("shape of inputs:", inputs.shape)
            #print("shape of labels:", labels.shape)


            model.zero_grad()
            start = time.time()
            output = model(inputs, labels)
            end = time.time()
            #print("elapsed time:", end-start)
            #print("shape out output:", output.shape)
            #print("shape of labels:", labels.shape)
            output = output[1:].view(-1, output.shape[-1])
            labels = labels[1:].reshape(-1)

            #print(output)
            #print(labels)
            
            loss = criterion(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) # make it something else

            optimizer.step()

            tr_loss = loss.item()
            global_step += 1
            
            if len(epoch_iterator) == step + 1:
                writer.add_scalar('train loss', tr_loss / global_step, global_step)
                torch.save(model, "models/pos_model_" + str(idx))
                
                dev(model, dev_dat, dev_label, idx)
        epochs_trained += 1
    writer.close()

def dev(model, dev_dat, label, iteration):
    dev_data = POSDataset(dev_dat, label)
    dev_sampler = RandomSampler(dev_data)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler)

    dev_loss = 0.0
    correct = 0
    total = 0
    nb_dev_step = 0

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for batch in tqdm(dev_dataloader, desc='Checking model accuracy...'):
        with torch.no_grad():
            inputs = torch.tensor(batch[0]).to(device).squeeze()
            labels = torch.tensor(batch[1]).to(device).squeeze()
            
            output = model(inputs, labels)
        
            output = output[1:].view(-1, output.shape[-1])
            labels = labels[1:].reshape(-1)

            loss = criterion(output, labels)

            dev_loss += loss.item()

            ypred = torch.max(output.cpu(), dim=1)[1]

            labels = labels.cpu()
            correct += sum(y_t==y_p for y_t, y_p in zip(labels, ypred))
            total += len(labels)

            nb_dev_step += 1

    accuracy = correct / float(total)
    loss = dev_loss / nb_dev_step
    writer.add_scalar('dev loss', loss, iteration)
    writer.add_scalar('dev accuracy', accuracy, iteration)
            

def evaluate(model, test, label):
    test_data = POSDataset(test, label)
    test_sampler = RandomSampler(test_data)

    test_dataloader = DataLoader(test_data, sampler=test_sampler)

    eval_loss = 0.0
    correct = 0
    total = 0
    nb_eval_step = 0

    model.eval()

    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = torch.tensor(batch[0]).to(device).squeeze()
            labels = torch.tensor(batch[1]).to(device).squeeze()

            output = model(inputs, labels, 0)

            output = output[1:].view(-1, output.shape[-1])
            labels = labels[1:].reshape(-1)

            loss = criterion(output, labels)
            eval_loss += loss.item()

            ypred = torch.max(output.cpu(), dim=1)[1]

            labels = labels.cpu() 
            correct += sum(y_t==y_p for y_t, y_p in zip(labels, ypred))
            total += len(labels)

            nb_eval_step += 1

            writer.add_scalar('test loss', eval_loss / nb_eval_step, nb_eval_step)
            writer.add_scalar('test accuracy', correct / float(total), nb_eval_step)

class POSDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        id_ = self.ids[index]
        label_ = self.labels[index]
        
        return id_, label_

class NERDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        id_ = self.ids[index]
        label_ = self.labels[index]

        return id_, label_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train', dest='mode', action='store_true', help='choose training')
    parser.add_argument('--eval', dest='mode', action='store_false', help='choose evalution')

    args = parser.parse_args()

    print("Starting running the file")
    if args.mode:
        print("Starting training")

        f_train_pos = h5py.File('bert_embeddings/atten_train_pos', 'r')
        f_dev_pos = h5py.File('bert_embeddings/atten_dev_pos', 'r')

        f_train_ner = h5py.File('bert_embeddings/atten_train_ner', 'r')
        f_dev_ner = h5py.File('bert_embeddings/atten_dev_ner', 'r')

        f_train_label_pos = h5py.File('bert_embeddings/atten_train_label_pos', 'r')
        f_dev_label_pos = h5py.File('bert_embeddings/atten_dev_label_pos', 'r')

        f_train_label_ner = h5py.File('bert_embeddings/atten_train_label_ner', 'r')
        f_dev_label_ner = h5py.File('bert_embeddings/atten_dev_label_ner', 'r')

        
        train_data_pos = [f_train_pos[key][()] for key in f_train_pos.keys()]
        train_label_pos = [f_train_label_pos[key][()] for key in f_train_label_pos.keys()]

        train_data_ner = [f_train_ner[key][()] for key in f_train_ner.keys()]
        train_label_ner = [f_train_label_ner[key][()] for key in f_train_label_ner.keys()]

        dev_data_pos = [f_dev_pos[key][()] for key in f_dev_pos.keys()]
        dev_label_pos = [f_dev_label_pos[key][()] for key in f_dev_label_pos.keys()]

        dev_data_ner = [f_dev_ner[key][()] for key in f_dev_ner.keys()]
        dev_label_ner = [f_dev_label_ner[key][()] for key in f_dev_label_ner.keys()]

        '''
        train_data_pos = []
        train_label_pos = []
        train_data_pos.append(f_train_pos['16'][()])
        train_data_pos.append(f_train_pos['32'][()])
        train_label_pos.append(f_train_label_pos['16'][()])
        train_label_pos.append(f_train_label_pos['32'][()])

        train_data_ner = []
        train_label_ner = []
        train_data_ner.append(f_train_ner['16'][()])
        train_data_ner.append(f_train_ner['32'][()])
        train_label_ner.append(f_train_label_ner['16'][()])
        train_label_ner.append(f_train_label_ner['32'][()])

        dev_data_pos = None
        dev_label_pos = None
        dev_data_ner = None
        dev_label_ner = None
        '''

        # input = number of inputs possible
        # output = number of outputs possible
        # intput can be some kinda large size probably
        INPUT_DIM = 50000
        # output needs to be the vocab size LOL but idk how to get this size
        OUTPUT_DIM = 50000
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

        # change this
        train(model, train_data_pos, train_label_pos, dev_data_pos, dev_label_pos, train_data_ner, train_label_ner, dev_data_ner, dev_label_ner)
   
    else:
        print("Starting evaluation")
        f_test_pos = h5py.File('bert_embeddings/atten_test_pos', 'r')
        f_test_label_pos = h5py.File('bert_embeddings/atten_test_label_pos', 'r')

        f_test_ner = h5py.File('bert_embeddings/atten_test_ner', 'r')
        f_test_label_ner = h5py.File('bert_embeddings/atten_test_label_ner', 'r')

        test_data_pos = [f_test_pos[key][()] for key in f_test_pos.keys()]
        test_label_pos = [f_test_label_pos[key][()] for key in f_test_label_pos.keys()]

        test_data_ner = [f_test_ner[key][()] for key in f_test_ner.keys()]
        test_label_ner = [f_test_label_ner[key][()] for key in f_test_label_ner.keys()]
        
        model = torch.load('models/pos_model_0', map_location='cpu')
        model.to(device)

        # also change this
        evaluate(model, test_data, test_label)
