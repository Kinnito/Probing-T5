import torch
import numpy as np

import tasks
import argparse
import csv
import run_t5 as r
import h5py

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer

parser = argparse.ArgumentParser(description='Parameters for training T5 on PoS with freezed layers')
parser.add_argument('--train', dest='mode', action='store_true', help='choose training')
parser.add_argument('--eval', dest='mode', action='store_false', help='choose evaluation')
parser.add_argument('--batch', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
parser.add_argument('--embsize', type=int, default=250, help='size of embeddings')
parser.add_argument('--gradclip', type=int, default=0.25, help='gradient clipping for training')
parser.add_argument('--log', dest='log', action='store_true', help='record data')
parser.add_argument('--no-log', dest='log', action='store_false', help='do not record data')
parser.add_argument('--ctrl', dest='control', action='store_true', help='control tasks')
parser.add_argument('--nctrl', dest='control', action='store_false', help='probing tasks')

parser.set_defaults(mode=True)

args = parser.parse_args()

device = torch.device('cuda')
writer = SummaryWriter()

# seq2seq parameters
INPUT_DIM = 512
OUTPUT_DIM = 512
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 10
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5


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


def train(model, train_dat, dev_dat, dev_mappings, tokenizer):
    print("starting the training")

    # define data here
    train_dataloader = DataLoader(train_dat, shuffle=False, batch_size=args.batch)
    
    num_trained_epochs = args.epochs

    global_step = 0
    epochs_trained = 0

    tr_loss = 0.0

    model.zero_grad()
    params = [p for n,p in model.named_parameters()]
    # just to have the layer names
    names = [n for n,p in model.named_parameters()]

    # freeze everything but the last layer by setting grad to false
    for idx, param in enumerate(model.parameters()):
        #if idx != len(params) - 3:
        if idx < len(params) - 1:
            #print(names[idx])
            param.requires_grad = False
        else:
            #print("last layer name:", names[idx])
            #print("values of the last layer:", param)
            param.data = torch.rand(param.size()).to(device)
            #print("here are the new params:", param)
    
    #h5 = h5py.File("fixed_dat_nctrl", "w")
    '''
    for idx, param in enumerate(model.parameters()):
        if idx < len(params) - 1:
            continue
        else:
            print("here's the param:", param)
    '''

    optimizer = AdamW(params, lr=args.lr)
    train_iterator = trange(epochs_trained, int(num_trained_epochs), desc="Epoch")

    for idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
        
            batch = tuple(t.to(device) for t in batch)

            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2].squeeze()

            outputs = model(input_ids=inputs, attention_mask=attention_mask, lm_labels=labels)
            #print("here's the len", len(outputs))

            '''
            print(outputs[0].shape)
            print(outputs[1].shape)
            print((outputs[2][0][0]).shape)
            print(outputs[3][0][0].shape)
            print(outputs[4][0][0].shape)
            print("checking shape:", outputs[5].shape)
            print("checking the 3:", len(outputs[3]))
            print("checking the 4:", len(outputs[4]))
            print("checking the 5:", len(outputs[5]))
            print("checking the 6:", len(outputs[6]))
            print("checking the 7:", len(outputs[7]))
            print("length of outputs:", len(outputs))
            print("checking size of -2:", len(outputs[-2]))
            print("checking shape with -2:", outputs[-2][0].shape)
            print("checking size with -1:", len(outputs[-1]))
            print("checking shape with -1:", outputs[-1][1].shape)
            exit()
            '''

            ## assumption: i need outputs[-1][4] to get the decoder's 0th layer
            #dset = h5.create_dataset(str(idx), data=outputs[-1][4].cpu())
            with h5py.File("fixed_dat_nctrl", "a") as h5:
                #dset = h5.create_dataset(str(idx), outputs[-1][4].shape)
                #dset[:] = outputs[-1][4].cpu()
                dset = h5.create_dataset(str(idx), data=outputs[-1][4].cpu())
                print("here's the dataset:", dset)
                print("here's the index:", idx)
                print("extracting the data:", h5[str(idx)])
            h5.close()
            exit()

            '''
            # gonna test for garbage
            garbage, logits = outputs[:2]
            print(torch.max(logits, dim=1)[1])
            print(labels)
            ypred = torch.max(logits.cpu(), dim=2)[1]
            ypred = [tokenizer.convert_ids_to_tokens(s) for s in ypred]
            ytrue = [tokenizer.convert_ids_to_tokens(s) for s in labels]

            ytrue = [item for sublist in ytrue for item in sublist]
            ypred = [item for sublist in ypred for item in sublist]

            print("correct:", sum(y_t==y_p for y_t, y_p in zip(ytrue, ypred)))
            print("total:", len(ytrue))
            #exit()
            # test is done
            '''
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.gradclip)
            optimizer.step()

            model.zero_grad()
            global_step += 1

            if (len(epoch_iterator) == step + 1):
                writer.add_scalar('train loss', tr_loss / global_step, global_step)
                
                torch.save(model, "models_fixed/fixed_pos_model_" + str(idx))

                dev(model, dev_dat, dev_mappings, tokenizer, idx)
                
                print(f'Loss: {tr_loss / global_step}')
        epochs_trained += 1

    print("finished!")

def dev(model, dev_data, mappings, tokenizer, iteration):
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch)

    total = 0
    correct = 0
    dev_loss = 0.0
    nb_dev_step = 0

    model.eval()
    f = None
    csvwriter = None
    if args.log:
        f = open('fixed_dev_results.csv', 'a')
        csvwriter = csv.writer(f)

    idx = 0
    for batch in tqdm(dev_dataloader, desc='Checking model accuracy...'):
        batch = tuple(t.to(device) for t in batch)
        maps = mappings[idx:idx+args.batch]
        idx += args.batch

        with torch.no_grad():
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2].squeeze()

            outputs = model(input_ids=inputs, attention_mask=attention_mask, lm_labels=labels)
            tmp_dev_loss, logits = outputs[:2]
            #print(labels)
            #print(torch.max(logits, dim=2)[1])

            dev_loss += tmp_dev_loss.item()

            ypred = torch.max(logits.cpu(), dim=2)[1]
            ypred = [tokenizer.convert_ids_to_tokens(s) for s in ypred]
            ytrue = [tokenizer.convert_ids_to_tokens(s) for s in labels]

            ytrue = [item for sublist in ytrue for item in sublist]
            ypred = [item for sublist in ypred for item in sublist]

            correct += sum(y_t==y_p for y_t, y_p in zip(ytrue, ypred))
            total += len(ytrue)
            #print("correct:", sum(y_t==y_p for y_t, y_p in zip(ytrue, ypred))) 
            #print("total:", len(ytrue))
            nb_dev_step += 1
            #exit()
    #print("total correct:", correct)
    #print("total total:", total)

    #exit()

    accuracy = correct / total
    loss = dev_loss / nb_dev_step
    writer.add_scalar('dev loss', loss, iteration)
    writer.add_scalar('dev accuracy', accuracy, iteration)
    if args.log:
        csvwriter.writerow([loss, accuracy, iteration])

def evaluate(model, test_data, mappings, tokenizer):
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch)

    total = 0
    correct = 0
    eval_loss = 0.0
    nb_eval_step = 0

    model.eval()

    f = None
    csvwriter = None
    if args.log:
        f = open('fixed_test_results.csv', 'w')
        csvwriter = csv.writer(f)
        csvwriter.writerow(['Loss', 'Accuracy'])

    idx = 0
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        maps = mappings[idx:idx+args.batch]
        idx += args.batch

        with torch.no_grad():
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2].squeeze()

            outputs = model(input_ids=inputs, attention_mask=attention_mask, lm_labels=labels)
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.item()

            ypred = torch.max(logits.cpu(), dim=2)[1]
            ypred = [tokenizer.convert_ids_to_tokens(s) for s in ypred]
            ytrue = [tokenizer.convert_ids_to_tokens(s) for s in labels]

            ytrue = [item for sublist in ytrue for item in sublist]
            ypred = [item for sublist in ypred for item in sublist]

            correct += sum(y_t==y_p for y_t, y_p in zip(ytrue, ypred))
            total += len(ytrue)

            nb_eval_step += 1

            if (nb_eval_step % 5 == 0):
                loss = eval_loss / nb_eval_step
                accuracy = correct / total
                writer.add_scalar('test loss', loss, nb_eval_step)
                writer.add_scalar('test accuracy', accuracy, nb_eval_step)
                if args.log:
                    csvwriter.writerow([loss, accuracy])

if __name__ == "__main__":
   
    if args.mode:
        print("starting to train")
        # train
        word_tokens_train, pos_tokens_train = tasks.pos('UD_English-EWT/en_ewt-ud-train.conllu')
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        ## i want to append pos: - do I include the pos token associated with it?
        if args.control:
            word_tokens_train, pos_tokens_train = tasks.make_control(tokenizer, word_tokens_train, pos_tokens_train, args.embsize)

        torch_ids_train, torch_masks_train, torch_token_starts, torch_labels_train = r.prepare_data(tokenizer, word_tokens_train, pos_tokens_train)

        # data for training
        split = int(0.75 * len(torch_ids_train))
        #dataset_train = Dataset(torch_ids_train[:split], torch_masks_train[:split], torch_labels_train[:split])
        #dataset_dev = Dataset(torch_ids_train[split:], torch_masks_train[split:], torch_labels_train[split:])
        config = T5Config.from_pretrained("t5-small", output_hidden_states=True, output_attentions=True)
        model = T5ForConditionalGeneration.from_pretrained("t5-small", config=config)
        model.to(device)
        #train(model, dataset_train, dataset_dev, torch_token_starts[split:], tokenizer)

        # 100 values test
        dataset_train = Dataset(torch_ids_train[:200], torch_masks_train[:200], torch_labels_train[:200])
        dataset_dev = Dataset(torch_ids_train[200:400], torch_masks_train[200:400], torch_labels_train[200:400])

        train(model, dataset_train, dataset_dev, torch_token_starts[200:400], tokenizer)

        print("done!")

    else:
        print("starting to evaluate")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        word_tokens_test, pos_tokens_test = tasks.pos("UD_English-EWT/en_ewt-ud-test.conllu")
        if args.control:
            word_tokens_test, pos_tokens_test = tasks.make_control(tokenizer, word_tokens_test, pos_tokens_test, args.embsize)
        torch_ids_test, torch_masks_test, torch_token_starts, torch_labels_test = r.prepare_data(tokenizer, word_tokens_test, pos_tokens_test)

        # data for evluating
        dataset = Dataset(torch_ids_test, torch_masks_test, torch_labels_test)
        model = torch.load("models_fixed/fixed_pos_model_9", map_location='cpu')
        model.to(device)
        evaluate(model, dataset, torch_token_starts, tokenizer)

        print("done!")


