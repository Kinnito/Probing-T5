import logging
import torch
import numpy as np

import tasks
import argparse
import pdb
import csv

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelWithLMHead
from tqdm import tqdm, trange
from transformers import (AdamW, AutoTokenizer)

parser = argparse.ArgumentParser(description='Parameters for training T5 on PoS')
parser.add_argument('--train', dest='mode', action='store_true', help='choose training')
parser.add_argument('--eval', dest='mode', action='store_false', help='choose evaluation')
parser.add_argument('--batch', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
parser.add_argument('--embsize', type=int, default=250, help='size of embeddings')
parser.add_argument('--gradclip', type=int, default=0.25, help='gradient clipping for training')
parser.add_argument('--log', dest='log', action='store_true', help='record data')
parser.add_argument('--no-log', dest='log', action='store_false', help='do not record data')

parser.set_defaults(mode=True)

args = parser.parse_args()

logger = logging.getLogger(__name__)
device = torch.device('cuda')
writer = SummaryWriter()

class Dataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        #print("idx:", index)
        #print(len(self.ids))
        id_ = self.ids[index]
        mask = self.masks[index]
        label = self.labels[index]
        # print("id_: ", id_)
        # print("mask: ", mask)
        # print("label: ", label)
        #pdb.set_trace()

        return id_, mask, label

def train(model, train_dat, dev_dat, dev_mappings, tokenizer):
    # prepare optimizer and scheduler
    # start training + text
    print("starting the training")
    logger.info("***** Running training *****")
   
    # define the data here
    train_dataloader = DataLoader(train_dat, shuffle=False, batch_size=args.batch)

    num_trained_epochs = args.epochs

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss = 0.0
    # logging_loss = 0.0

    model.zero_grad()
    params = [p for n,p in model.named_parameters()]
    optimizer = AdamW(params, lr=args.lr)
    train_iterator = trange(epochs_trained, int(num_trained_epochs), desc="Epoch")

    for idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #model.to(device)
            
            batch = tuple(t.to(device) for t in batch)
            
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2].squeeze()

            # print("inputs size:", inputs.size())
            # print("attention size:", attention_mask.size())
            # print("labels size:", labels.size())

            outputs = model(input_ids=inputs, attention_mask=attention_mask, lm_labels=labels)
            loss = outputs[0]

            #print("outputs:", outputs)

            loss.backward()

            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.gradclip)
            optimizer.step()
            
            # include a scheduler step here if necessary
            model.zero_grad()
            global_step += 1
            
            if (len(epoch_iterator) == step + 1):
                writer.add_scalar('train loss', tr_loss / global_step, global_step)

                # save the model, then run on the development set
                torch.save(model, "pos_model_" + str(idx))
                #print(f"Epoch: {epochs_trained}, Step: {step}, Loss: {tr_loss / global_step}")

                # test on dev set
                dev(model, dev_dat, dev_mappings, tokenizer, idx)

                # load back the original model
                # model = torch.load("pos_model_" + str(idx), map_location="cpu")
                # model.to(device)
        epochs_trained += 1
    
    writer.close()
    torch.save(model, "pos_model")
    print("finished!")

def dev(model, dev_data, mappings, tokenizer, iteration):
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch)
    logger.info("***** Running development *****")
    
    total = 0
    correct = 0
    dev_loss = 0.0
    nb_dev_step = 0

    model.eval()
    f = None
    csvwriter = None
    if args.log:
        f = open('dev_results', 'a')
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

            dev_loss += tmp_dev_loss.item()

            ypred = torch.max(logits.cpu(), dim=2)[1]
            ypred = [tokenizer.convert_ids_to_tokens(s) for s in ypred]
            ytrue = [tokenizer.convert_ids_to_tokens(s) for s in labels]

            ytrue = [item for sublist in ytrue for item in sublist]
            ypred = [item for sublist in ypred for item in sublist]

            correct += sum(y_t==y_p for y_t, y_p in zip(ytrue, ypred))
            total += len(ytrue)
            nb_dev_step += 1

    accuracy = correct / total
    loss = dev_loss / nb_dev_step
    writer.add_scalar('dev loss', loss, iteration)
    writer.add_scalar('accuracy', accuracy, iteration)
    if args.log:
        csvwriter.writerow([loss, accuracy, iteration])

    #print(f"Accuracy: {correct / total}")   


def evaluate(model, testdata, mappings, tokenizer):
    test_dataloader = DataLoader(testdata, shuffle=False, batch_size=args.batch)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataloader))
    logger.info("  Batch size = %d", args.batch)

    total = 0
    correct = 0
    eval_loss = 0.0
    nb_eval_step = 0

    model.eval()

    f = None
    csvwriter = None
    if args.log:
        f = open('results.csv', 'w')
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
                writer.add_scalar('accuracy', accuracy, nb_eval_step)
                if args.log:
                    csvwriter.writerow([loss, accuracy])
                print(f"Step: {nb_eval_step}, Loss: {eval_loss / nb_eval_step}")

    #print(f"Step: {nb_eval_step}, Loss: {eval_loss / nb_eval_step}")
    print(f"Accuracy: {correct / total}")

def prepare_data(tokenizer, word_tokens, pos_tokens):
    word_tokens, pos_tokens = tasks.pos('UD_English-EWT/en_ewt-ud-train.conllu')
    
    pos_tokens_id = []
    for token in pos_tokens:
        for label in token:
            if label not in pos_tokens_id:
                pos_tokens_id.append(label)

    # add special tokens to the dictionary
    special_tokens_dic = {'cls_token': '<s>', 'sep_token': '</s>', 'additional_special_tokens': pos_tokens_id}
    tokenizer.add_special_tokens(special_tokens_dic)

    labels = [tokenizer.encode(sent_tok) for sent_tok in pos_tokens]
    
    # labels are correct now
    torch_labels = [torch.LongTensor(sent).view(1, -1) for sent in labels]

    padded_labels = []
    for i in torch_labels:
        padded_labels.append(tasks.pad_labels(i, args.embsize))

    # padding and aggregating data
    torch_ids = []
    torch_masks = []
    token_starts = []
    for i in word_tokens:
        subword_id, mask, token_start = tasks.subword_tokenize_to_ids(i, tokenizer, args.embsize)
        torch_ids.append(torch.flatten(subword_id))
        torch_masks.append(torch.flatten(mask))
        token_starts.append(torch.flatten(token_start))

    return torch_ids, torch_masks, token_starts, padded_labels

if __name__ == "__main__":
    # word and position tokens
    
    if args.mode:
        print("starting to train")
        word_tokens_train, pos_tokens_train = tasks.pos('UD_English-EWT/en_ewt-ud-train.conllu')

        #tokenizer = AutoTokenizer.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        torch_ids_train, torch_masks_train, torch_token_starts, torch_labels_train = prepare_data(tokenizer, word_tokens_train, pos_tokens_train)

        ### For training
        # got the data, split into train and dev sets

        split = int(0.75 * len(torch_ids_train))

        dataset_train = Dataset(torch_ids_train[:split], torch_masks_train[:split], torch_labels_train[:split])
        dataset_dev = Dataset(torch_ids_train[split:], torch_masks_train[split:], torch_labels_train[split:])
        # not sure what T5ForConditionalGeneration does vs. the other T5 models
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        #model = AutoModelWithLMHead.from_pretrained("t5-small")
        model.to(device)
        train(model, dataset_train, dataset_dev, torch_token_starts[split:], tokenizer)

        print("done!")

    else:
        print("starting to evaluate")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        word_tokens_test, pos_tokens_test = tasks.pos('UD_English-EWT/en_ewt-ud-test.conllu')
        torch_ids_test, torch_masks_test, torch_token_starts, torch_labels_test = prepare_data(tokenizer, word_tokens_test, pos_tokens_test)

        ### For evaluating
        # got the data, start evaluating
        dataset = Dataset(torch_ids_test, torch_masks_test, torch_labels_test)
        model = torch.load("pos_model_9", map_location='cpu')
        model.to(device)
        evaluate(model, dataset, torch_token_starts, tokenizer)

        print("done!")
