import logging
import torch
import numpy as np

import tasks

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoModelWithLMHead
from tqdm import tqdm, trange
from transformers import (AdamW,
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
)

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

def train(model, traindata, mappings):
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda:0')

    ## get bach size
    ## get training data

    ## get number of training epochs

    # prepare optimizer and scheduler


    # start training + text
    print("starting the training")
    logger.info("***** Running training *****")
    #logger.info("  Num examples = %d",)
    
    count = 0
    for i, j in enumerate(traindata):
        print(i)
        print(j)
        if (count == 20): break
        count += 1
        #print("shape of id:", i.shape)
        #print("shape of mask:", j.shape)

    # define the data here
    train_dataloader = DataLoader(traindata, shuffle=False, batch_size=10)

    num_trained_epochs = 100

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss = 0.0
    # logging_loss = 0.0

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_trained_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            
            #batch = tuple(t.to(device) for t in batch)
            
            print(batch)
            # formatting data here
            #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2]

            print("shape of inputs:", inputs.shape)
            # do I have to encode first?

            encoding = tokenizer.encode(input_ids=inputs, attention_mask=attention_mask, inputs_embeds=labels)


            outputs = model(input_ids=encoding, attention_mask=attention_mask, lm_labels=labels)
            loss = outputs[0]

            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0)
            optimizer.step()
            
            # include a scheduler step here if necessary
            model.zero_grad()
            global_step += 1


# few design choices to change:
#   use collate_fn
#   pad using tokenizer.pad_token_id
#   pad both the data and label at the same time
#   question: do I use the tokenizer for both the data and labels?
if __name__ == "__main__":
    # word and position tokens
    word_tokens, pos_tokens = tasks.pos()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # prepare the labels
    pos_tokens_id = [label for token in pos_tokens for label in token]
    
    #label2idx = {label:i for i, label in enumerate(pos_tokens_id)}
    label2idx = {}
    idx = 1
    for i, label in enumerate(pos_tokens_id):
        if label not in label2idx:
            label2idx[label] = idx
            idx += 1

    idx2label = {i:label for i, label in enumerate(label2idx)}
    labels = [[label2idx[i] for i in sent] for sent in pos_tokens]
    
    # labels are correct now
    torch_labels = [torch.LongTensor(sent).view(1,-1) for sent in labels]

    padded_labels = []
    for i in torch_labels:
        padded_labels.append(tasks.pad_labels(i))

    print(padded_labels)

    # padding and aggregating data
    torch_ids = []
    torch_masks = []
    token_starts = []
    for i in word_tokens:
        subword_id, mask, token_start = tasks.subword_tokenize_to_ids(i, tokenizer)
        #print(subword_id)
        #print(mask)
        #print(token_start)
        #print("shape of ids:", subword_id.shape)
        #print("shape of ids after flatten:", torch.flatten(subword_id).shape)
        torch_ids.append(torch.flatten(subword_id))
        torch_masks.append(torch.flatten(mask))
        token_starts.append(torch.flatten(token_start))

    print("len of ids:", len(torch_ids))
    print("len of labels:", len(torch_labels))


    # got the data and everything, start training
    dataset = Dataset(torch_ids, torch_masks, padded_labels)

    model = AutoModelWithLMHead.from_pretrained("t5-small")

    train(model, dataset, token_starts)

    print("done!")
