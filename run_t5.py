import logging
import torch
import numpy as np

import tasks

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelWithLMHead
from tqdm import tqdm, trange
from transformers import (AdamW,
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
)

def train(model):
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

    # define the data here
    train_dataloader = None

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
            batch = tuple(t.to(device) for t in batch)
            
            # formatting data here
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0)
            optimizer.step()
            
            # include a scheduler step here if necessary
            model.zero_grad()
            global_step += 1

if __name__ == "__main__":
    # pairs but in word form
    word_tokens = tasks.pos()
    # total list of vocab
    vocab = [word[0] for sent in word_tokens for word in sent]
    labels = [word[1] for sent in word_tokens for word in sent]
    
    # words
    word2idx = {}
    curr = 1
    for i in vocab:
        if i not in word2idx:
            word2idx[i] = curr
            curr += 1

    idx2words = {word2idx[w]: w for w in word2idx}

    # labels
    labels2idx = {}
    curr = 1
    for i in labels:
        if i not in labels2idx:
            labels2idx[i] = curr
            curr += 1

    # pairs but in idx form
    idx_tokens = [[(word2idx[pair[0]], labels2idx[pair[1]]) for pair in sent] for sent in word_tokens]

    # sentences
    sents = [[word[0] for word in sent] for sent in idx_tokens]
    # parts of speech/labels
    speech = [[word[1] for word in sent] for sent in idx_tokens]
    # lengths 
    # sents_lengths = [len(sents[i]) for i in range(len(sents))]
    # lengths = [length for length in sents_lengths]

 
    #sents_tensor = [[torch.LongTensor(w) for w in s] for s in sents]
    #speech_tensor = [[torch.LongTensor(w) for w in s] for s in speech]
    sents_tensor = [torch.LongTensor(w) for w in sents]
    speech_tensor = [torch.LongTensor(w) for w in speech]
    # lengths_tensor = [torch.LongTensor(w) for w in sents_lengths]
    
    # sents = torch.Tensor(sents)
    # speech = torch.Tensor(speech)

    # input: word, attention mask, then PoS token
    # pad first, then figure out attention mask token
    sents = pad_sequence(sents_tensor, batch_first=True, padding_value=0)
    speech = pad_sequence(speech_tensor, batch_first=True, padding_value=0)
    print(sents_tensor)
    print(sents)
    print(sents_tensor != sents)
    input_msk = (sents_tensor != sents.data).unsqueeze(1)
    print(imput_msk)

    model = AutoModelWithLMHead.from_pretrained("t5-small")

    #train(model)
