import logging
import torch
import numpy as np

import tasks
import argparse

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

parser = argparse.ArgumentParser(description='Parameters for training T5 on PoS')
parser.add_argument('--batch', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
parser.add_argument('--embsize', type=int, default=250, help='size of embeddings')
parser.add_argument('--gradclip', type=int, default=0, help='gradient clipping for training')

args = parser.parse_args()

logger = logging.getLogger(__name__)
device = torch.device('cuda')

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

def train(model, traindata):
    # prepare optimizer and scheduler
    # start training + text
    print("starting the training")
    logger.info("***** Running training *****")
   
    # define the data here
    train_dataloader = DataLoader(traindata, shuffle=False, batch_size=args.batch)

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

    torch.save(model, "pos_model")
    print("finished!")

def evaluate(model, testdata, mappings):
    test_dataloader = DataLoader(testdata, shuffle=False, batch_size=args.batch)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataloader))
    logger.info("  Batch size = %d", args.batch)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # model.train()

    print("printing out device:", device)
    model.to(device)
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2].squeeze()

            #print("checking inputs:", inputs.is_cuda)
            #print("checking attention_mask:", attention_mask.is_cuda)
            #print("checking labels:", labels.is_cuda)
            #print("checking model:", next(model.parameters()).is_cuda)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, lm_labels=labels)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    print("here are the preds:", preds)
    print("here are the out_label_ids:", out_label_ids)




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
    #pos_tokens_id = [label for token in pos_tokens for label in token]
    pos_tokens_id = []
    for token in pos_tokens:
        for label in token:
            if label not in pos_tokens_id:
                pos_tokens_id.append(label)

    # added the special tokens to the dictionary
    special_tokens_dic = {'cls_token': '<s>', 'sep_token': '</s>', 'additional_special_tokens': pos_tokens_id}
    tokenizer.add_special_tokens(special_tokens_dic)

    labels = [tokenizer.encode(sent_tok) for sent_tok in pos_tokens]

    # labels are correct now
    torch_labels = [torch.LongTensor(sent).view(1,-1) for sent in labels]

    padded_labels = []
    for i in torch_labels:
        padded_labels.append(tasks.pad_labels(i, args.embsize))

    # padding and aggregating data
    torch_ids = []
    torch_masks = []
    token_starts = []
    for i in word_tokens:
        subword_id, mask, token_start = tasks.subword_tokenize_to_ids(i, tokenizer, args.embsize)
        #print(subword_id)
        #print(mask)
        #print(token_start)
        #print("shape of ids:", subword_id.shape)
        #print("shape of ids after flatten:", torch.flatten(subword_id).shape)
        torch_ids.append(torch.flatten(subword_id))
        torch_masks.append(torch.flatten(mask))
        token_starts.append(torch.flatten(token_start))

    ### For training
    # got the data and everything, start training
    #dataset = Dataset(torch_ids, torch_masks, padded_labels)
    #model = AutoModelWithLMHead.from_pretrained("t5-small")
    #model.to(device)
    #train(model, dataset)

    ### For evaluating
    # to change: use a different dataset for testing
    dataset = Dataset(torch_ids, torch_masks, padded_labels)
    model = torch.load("pos_model")
    model.to(device)
    evaluate(model, dataset, token_starts)

    print("done!")
