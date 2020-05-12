import logging
import torch
import numpy as np

import tasks
import argparse
import pdb

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
parser.add_argument('--feature', default=True, help='choose training or evaluation')
parser.add_argument('--batch', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
parser.add_argument('--embsize', type=int, default=250, help='size of embeddings')
parser.add_argument('--gradclip', type=int, default=0, help='gradient clipping for training')

args = parser.parse_args()

logger = logging.getLogger(__name__)
device = torch.device('cuda:0')

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

    torch.save(model, "pos_model")
    print("finished!")

def evaluate(model, testdata, mappings, tokenizer):
    test_dataloader = DataLoader(testdata, shuffle=False, batch_size=args.batch)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataloader))
    logger.info("  Batch size = %d", args.batch)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.eval()

    #preds = []
    #out_label_ids = []
    correct = 0
    incorrect = 0
    total = 0

    print("size of test_dataloader:", len(testdata))
    print("size of mappings:", len(mappings))
    print("vocabulary:", tokenizer.get_vocab())
    vocab = tokenizer.get_vocab()
   
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

            for out, label, m, mask in zip(logits, labels, maps, attention_mask):
                result = [torch.max(x.cpu(), 0)[1].item() for x in out]
                result = np.array(result)
                #print("shape of out:", out.shape)
                #print("shape of label:", label.shape)
                #print("result of result:", result)
                vals = np.where(m.cpu().numpy() != 0)[0]
                print("showing the mapping:", m)
                print("showing the mask", mask)
                
                #print(vals)
                #print("shape of vals:", vals.shape)
                pos = result[vals]
                lab = label[vals]
               
                #for p, l in zip(pos, lab):
                #    print(f"predicted: {p}, actual: {l}")  
                
                #i = np.where(mask.cpu().numpy() != 0)[0]
                #print("values of i:", i)
                # I need to fix this label mapping
                #print("mask values:", mask.cpu().numpy())
                #lab = label[mask.cpu().numpy()]
                #print("here are the wanted positions:", pos)
                #print("mask values:", mask)
                #print("here are all the labels:", label)
                #print("here are the wanted labels:", lab)

        # preds.append(logits.cpu())
        # out_label_ids.append(labels.cpu())
        #print("shape of logits:", logits.shape)
        #print("shape of labels", labels.shape)
       
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


# few design choices to change:
#   use collate_fn
#   pad using tokenizer.pad_token_id
#   pad both the data and label at the same time
#   question: do I use the tokenizer for both the data and labels?
if __name__ == "__main__":
    # word and position tokens

    
    if args.feature:
        print("starting to train")
        word_tokens_train, pos_tokens_train = tasks.pos('UD_English-EWT/en_ewt-ud-train.conllu')

        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        torch_ids_train, torch_masks_train, torch_token_starts, torch_labels_train = prepare_data(tokenizer, word_tokens_train, pos_tokens_train)

        ### For training
        # got the data, start training
        dataset = Dataset(torch_ids_train, torch_masks_train, torch_labels_train)
        model = AutoModelWithLMHead.from_pretrained("t5-small")
        model.to(device)
        train(model, dataset)

        print("done!")

    else:
        print("starting to evaluate")
        word_tokens_test, pos_tokens_test = tasks.pos('UD_English-EWT/en_ewt-ud-test.conllu')
        torch_ids_test, torch_masks_test, torch_token_starts, torch_labels_test = prepare_data(tokenizer, word_tokens_test, pos_tokens_test)

        ### For evaluating
        # got the data, start evaluating
        dataset = Dataset(torch_ids_test, torch_masks_test, torch_labels_test)
        model = torch.load("pos_model", map_location='cpu')
        model.to(device)
        evaluate(model, dataset, torch_token_starts, tokenizer)

        print("done!")
