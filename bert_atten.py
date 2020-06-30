import tasks
import torch
import h5py
import argparse

from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

### class to run BERT and put the attention embeddings into a h5py file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def evaluate(f, data, model):
    # load the data
    dataloader = DataLoader(data, shuffle=False, batch_size=args.batch)

    # printing out the parameters to test
    '''
    for n,p in model.named_parameters():
        print(n)
    '''

    model.eval()
    idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        #maps = mappings[idx:idx+args.batch] # 1 for now, change batch size if needed
        idx += args.batch # same here

        with torch.no_grad():
            inputs = batch[0]
            attention_mask = batch[1]
            labels = batch[2].squeeze()

            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            # want the very last layer
            atten = outputs[-1][11]
            dset = f.create_dataset(str(idx), data=outputs[-1][11].cpu())
    f.close()


def prepare_data(tokenizer, word_tokens, pos_tokens):
    pos_tokens_id = []
    for token in pos_tokens:
        for label in token:
            if label not in pos_tokens_id:
                pos_tokens_id.append(label)

    special_tokens_dic = {'cls_token': '[CLS]', 'sep_token': '[SEP]', 'additional_special_tokens': pos_tokens_id}
    tokenizer.add_special_tokens(special_tokens_dic)

    labels = [tokenizer.encode(sent_tok) for sent_tok in pos_tokens]

    torch_labels = [torch.LongTensor(sent).view(1, -1) for sent in labels]

    padded_labels = []
    for i in torch_labels:
        padded_labels.append(tasks.pad_labels(i, args.embsize))

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
    # init argparse with following arg parse items
    parser = argparse.ArgumentParser()
    parser.add_argument('--embsize', type=int, default=400)
    parser.add_argument('--batch', type=int, default=1)

    args = parser.parse_args()

    # init the file to save stuff
    f_train = h5py.File('atten_train', 'w')
    f_dev = h5py.File('atten_dev', 'w')
    f_test = h5py.File('atten_test', 'w')

    print("starting up")

    # set up config and model and tokenizer
    #config = BertConfig.from_pretrained("bert-base-cased", output_attention=True)
    model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # set up the data
    word_tokens, pos_tokens = tasks.pos('UD_English-EWT/en_ewt-ud-train.conllu')
    torch_ids, torch_masks, torch_token_starts, torch_labels = prepare_data(tokenizer, word_tokens, pos_tokens)

    word_tokens_test, pos_tokens_test = tasks.pos('UD_English-EWT/en_ewt-ud-test.conllu')
    torch_ids_test, torch_masks_test, torch_token_starts_test, torch_labels_test = prepare_data(tokenizer, word_tokens_test, pos_tokens_test)

    # split for train/dev/eval
    split = int(0.75 * len(torch_ids))

    dataset_train = Dataset(torch_ids[:split], torch_masks[:split], torch_labels[:split])
    dataset_dev = Dataset(torch_ids[split:], torch_masks[split:], torch_labels[split:])
    dataset_test = Dataset(torch_ids_test, torch_masks_test, torch_labels_test)
    
    # start getting the outputs
    evaluate(f_train, dataset_train, model)
    evaluate(f_dev, dataset_dev, model)
    evaluate(f_test, dataset_test, model)

    # close the file and exit
    f_train.close()
    f_dev.close()
    f_test.close()
