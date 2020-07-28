from __future__ import print_function
import nltk
import numpy as np
import random
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from nltk.corpus import conll2000
from nltk.util import LazyConcatenation, LazyMap
from nltk.corpus.reader import ConllCorpusReader

# appending a path every time because it won't stick
nltk.data.path.append('/data/limill01/Probing-T5/nltk_data/')

device = torch.device('cuda')

pos_list = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

# get the dataset into the correct form and append "chunking:" to the front
def chunking():
    train_sents = conll2000.chunked_sents('train.txt')
    train_data = [[w for w,t,c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]
    train_label = [[c for w,t,c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents] 
  
    # now append chunking to the front of each group/string
    return train_data, train_label

def ner(filename):  
    class nerConllReader(ConllCorpusReader):
        def iob_words(self, fileids=None, tagset=None, column="ne"):
            """
            :return: a list of word/tag/IOB tuples
            :rtype: list(tuple)
            :param fileids: the list of fileids that make up this corpus
            :type fileids: None or str or list
            """
            self._require(self.WORDS, self.POS, self.NE)
            def get_iob_words(grid):
                return self._get_iob_words(grid, tagset, column)
            return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

        def _get_iob_words(self, grid, tagset=None, columns="ne"):
            #print("pos here", self._colmap['pos'])
            #print("length of grid:", len(grid))
            pos_tags = self._get_column(grid, self._colmap['pos'])
            if tagset and tagset != self._tagset:
                pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
            return list(zip(self._get_column(grid, self._colmap['words']), pos_tags,
                self._get_column(grid, self._colmap[columns])))

    f = filename[filename.rindex('/') + 1:]
    directory = filename[:filename.rindex('/') + 1]
    
    #reader = nerConllReader(r'nltk_data/corpora/conll2003/', r'eng.train', ('words', 'pos', 'ne'))
    reader = nerConllReader(directory, f, ('words', 'pos', 'ne'))
    grid = reader._grids()
 
    sentence = []
    label = []
    for sent in grid:
        s = []
        l = []
        for word in sent:
            s.append(word[0])
            l.append(word[3])
        if len(s) > 0:
            sentence.append(s)
            label.append(l)
    return sentence, label

def pos(filename):
    # filename = 'UD_English-EWT/en_ewt-ud-train.conllu'
    sents = []
    pos_tokens = []

    with open(filename) as f:
        sent = []
        token = []
        for line in f:
            stripped = line.split()
            if line.startswith('#'):
                continue
            elif len(stripped) == 0:
                sents.append(sent)
                pos_tokens.append(token)
                sent = []
                token = []
            else:
                sent.append(stripped[1].lower())
                token.append(stripped[4])

    return sents, pos_tokens

# ------------------------------ helper functions
def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

def subword_tokenize(tokens, tokenizer):
    subwords = list(map(tokenizer.tokenize, tokens))
    #subwords = [[s.replace('▁', '') for s in x] for x in subwords]
    #subwords = [[s for s in x if s != ''] for x in subwords]
    # deal with the weird tokenization issues
    for i in subwords:
        #print("not in here")
        if i[0] == '▁':
            #print("in here")
            i.pop(0)
            i[0] = '▁' + i[0]
            #print(i[0])
    subword_lengths = list(map(len, subwords))
    subwords = ["<s>"] + list(flatten(subwords)) + ["</s>"]
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    return subwords, token_start_idxs

def subword_tokenize_to_ids(tokens, tokenizer, emb_size):
    subwords, token_start_idxs = subword_tokenize(tokens, tokenizer)
    subword_ids, mask = convert_tokens_to_ids(subwords, tokenizer, emb_size)
    token_starts = torch.zeros(1, emb_size).to(subword_ids)
    token_starts[0, token_start_idxs] = 1
    return subword_ids, mask, token_starts


def convert_tokens_to_ids(tokens, tokenizer, emb_size):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor([token_ids]).to(device=device)
    padded_ids = torch.zeros(1, emb_size).to(ids)
    padded_ids[0, :ids.size(1)] = ids
    mask = torch.zeros(1, emb_size).to(ids)
    mask[0, :ids.size(1)] = 1
    return padded_ids, mask

def pad_labels(labels, emb_size):
    padded_labels = torch.zeros(1, emb_size).to(labels)
    padded_labels[0, :labels.size(1)] = labels
    return padded_labels

def make_control(tokenizer, sent_toks, lab_toks, embsize):
    print("in make_control")

    # get the tokens needed
    lab_tokens_id = []
    for token in lab_toks:
        for label in token:
            if label not in lab_tokens_id:
                lab_tokens_id.append(label)

    special_tokens_dic = {'cls_token': '<s>', 'sep_token': '</s>', 'additional_special_tokens': lab_tokens_id}
    tokenizer.add_special_tokens(special_tokens_dic)

    new_lab_toks = []
    map_word_lab = {}
    # takes in the tokenizer, sentence tokens, label tokens
    # gets the ids for the sentences
    special_ids = tokenizer.additional_special_tokens
    for sent in sent_toks:
        #print("here's the sent:", sent)
        s = []
        for word in sent:
            if word not in map_word_lab:
                # select a random id
                select = random.choice(special_ids)
                #print("selected token:", select)
                # associate the word with the id
                map_word_lab[word] = select
                s.append(select)
            else:
                s.append(map_word_lab[word])
        #s = torch.LongTensor(s).view(1,-1)
        #new_lab_toks.append(pad_labels(s, embsize))
        new_lab_toks.append(s)

    return sent_toks, new_lab_toks

    # testing adding tokens
    # special_tokens_dic = {'cls_token': '<s>', 'sep_token': '</s>', 'additional_special_tokens': pos_list}
    # tokenizer.add_special_tokens(special_tokens_dic)
    # spec = tokenizer.additional_special_tokens
    # spec_ids = tokenizer.additional_special_tokens_ids
    # print("here are the special tokens:", spec)
    # print("here are the special token ids", spec_ids)

    # pass all of the labels in to get corresponding indices


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("t5-small", padding_side='left')

    '''
    # sentence format:
    sentence = [['hello', 'my', 'name', 'is', 'phil', 'because', 'asdfasfdsfsbs', 'fasting']]
    labels = [['WP', 'IN', 'NNP', 'VBD', 'IN', 'NNP', '.']]

    
    #subwords, token_start_idxs = subword_tokenize(sentence, tokenizer)
    #print(subwords, token_start_idxs)

    sents, labs = make_control(tokenizer, sentence, labels, 250)

    print("here are the sents:", sents)
    print("here are the labs:", labs)
    print("here's the specific vocab number for hello:", tokenizer.convert_tokens_to_ids('hello'))
    #for i in labs:
    print(labs[0])
    print(tokenizer.convert_ids_to_tokens(labs[0][0].numpy()))
    #print("here's the vocab:", tokenizer.get_vocab())
    '''
    #result = tokenizer.encode("pos: testing this stuff")
    #print(tokenizer.convert_ids_to_tokens(result))
    #print(result)

    grid = ner('nltk_data/corpora/conll2003/eng.train')
    ner_sents = [[(word[0],word[3]) for word in sent] for sent in grid]
    print("ner_sents:", ner_sents)
    sentence = []
    label = []
    for sent in grid:
        s = []
        l = []
        for word in sent:
            s.append(word[0])
            l.append(word[3])
        sentence.append(s)
        label.append(l)
    print("here's the sentences:", sentence)
    print("here's the label:", label)



