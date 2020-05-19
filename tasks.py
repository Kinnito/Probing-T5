from __future__ import print_function
import nltk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from nltk.corpus import conll2000
from nltk.util import LazyConcatenation, LazyMap
from nltk.corpus.reader import ConllCorpusReader

# appending a path every time because it won't stick
nltk.data.path.append('/data/limill01/Probing-T5/nltk_data/')

device = torch.device('cuda')

# get the dataset into the correct form and append "chunking:" to the front
def chunking():
    train_sents = conll2000.chunked_sents('train.txt')
    train_data = [[w for w,t,c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]
    train_label = [[c for w,t,c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents] 
  
    # now append chunking to the front of each group/string
    return train_data, train_label

def ner():  
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

 
    reader = nerConllReader(r'nltk_data/corpora/conll2003/', r'eng.train', ('words', 'pos', 'ne'))
    print(reader.iob_words())

    '''
    print("here's the grids:", temp._grids())
    print(reader._get_iob_words(temp._grids(), columns=['ne']))
    print(reader._get_iob_words(temp._grids(), columns=['chunk', 'ne']))
    '''

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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("t5-small", padding_side='left')

    # sentence format:
    sentence = ['hello', 'my', 'name', 'is', 'phil', 'because', 'asdfasfdsfsbs', 'fasting']
    labels = ['WP', 'IN', 'NNP', 'VBD', 'IN', 'NNP', '.']

    
    subwords, token_start_idxs = subword_tokenize(sentence, tokenizer)
    print(subwords, token_start_idxs)

    print()

    #subword_ids, mask, token_starts = subword_tokenize_to_ids(sentence, tokenizer, 250)
    # print(subword_ids, mask, token_starts)
    #print(subword_ids) # only look in the ids where we have a 1 for starts
    #print(mask) # don't need this wen testing
    #print(token_starts) # only look in the starts where 

    #sents, pos_tokens = pos()
    # print(sents)
    # print(pos_tokens)
    # print(pos())
