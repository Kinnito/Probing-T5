from __future__ import print_function
import nltk
import ipywidgets as widgets
from transformers import AutoTokenizer, AutoModelWithLMHead
from nltk.corpus import conll2000
from nltk.corpus import treebank
from nltk.util import LazyConcatenation, LazyMap
from nltk.corpus.reader import ConllCorpusReader
from tokenizers import (ByteLevelBPETokenizer,
                            CharBPETokenizer,
                            SentencePieceBPETokenizer,
                            BertWordPieceTokenizer)

# appending a path every time because it won't stick
nltk.data.path.append('/data/limill01/Probing-T5/nltk_data/')

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

def pos2():
    '''
    def convert(line):
        sent = line.split()
        if not sent:
            return []
        else:
            return (sent[1], sent[4])
    '''

    filename = 'UD_English-EWT/en_ewt-ud-train.conllu'
    word_pos = []
    with open(filename) as f:

        # list comp that doesn't work ish
        # word_pos = [[result for result in convert(line) if result]
        #        for line in f if not line.startswith('#')] 

        sent = []
        for line in f:
            stripped = line.split()
            if line.startswith('#'):
                continue
            elif len(stripped) == 0:
                word_pos.append(sent)
                sent = []
            else:
                sent.append((stripped[1], stripped[4]))

def create_batches():


if __name__ == "__main__":
    # TODO:
    ## 1. Figure out the format of the data that needs to be fed in (take a look at the tokenization)
    ##    that was given (probably will have to pad sequence)
    ## 2. Figure out what exactly the output of the model means in this case
    # tokenizer = AutoTokenizer.from_pretrained("t5-small")
    #model = AutoModelWithLMHead.from_pretrained("t5-small")
    
    #pos()
    pos2()
