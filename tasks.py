from __future__ import print_function
import nltk
import ipywidgets as widgets
from transformers import AutoTokenizer, AutoModelWithLMHead
from nltk.corpus import conll2000
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
        print("pos here", self._colmap['pos'])
        print("length of grid:", len(grid))
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['words']), pos_tags,
                   self._get_column(grid, self._colmap[columns])))

  temp = nerConllReader(r'nltk_data/corpora/conll2003/', r'eng.train', ('words', 'pos', 'chunk', 'ne'))
  print(temp)
  print(temp.iob_words())

  '''
  print("here's the grids:", temp._grids())
  print(temp._get_iob_words(temp._grids(), columns=['ne']))
  print(temp._get_iob_words(temp._grids(), columns=['chunk', 'ne']))
  '''
  train_sents = temp.iob_words()
  train_data = [sent[0] for sent in train_sents]
  train_label = [sent[2] for sent in train_sents]
  
  print("train data first", train_data[0])
  print("train label first", train_label[0])

if __name__ == "__main__":
  # TODO:
  ## 1. Figure out the format of the data that needs to be fed in (take a look at the tokenization)
  ##    that was given (probably will have to pad sequence)
  ## 2. Figure out what exactly the output of the model means in this case
  tokenizer = AutoTokenizer.from_pretrained("t5-small")
  model = AutoModelWithLMHead.from_pretrained("t5-small")
  
  train_data, train_label = chunking()
  
  d, l = ner()
  print(d[0])
  print(l[0])
  '''
  print("train data", train_data[0])
  print("train label", train_label[0])
  train_d = ' '.join(train_data[0])
  train = "chunking: " + train_d
  print(train)
  train_label = ' '.join(train_label[0])
  print(train_label)
  inputs = tokenizer.encode(train_d, return_tensors='pt')
  labels = tokenizer.encode(train_label, return_tensors='pt')
  print("here's the inputs", inputs)
  print("here's the labels", labels)
  result = model(input_ids=inputs, lm_labels=labels)
  print(result)
  print("size of the results", result.size())
  print("size of the inputs", labels.size())
  print("chunking now")
  '''
