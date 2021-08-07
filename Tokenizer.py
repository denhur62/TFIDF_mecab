from konlpy.tag import Mecab
import pandas as pd


stopwords=pd.read_csv('data/stopwords.csv')

stopwords =list(stopwords['stopword'])

def tokenizer(raw, pos=["NNG", "NNP", "VV", "VA"], stopword=stopwords):
    m = Mecab()
    return [word for word, tag in m.pos(raw) if len(word) > 1 and tag in pos and word not in stopword]