import pandas as pd

data = pd.read_csv('stopwords.txt', sep = "\t",names=['stopword', 'tag', 'per'])
stopwords= data['stopword']

stopwords.to_csv("data/stopwords.csv",index=False)
