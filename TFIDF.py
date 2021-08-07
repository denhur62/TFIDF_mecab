from scipy.io import mmwrite,mmread
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PartitionFit import partial_fit
from Tokenizer import tokenizer,stopwords
from DataLoader import data_loader

tf = TfidfVectorizer(
    stop_words=stopwords, tokenizer=tokenizer, ngram_range=(1, 2), max_features = 200000, sublinear_tf=True)
TfidfVectorizer.partial_fit = partial_fit

catergory=['book','concern','daliy','excercise','hobby','love','movie','pet','restaurant','study','travel']
mtx_path='model/model.mtx'

def first_learning(tf,mtx_path):
    article=[]
    for i in catergory:
        data=data_loader(i)
        article+=data
    X = tf.fit_transform(article)
    tf.n_docs = len(article)
    mmwrite(mtx_path,X)

def after_learning(tf,more):
    tf.partial_fit([more])

def recommand(tf,more):
    after_learning(tf,more)
    X = mmread(mtx_path).tocsr()
    example_vector = tf.fit_transform([more])
    print(example_vector.shape)
    print(X.shape)
    cos_similar= linear_kernel(example_vector,X).flatten()
    sim_rank_idx=cos_similar.argsort()[::-1]
    sim_rank_idx=sim_rank_idx[:5]
    for i in sim_rank_idx:
        print(cos_similar[i])
    
    
# first_learning(tf,mtx_path)

# X = mmread(mtx_path).tocsr() (22515,441976)

recommand(tf,'오늘은 책을 읽고 싶은 밤이다. 일기를 쓰려고 누웠는데 책이 눈에 들어왔다. 내돈내산하고 작성하는 책 리뷰 여행을 가고자 하였다.')



