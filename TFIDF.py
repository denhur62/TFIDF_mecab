import re
import numpy as np
from scipy.sparse.dia import dia_matrix
from scipy.io import mmwrite,mmread
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from konlpy.tag import Mecab

# 추후 들어오는 데이터는 정규식을 통해 refine 할 것

def partial_fit(self, X):
    max_idx = max(self.vocabulary_.values())
    for a in X:
        # update vocabulary_
        
        tokens = re.findall(self.token_pattern, a)
        for w in tokens:
            if w not in self.vocabulary_:
                max_idx += 1
                self.vocabulary_[w] = max_idx

        # update idf_
        df = (self.n_docs + self.smooth_idf) / \
            np.exp(self.idf_ - 1) - self.smooth_idf
        self.n_docs += 1
        df.resize(len(self.vocabulary_))
        for w in tokens:
            df[self.vocabulary_[w]] += 1
        idf = np.log((self.n_docs + self.smooth_idf) /
                     (df + self.smooth_idf)) + 1
        self._tfidf._idf_diag = dia_matrix(
            (idf, 0), shape=(len(idf), len(idf)))


stopwords = ['아', '이', '애', '오']


def tokenizer(raw, pos=["NNG", "NNP", "VV", "VA"], stopword=stopwords):
    m = Mecab()
    return [word for word, tag in m.pos(raw) if len(word) > 1 and tag in pos and word not in stopword]


tf = TfidfVectorizer(
    stop_words=stopwords, tokenizer=tokenizer, ngram_range=(1, 3), min_df=2, sublinear_tf=True)


articleList = ['그림이 작아서 진짜 금방 완성한다특히 큰 사이즈 보석 십자수 해보신 분들은난이도가 하중에 하일 듯물론 작아서 좋은 건 있는데', '불편한 건 컬러 구분이 잘 안된다분명 그림을 보면 색이 은근 다른데구슬에서 구분이 안 간다큰 사이즈의 그림들은 다 나눠져있으니까쉽게 구분이 가고',
               '그 컬러만 꺼내서 붙이면 되는데작다 보니 모든 컬러의 보석들이 함께 들어있어서구분하기 어려우니 머리카락이나 피부처럼 구분 가능한 것부터 해봐도 좋다그리고 잠깐 다른 리뷰도 함께사실 저 지갑 컬러도 마음에 들고 디자인도 좋아서 구입했는데 구입하고 일주일 만에 안녕입구가 저렇게 좁은 거 생각 못 하고 구입해서몇 번 쓰다가 불편해서 힘으로 ㅋㅋ']
articleList2="큰 사이즈 십자수가 좋아요 짱짱 좋아요~ 다음에 또 사고 시펑요~!"
X = tf.fit_transform(articleList)
TfidfVectorizer.partial_fit = partial_fit
tf.n_docs = len(articleList)

print(tf.vocabulary_)
mtx_path='model.mtx'
mmwrite(mtx_path,X)

diary_tdm=mmread(mtx_path)
diary_tdm=diary_tdm.tocsr()


# tf.partial_fit([articleList2])
print(tf.get_feature_names())

example_vector = tf.transform([articleList2])
cos_similar= linear_kernel(example_vector,X).flatten()
sim_rank_idx=cos_similar.argsort()[::-1]
print(sim_rank_idx)
for i in sim_rank_idx:
    print(cos_similar[i])