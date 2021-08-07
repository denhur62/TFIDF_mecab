import re
import numpy as np
from scipy.sparse.dia import dia_matrix
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