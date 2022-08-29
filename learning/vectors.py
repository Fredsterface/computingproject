import json 
with open("../notebooks/larry.json", "r") as F:
    data = json.load(F)
    
from nltk.data import find
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import gensim
import numpy as np
from scipy import spatial
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

def norm(vec):
	return np.linalg.norm(vec)

def similarity(vec1, vec2):
	if norm(vec1) < 0.1 or norm(vec2) < 0.1:
		return 0.0
	return 1.0 - spatial.distance.cosine(vec1, vec2)


def doc2vec(doc):
	v = model['house'] - model['house']
	cnt = 0
	for w in doc.split(' '):
		if model.has_index_for(w) and not w in stopwords:
			v += model.get_vector(w)
			cnt += 1
	if cnt > 0:		
		v /= norm(v)
	return v

docvecs = [doc2vec(doc) for doc in data]

def search(word):
	v = model.get_vector(word)
	scores = [similarity(v,dv) for dv in docvecs]
	idxs = np.argsort(scores)[::-1][:10]
	return [data[i] for i in idxs]
