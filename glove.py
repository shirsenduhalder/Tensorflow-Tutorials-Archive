import numpy as np
import scipy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

EMBEDING_FILE = 'glove/glove.6B.300d.txt'

print("Indexing word vectors")
embeddings_index = {}
f = open(EMBEDING_FILE)
count = 0

for line in f:
	if count == 0:
		count = 1
		continue
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype=np.float32)
	embeddings_index[word] = coefs

f.close()

king_wordvec = embeddings_index['king']
queen_wordvec = embeddings_index['queen']
man_wordvec = embeddings_index['man']
woman_wordvec = embeddings_index['woman']
pseudo_king = queen_wordvec - woman_wordvec + man_wordvec

cosine_simi = np.dot(pseudo_king/np.linalg.norm(pseudo_king), king_wordvec/np.linalg.norm(king_wordvec))

print("Cosine Similarity: {}".format(cosine_simi))

tsne = TSNE(n_components=2)
words_array = []
word_list = ['king', 'queen', 'man', 'woman']

for w in word_list:
    words_array.append(embeddings_index[w])

index1 = list(embeddings_index.keys())[0:100]
for i in range(100):
	words_array.append(embeddings_index[index1[i]])

words_array = np.array(words_array)
words_tsne = tsne.fit_transform(words_array)

ax = plt.subplot(111)

for i in range(4):
	plt.text(words_tsne[i, 0], words_tsne[i, 1], word_list)
	
plt.xlim((50, 125))
plt.ylim((0, 80))