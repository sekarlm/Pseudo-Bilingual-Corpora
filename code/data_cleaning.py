import numpy as np
import sys

ROOT_VECTORS = '../data/vectors/'

def read_and_filter_embeddings(embeddings_file, english_words):
  word2id = {}
  id2vec = []

  with open(embeddings_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    num = 0
    for i, line in enumerate(f):
      if i == 0:
        split = line.split()
        print("number of embeddings:", int(split[0]))
      else:
        word, vect = line.rstrip().split(' ', 1)
        if word not in english_words:
          word2id[word] = len(word2id)
          id2vec.append(np.fromstring(vect, sep=' '))
        else:
          num += 1
  print(num)
  print(len(word2id))
  print(len(id2vec))

  return word2id, id2vec

def write_vectors(word2id, id2vec, out_file):
    print("Writing vectors to a file..")
    file = open(out_file, 'w+', encoding='utf-8')
    file.write(str(len(word2id)) + " " + str(len(id2vec[0])) + "\n")

    for key, value in word2id.items():
        file.write(key + " ")
        file.write(" ".join([str(x) for x in id2vec[value].tolist()]) + "\n")

    file.close()
    print("Successfullt writing {} word vectors to {}".format(len(word2id), out_file))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Help: python3 data_cleaning.py <embedding_file> <english_words_file> <output_file>')
        sys.exit()

    EMBEDDING_FILE = ROOT_VECTORS + sys.argv[1]
    ENGLISH_FILE = ROOT_VECTORS + sys.argv[2]
    OUT_FILE = ROOT_VECTORS + sys.argv[3]

    f = open(ENGLISH_FILE, "r")
    data = f.readlines()
    f.close()

    english_words = [ x.rstrip() for x in data ]
    print("number of english words: ", len(english_words))

    word2id, id22vec = read_and_filter_embeddings(EMBEDDING_FILE, english_words)

    write_vectors(word2id, id22vec, OUT_FILE)
