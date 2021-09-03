import sys
import time
import multiprocessing
from gensim import utils
from gensim.models import Word2Vec

import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# PARAMETER
EMBEDDING_SIZE = 300
EMBEDDING_EPOCH = 20
EMBEDDING_MIN_COUNT = 5
SAMPLE = 6e-5
ALPHA = 0.03
MIN_ALPHA = 0.0007
NEGATIVE_SAMPLING = 20

# DATA PATH
ROOT_CORPUS = '../data/corpus/'
ROOT_DATA = '../data/'
ROOT_VECTOR = '../data/vectors/'
CORPUS_PATH = ROOT_CORPUS + 'jvwiki_50.txt'

class Corpus:
    def __iter__(self):
        corpus = load_corpus(CORPUS_PATH)
        for line in corpus:
            yield utils.simple_preprocess(line)

def write_vectors(model, out_file):
    print("Writing vectors to a file..")
    file = open(out_file, 'w+', encoding='utf-8')
    file.write(str(len(model.wv.index_to_key)) + " " + str(EMBEDDING_SIZE) + "\n")

    for word in model.wv.index_to_key:
        file.write(word + " ")
        file.write(" ".join([str(x) for x in model.wv[word].tolist()]) + "\n")
    
    file.close()
    print("Successfullt writing {} word vectors to {}".format(len(model.wv.index_to_key), out_file))

def load_corpus(input_file):
    print('Loading corpus...')
    t1 = time.time()
    data = open(input_file, 'r', encoding='utf-8')
    corpus = data.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))
    
    return corpus

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Help: python3 word2vec.py <corpus>')
        sys.exit()

    OUT_PATH = ROOT_VECTOR + sys.argv[2]
    
    # prepare corpus for training
    sentences = Corpus()

    cores = multiprocessing.cpu_count()

    # define model
    print("Define model...")
    w2v_model = Word2Vec(min_count=EMBEDDING_MIN_COUNT,
                         vector_size=EMBEDDING_SIZE,
                         sample=SAMPLE,
                         alpha=ALPHA,
                         min_alpha=MIN_ALPHA,
                         negative=NEGATIVE_SAMPLING,
                         workers=cores-1,
                         compute_loss=True)
    
    # build vocabulary
    print("Build vocabulary...")
    t = time.time()
    w2v_model.build_vocab(sentences, progress_per=5000)
    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

    # training model
    print("Train model...")
    t = time.time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=EMBEDDING_EPOCH, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

    # save model
    # print("Save model...")
    # w2v_model.save("../model/w2v_jawa_sunda.model")

    # training loss
    training_loss = w2v_model.get_latest_training_loss()
    print("Training loss: ", training_loss)

    # write vectors to file
    write_vectors(w2v_model, OUT_PATH)
    
# python3 word2vec.py shuffled_corpus.txt su_words.txt jv_words.txt id_words.txt
