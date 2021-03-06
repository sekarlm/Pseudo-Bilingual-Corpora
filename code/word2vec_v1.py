import sys
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from gensim import utils
from gensim.models import Word2Vec
import logging

import xlsxwriter

import operator

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# PARAMETER
EMBEDDING_SIZE = 300
# EMBEDDING_WINDOW = 15
EMBEDDING_EPOCH = 20
EMBEDDING_MIN_COUNT = 5
SAMPLE = 6e-5
ALPHA = 0.03
MIN_ALPHA = 0.0007
NEGATIVE_SAMPLING = 20

# DATA PATH
ROOT_CORPUS = '../data/corpus/'
ROOT_DATA = '../data/'
ROOT_RESULT = '../result/'
CORPUS_PATH = ROOT_CORPUS + 'su_latest.txt'

class Corpus:
    def __iter__(self):
        corpus = open(CORPUS_PATH, 'r+', encoding='utf-8')
        for line in corpus:
            yield utils.simple_preprocess(line)

def write_vectors(model, out_file):
    file = open(out_file, 'w+', encoding='utf-8')
    file.write(str(len(model.wv.vocab)) + " " + str(EMBEDDING_SIZE) + "\n")

    for word in model.wv.vocab:
        file.write(word + " ")
        file.write(" ".join([str(x) for x in model.wv[word].tolist()]) + "\n")
    
    file.close()

def load_corpus(input_file):
    print('Loading corpus...')
    t1 = time.time()
    data = open(input_file, 'r', encoding='utf-8')
    corpus = data.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))
    
    return corpus

def preprocessing(text):
    return [ x for x in text.split() ]

def load_eval_data(eval_file):    
    with open(eval_file, 'r', encoding='utf-8') as f:
        res = [ x.replace("\n", "") for x in f.readlines() ]
        return res

def most_similar_to(word, eval_data):
    print("most similar to ", word)
    sim_dict = {test: w2v_model.wv.similarity(word, test) for test in eval_data if test in w2v_model.wv.vocab}
    
    sim_dict = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sim_dict = {k: v for k, v in sim_dict[:3]}

    return (word, sorted_sim_dict)

def write_similarity(filename, sim_1, sim_2):
    print("Write word similarity to file")
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    for i in range(len(sim_1)):
        k, v = sim_1[i]
        worksheet.write(i, 0, k)
        j = 1
        k = 2
        for index, key in enumerate(v):
            worksheet.write_string(i, j, key)
            worksheet.write_number(i, k, v[key])
            j += 2
            k += 2
    
    for i in range(len(sim_2)):
        k, v = sim_2[i]
        worksheet.write(i, 8, k)
        j = 11
        k = 12
        for index, key in enumerate(v):
            worksheet.write(i, j, key)
            worksheet.write(i, k, v[key])
            j += 2
            k += 2

    workbook.close()
    print("Finish writing...")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Help: python3 word2vec.py <merged_corpus> <eval_file_sunda> <eval_file_jawa> <eval_file_indo>')
        sys.exit()

    input_file = ROOT_CORPUS + sys.argv[1]
    eval_file_sunda = ROOT_DATA + sys.argv[2]
    eval_file_jawa = ROOT_DATA + sys.argv[3]
    eval_file_indo = ROOT_DATA + sys.argv[4]
    corpus = load_corpus(input_file)

    cores = multiprocessing.cpu_count()

    # data preparation
    sentences = [ preprocessing(text) for text in corpus ]

    # eval data preparation
    sunda_eval = load_eval_data(eval_file_sunda)
    jawa_eval = load_eval_data(eval_file_jawa)
    indo_eval = load_eval_data(eval_file_indo)

    # define model
    print("Define model...")
    w2v_model = Word2Vec(min_count=EMBEDDING_MIN_COUNT,
                         window=EMBEDDING_WINDOW,
                         size=EMBEDDING_SIZE,
                         sample=SAMPLE,
                         alpha=ALPHA,
                         min_alpha=MIN_ALPHA,
                         negative=NEGATIVE_SAMPLING,
                         workers=cores-1)
    
    # build vocabulary
    print("Build vocabulary...")
    t = time.time()
    w2v_model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

    # training model
    print("Train model...")
    t = time.time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=EMBEDDING_EPOCH, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

    # save model
    # print("Save model...")
    # w2v_model.save("../model/w2v_jawa_sunda.model")

    # init similarity
    w2v_model.init_sims(replace=True)

    inputs = tqdm(sunda_eval[:1000])
    sim_jawa = Parallel(n_jobs=cores)(delayed(most_similar_to)(word, jawa_eval) for word in inputs)
    sim_indo = Parallel(n_jobs=cores)(delayed(most_similar_to)(word, indo_eval) for word in inputs)

    print(sim_jawa[0])
    print(sim_indo[0])

    write_similarity(ROOT_RESULT + 'w2v_similarity.xlsx', sim_indo, sim_jawa)

# python3 word2vec.py shuffled_corpus.txt su_words.txt jv_words.txt id_words.txt