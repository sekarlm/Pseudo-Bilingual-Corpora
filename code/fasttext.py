import sys
import time
import multiprocessing
from gensim.models import FastText
import logging

import xlsxwriter
import operator

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

# PARAMETER
EMBEDDING_SIZE = 300
EMBEDDING_WINDOW = 15
EMBEDDING_EPOCH = 1
EMBEDDING_MIN_COUNT = 20
SAMPLE = 6e-5
ALPHA = 0.03
MIN_ALPHA = 0.0007
NEGATIVE_SAMPLING = 20

# PATH
ROOT_CORPUS = '../data/corpus/'
ROOT_DATA = '../data/'
ROOT_RESULT = '../result/'

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
    sim_dict = {test: ft_model.wv.similarity(word, test) for test in eval_data if test in ft_model.wv.key_to_index}
    
    sim_dict = sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sim_dict = {k: v for k, v in sim_dict[:5]}

    return sorted_sim_dict

def write_similarity(filename, sim_1, sim_2):
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
        j = 9
        k = 10
        for index, key in enumerate(v):
            worksheet.write(i, j, key)
            worksheet.write(i, k, v[key])
            j += 2
            k += 2

    workbook.close()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Help: python3 fasttext.py <merged_corpus> <eval_file_jawa> <eval_file_sunda> <eval_file_indo>')
        sys.exit()

    input_file = ROOT_CORPUS + sys.argv[1]
    eval_file_jawa = ROOT_DATA + sys.argv[2]
    eval_file_sunda = ROOT_DATA + sys.argv[3]
    eval_file_indo = ROOT_DATA + sys.argv[4]
    corpus = load_corpus(input_file)

    cores = multiprocessing.cpu_count()

    # data preparation
    sentences = [ preprocessing(text) for text in corpus ]

    # eval data preparation
    jawa_eval = load_eval_data(eval_file_jawa)
    sunda_eval = load_eval_data(eval_file_sunda)
    indo_eval = load_eval_data(eval_file_indo)

    # define model
    print("Define model...")
    ft_model = FastText(min_count=EMBEDDING_MIN_COUNT,
                         window=EMBEDDING_WINDOW,
                         size=EMBEDDING_SIZE,
                         sample=SAMPLE,
                         alpha=ALPHA,
                         negative=NEGATIVE_SAMPLING,
                         workers=cores-1)
    
    # build vocabulary
    print("Build vocabulary...")
    t = time.time()
    ft_model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))

    # training model
    print("Train model...")
    t = time.time()
    ft_model.train(sentences, total_examples=ft_model.corpus_count, epochs=EMBEDDING_EPOCH, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))

    # save model
    # print("Save model...")
    # ft_model.save("../model/w2v_jawa_sunda.model")

    # init similarity
    ft_model.init_sims(replace=True)

    inputs = tqdm(jawa_eval)
    sim_sunda = Parallel(n_jobs=cores)(delayed(most_similar_to)(word, sunda_eval) for word in inputs)
    sim_indo = Parallel(n_jobs=cores)(delayed(most_similar_to)(word, indo_eval) for word in inputs)

    print(sim_sunda[0])
    print(sim_indo[0])

    write_similarity(ROOT_RESULT + 'ft_similarity.xlsx', sim_indo, sim_sunda)