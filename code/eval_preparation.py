import sys, time
from collections import defaultdict

ROOT_CORPUS = '../data/corpus/'
ROOT_DATA = '../data/'

def word_freq(sentences):
    print("Counting words frequency...")
    freq = defaultdict(int)
    for sent in sentences:
        for word in sent.split():
            freq[word] += 1

    print("Total number of words: {}.".format(len(freq)))

    return freq

def write_most_frequent(list_freq, output_file):
    print("Writing most frequent words...")
    data = open(output_file, 'w', encoding='utf-8')

    for word in list_freq:
        data.write(word + "\n")
    
    data.close()
    print("Finish writing...")

def load_corpus(input_file):
    print('Loading corpus...')
    t1 = time.time()
    corpus = input_file.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))
    
    return corpus

def load_eval_data(eval_file):    
    with open(eval_file, 'r', encoding='utf-8') as f:
        res = [ x.replace("\n", "") for x in f.readlines() ]
        return res

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Help: python3 eval_preparation.py <corpus_file> <eval_data_file> <min_count>')
        sys.exit()
    
    input_file = ROOT_CORPUS + sys.argv[1]
    output_file = ROOT_DATA + sys.argv[2]
    min_count = int(sys.argv[3])

    corpus_file = open(input_file, 'r', encoding='utf-8')
    
    corpus = load_corpus(corpus_file)
    frequency = word_freq(corpus)

    most_freq = defaultdict(int)
    for word in frequency:
        if frequency[word] >= min_count:
            most_freq[word] = frequency[word]
    
    keys = sorted(most_freq, key=most_freq.get, reverse=True)
    new_freq = defaultdict(int)
    for key in keys:
        new_freq[key] = most_freq[key]
        # print("{} ---- {}".format(key, most_freq[key]))
    
    print(len(new_freq))

    write_most_frequent(new_freq, output_file)

# python3 eval_preparation.py id_latest.txt id_words.txt 100
# python3 eval_preparation.py jv_latest.txt jv_words.txt 100
# python3 eval_preparation.py su_latest.txt su_words.txt 100