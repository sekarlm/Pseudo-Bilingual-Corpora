"""
Creates a corpus from Wikipedia dump file.
"""

import sys, time
from gensim.corpora import WikiCorpus
from statistics import mean

ROOT_WIKI_DUMPS = '../data/wiki_dumps/'
ROOT_CORPUS = '../data/corpus/'

def generate_corpus(input_file, output_file):
    output = open(output_file, 'a', encoding='utf-8')
    wiki = WikiCorpus(input_file, lemmatize=False, dictionary={}, lower=True)

    i = 0
    space = " "
    for text in wiki.get_texts():
        article = space.join(text) + "\n"
        output.write(article)
        i += 1
        if (i % 10000 == 0):
            print('Saved ' + str(i) + ' articles')
    output.close()
    print('Processing Wikipedia dump file complete!')

def check_corpus(input_file):
    while(1):
        for lines in range(2):
            print(input_file.readline())
        user_input = input('>>> Type STOP to quit or hit Enter key for more <<< ')
        if user_input == 'STOP':
            break

def load_corpus(input_file):
    print('Loading corpus...')
    t1 = time.time()
    corpus = input_file.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))
    
    return corpus

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Help: python3 data_preparation.py <wikipedia_dump_file> <processed_text_file> <n_copies>')
        sys.exit()

    input_file = ROOT_WIKI_DUMPS + sys.argv[1]
    output_file = ROOT_CORPUS + sys.argv[2]
    n_copies = int(sys.argv[3])

    print('Creating corpus file...')
    for i in range(0, n_copies):
        generate_corpus(input_file, output_file)

    print('Checking corpus file...')
    corpus_file = open(output_file, 'r', encoding='utf-8')
    # check_corpus(corpus_file)
    corpus = load_corpus(corpus_file)
    print('Size of corpus file : {} sentences'.format(len(corpus)))
    length_line = [ len(line.split()) for line in corpus ]

    print("Max length: ", max(length_line))
    print("Min length: ", min(length_line))
    print("Average length: ", mean(length_line))

# python3 data_preparation.py idwiki-latest-pages-articles.xml.bz2 id_latest.txt 1
# python3 data_preparation.py jvwiki-latest-pages-articles.xml.bz2 jv_latest.txt 1
# python3 data_preparation.py suwiki-latest-pages-articles.xml.bz2 su_latest.txt 1   