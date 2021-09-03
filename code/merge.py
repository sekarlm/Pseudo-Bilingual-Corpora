import sys
import time

ROOT_CORPUS = '../data/corpus/'

def load_corpus(filename):
    print('Loading corpus...')
    t1 = time.time()
    input_file = open(filename, 'r', encoding='utf-8')
    corpus = input_file.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))
    
    return corpus

def merge_shuffle(c1, c2, output_file):
    print('Merging two corpus...')
    r1 = int(len(c1)/len(c2))
    new_corpus = []

    j = 0
    for i in range(len(c1)):
        new_corpus.append(c1[i])
        if (i%r1 == 0) and (i<len(c2)):
            new_corpus.append(c2[j])
            j += 1
    
    out = open(output_file, 'w', encoding='utf-8')
    out.writelines(new_corpus)
    out.close()
    print('Merged...')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Help: python3 merge_corpus.py <input_file1> <input_file2> <output_file>')
        sys.exit()
    
    print('Loading corpus file...')

    input_file1 = ROOT_CORPUS + sys.argv[1]
    input_file2 = ROOT_CORPUS + sys.argv[2]
    output_file = ROOT_CORPUS + sys.argv[3]

    c1 = load_corpus(input_file1)
    c2 = load_corpus(input_file2)

    print(len(c1), len(c2))

    merge_shuffle(c1, c2, output_file)

    out = open(output_file, 'r', encoding='utf-8')
    corpus = out.readlines()
    out.close()

    print('Total number of sentences in merged corpus: {}'.format(len(corpus)))

# python3 merge_corpus.py id_latest.txt jv_latest.txt su_latest.txt shuffled_corpus.txt
