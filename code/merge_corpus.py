import sys

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

def load_merge(input_file1, input_file2, output_file):
    print('Merging corpus...')
    f1 = open(input_file1, 'r+', encoding='utf-8')
    corpus1 = f1.readlines()
    f1.close()

    f2 = open(input_file2, 'r+', encoding='utf-8')
    corpus2 = f2.readlines()
    f2.close()

    out = open(output_file, 'w', encoding='utf-8')
    out.writelines(corpus1 + corpus2)
    out.close()
    print('Merged...')

def merge_shuffle(c1, c2, c3, output_file):
    print('Merging three corpus...')
    r1 = int(len(c1)/len(c2))
    r2 = int(len(c1)/len(c3))
    new_corpus = []

    j = 0
    k = 0
    for i in range(len(c1)):
        new_corpus.append(c1[i])
        if (i%r1 == 0) and (i<len(c2)):
            new_corpus.append(c2[j])
            j += 1
        if (i%r2 == 0) and (i<len(c3)):
            new_corpus.append(c3[k])
            k += 1
    
    out = open(output_file, 'w', encoding='utf-8')
    out.writelines(new_corpus)
    out.close()
    print('Merged...')

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Help: python3 merge_corpus.py <input_file1> <input_file2> <input_file3> <output_file>')
        sys.exit()
    
    print('Loading corpus file...')

    input_file1 = ROOT_CORPUS + sys.argv[1]
    input_file2 = ROOT_CORPUS + sys.argv[2]
    input_file3 = ROOT_CORPUS + sys.argv[3]
    output_file = ROOT_CORPUS + sys.argv[4]

    c1 = load_corpus(input_file1)
    c2 = load_corpus(input_file2)
    c3 = load_corpus(input_file3)

    print(len(c1), len(c2), len(c3))

    merge_shuffle(c1, c2, c3, output_file)

    out = open(output_file, 'r', encoding='utf-8')
    corpus = out.readlines()
    out.close()

    print('Total number of sentences in merged corpus: {}'.format(len(corpus)))

# python3 merge_corpus.py id_latest.txt jv_latest.txt su_latest.txt shuffled_corpus.txt