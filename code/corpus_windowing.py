import time
import sys

ROOT_CORPUS = '../data/corpus/'
WINDOW_SIZE = 78

def load_corpus(input_file):
    print('Loading corpus...')
    t1 = time.time()
    data = open(input_file, 'r', encoding='utf-8')
    corpus = data.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))

    return corpus

def window_corpus(corpus_file, out_file):
  data = load_corpus(corpus_file)

  with open(out_file, 'w', encoding='utf-8') as f:
    for line in data:
      if len(line.split()) > WINDOW_SIZE:
        words = line.rstrip()
        words = words.split()
        size = len(words)
        for i in range(0, size, WINDOW_SIZE):
          sent = " ".join([str(x) for x in words[i:i+WINDOW_SIZE]]) + "\n"
          f.write(sent)
          last_idx = i
        sent_last = " ".join([str(x) for x in words[i:]]) + "\n"
        f.write(sent)
      else:
        f.write(line)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Help: python3 corpus_windowing.py <corpus_file> <out_file>')
        sys.exit()

    CORPUS_PATH = ROOT_CORPUS + sys.argv[1]
    OUT_PATH = ROOT_CORPUS + sys.argv[2]

    window_corpus(CORPUS_PATH, OUT_PATH)
