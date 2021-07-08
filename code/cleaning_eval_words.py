import sys

eng_words = ['will', 'image', 'press', 'into', 'if', 'they', 'part', 'us', 'man', 'his', 'the', 'can', 'was', 'thump', 'number', 'central', 'center', 'known', 'use', 'of', 'new', 'has', 'td',
'american', 'example', 'who', 'than', 'union', 'which', 'px', 'jpg', 'jpeg', 'png', 'such', 'i', 'ii', 'iii', 'state', 'each', 'these', 'one', 'first', 'up', 'where', 'then', 'york', 'and', 'on',
'in', 'this', 'not', 'but', 'high', 'jr', 'www', 'http', 'company', 'to', 'it', 'also', 'more', 'many', 'south', 'north', 'east', 'west', 'march', 'is', 'at', 'have', 'fm', 'see', 'we', 'power',
'as', 'line', 'world', 'com', 'right', 'left', 'about', 'an', 'be', 'used', 'cc', 'there', 'field', 'common', 'only', 'for', 'or', 'time', 'pt', 'function', 'current', 'series', 'station', 'by',
'from', 'railway', 'called', 'their', 'states', 'he', 'united', 'on', 'with', 'other', 'two', 'over', 'list', 'do', 'been', 'when', 'jpg', 'were', 'that', 'system', 'may', 'john', 'between', 'most',
'probability', 'same', 'after', 'through', 'would']

ROOT_DATA = '../data/'

def clean_english_word(words):
    return list(set(words) - set(eng_words))

def write_words(words, output_file):
    print("Writing most frequent words...")
    data = open(output_file, 'w', encoding='utf-8')

    for word in words:
        data.write(word + "\n")
    
    data.close()
    print("Finish writing...")

def load_eval_data(eval_file):    
    with open(eval_file, 'r', encoding='utf-8') as f:
        res = [ x.replace("\n", "") for x in f.readlines() ]
        return res

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Help: python3 cleaning_eval_words.py <eval_file> <list_words>')
        sys.exit()
    
    filename = ROOT_DATA + sys.argv[1]
    id_filename = ROOT_DATA + sys.argv[2]
    output = ROOT_DATA + 'id_filtered.txt'
    # print(filename)
    words = load_eval_data(filename)
    id_words = load_eval_data(id_filename)

    new_words = list(set(id_words) - (set(id_words) - set(words)))
    print(len(new_words))

    # cleaned_words = clean_english_word(words)

    write_words(new_words, output)

# python3 cleaning_eval_words.py <eval_file>