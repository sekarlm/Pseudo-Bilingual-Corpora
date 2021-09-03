import time
import sys
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import numpy as np

ROOT_CORPUS = '../data/corpus/'
ROOT_VECTOR = '../data/vectors/'
MIN_COUNT = 5
EMBEDDING_SIZE = 300

import logging
logging.basicConfig(level=logging.INFO)

def get_tokens_and_segments_tensor(sentence, tokenizer):
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokens_tensor, segments_tensors, tokenized_text

def extract_vectors(model, tokens_tensor, segments_tensor, mode):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    token_vecs = [ torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0) for token in token_embeddings ]

    return token_vecs

def get_word_and_vectors(vectors, tokenized_text):
    print("Get words and its vectors...")
    word_and_vectors = []

    for i, token_str in enumerate(tokenized_text):
        x = {'word': "", "vector": []}
        if token_str[:2] == '##':
            embedding = word_and_vectors[len(word_and_vectors)-1]
            word = embedding['word']+token_str[2:]
            embedding['word'] = word
            embedding['vector'].append(vectors[i])
        else:
            x['word'] = token_str
            x['vector'].append(vectors[i])
            word_and_vectors.append(x)

    for i, item in enumerate(word_and_vectors):
        if len(item['vector']) > 1:
            divisor = len(item['vector'])
            word_and_vectors[i]['vector'] = torch.stack(item['vector'], dim=0).sum(dim=0) / divisor
        else:
            word_and_vectors[i]['vector'] = item['vector'][0]

    # print(type(word_and_vectors[i]['vector']))

    return word_and_vectors

def write_vectors(word2id, id2vec, out_file):
    print("Writing vectors to a file..")
    file = open(out_file, 'w+', encoding='utf-8')
    file.write(str(len(word2id)) + " " + str(len(id2vec[0])) + "\n")

    for key, value in word2id.items():
        file.write(key + " ")
        file.write(" ".join([str(x) for x in id2vec[value].tolist()]) + "\n")

    file.close()
    print("Successfullt writing {} word vectors to {}".format(len(word2id), out_file))

def load_corpus(input_file):
    print('Loading corpus...')
    print(input_file)
    t1 = time.time()
    data = open(input_file, 'r', encoding='utf-8')
    corpus = data.readlines()
    t2 = time.time()
    total_time = t2 - t1
    print('Time to load corpus : {:0.3f}'.format(total_time))

    return corpus

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Help: python3 mBERT.py <corpus> <output_file>')
        sys.exit()

    CORPUS_PATH = ROOT_CORPUS + sys.argv[1]
    OUTPUT_PATH = ROOT_VECTOR + sys.argv[2]

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

    # Load pre-trained model (weight)
    model = BertModel.from_pretrained('bert-base-multilingual-uncased',
                                  output_hidden_states = True)

    # Put the model in "evaluation" model, meaning feed-forward operation
    model.eval()

    # Load corpus
    sentences = load_corpus(CORPUS_PATH)
    print("Length of sentences: ", len(sentences))

    # Collect all vectors of each words
    word2id = {}
    id2vec = []

    for i, sentence in enumerate(sentences):
        print("processing sentence-{}".format(i))
        tokens_tensor, segments_tensor, tokenized_text = get_tokens_and_segments_tensor(sentence, tokenizer)
        token_vectors = extract_vectors(model, tokens_tensor, segments_tensor, mode="sum")
        word_and_vectors = get_word_and_vectors(token_vectors, tokenized_text)

        for i, item in enumerate(word_and_vectors):
            # print("{} {}".format(i, type(item['vector'])))
            if item['word'] in word2id:
                idx = word2id[item['word']]
                id2vec[idx].append(item['vector'])
            else:
                word2id[item['word']] = len(word2id)
                vecs = [item['vector']]
                id2vec.append(vecs)

    print("Number of unique words before filtering: ", len(word2id))

    # Filter and collect all words which appeare <min_count> times or more
    word2id_filtered = {}
    id2vec_filtered = []

    for key, value in word2id.items():
        print("processing word-{}".format(value))
        if len(id2vec[value]) >= MIN_COUNT:
            word2id_filtered[key] = len(word2id_filtered)
            divisor = len(id2vec[value])
            vector = torch.stack(id2vec[value], dim=0).sum(dim=0) / divisor
            id2vec_filtered.append(vector.tolist())

    # Dimensionality reduction
    pca = PCA(n_components=EMBEDDING_SIZE)
    pca.fit(np.array(id2vec_filtered))
    id2vec_filtered = pca.transform(id2vec_filtered)

    print("Number of unique words after filtering: ", len(word2id_filtered))

    write_vectors(word2id_filtered, id2vec_filtered, OUTPUT_PATH)
