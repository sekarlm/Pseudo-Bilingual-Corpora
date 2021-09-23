import sys
import torch
import numpy as np

VECTORS_PATH = '../data/vectors/'
DICTIONARY_PATH = '../data/dictionaries/'
RESULTS_PATH = '../result/'

"""
TRANSLATION RETRIEVAL
"""
def load_embedding(embedding_path):
    """
    Load word embeddings from a txt file
    """
    word2id = {}
    id2word = {}
    vectors = []

    with open(embedding_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

        for i, line in enumerate(data):
            if i == 0:
                split = line.split()
                n_emb = int(split[0])
                emb_dim = int(split[1])
                print("Number of word embeddings: ", n_emb)
                print("Embedding dimension: ", emb_dim)
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')

                # avoid to have null embeddings
                if np.linalg.norm(vect) == 0:
                    vect[0] = 0.01

                # check for double words
                if word in word2id:
                    print("Word {} found twice in embedding file.".format(word))
                else:
                    if not vect.shape == (emb_dim,):
                        print("Invalid dimension")

                    assert vect.shape == (emb_dim,), i
                    word2id[word] = len(word2id)
                    id2word[len(word2id)-1] = word
                    vectors.append(vect[None])

        embeddings = np.concatenate(vectors, 0)
        embeddings = torch.from_numpy(embeddings).float()
        print(embeddings.size())

    return word2id, id2word, embeddings

def load_dictionary(path, word2id):
    """
    Load pair translation from a txt file for testing
    """
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with open(path, 'r', encoding='utf-8') as f:
        # print('Number of seed dictionary: ', len(f.readlines()))
        data = f.readlines()
        n_gold_std = len(data)
        for index, line in enumerate(data):
            word1, word2 = line.rstrip().split()
            if (word1 in word2id) and (word2 in word2id):
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id)
                not_found2 += int(word2 not in word2id)

    print('Not found pair words: ', not_found)
    print('Not found source words: ', not_found1)
    print('Not found target words: ', not_found2)
    print('Number of found pairs words: ', len(pairs))
    
    # sort dictionary by source words frequency
    pairs = sorted(pairs, key=lambda x: word2id[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id[word1]
        dico[i, 1] = word2id[word2]
    
    return dico, n_gold_std

def get_word_translation(emb, word2id, dico):
    """
    Get all word translations of source words using nearest neighbor method
    """
    # normalize embedding
    emb = emb / emb.norm(2, 1, keepdim=True).expand_as(emb)

    # nearest neighbor
    query = emb[dico[:, 0]]
    scores = query.mm(emb.transpose(0, 1))
    top_matches = scores.topk(11, 1, True)[1]
    top_matches = top_matches[:, 1:]

    return top_matches

"""
TRANSLATION EVALUATION
"""
def eval(top_matches, dico, n_gold_std):
    results = []
    matching_at_k = {}

    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        
        listed_dico = [ x for sub in dico[:, 1][:, None].cpu().numpy() for x in sub ]
        n_relevant = 0

        for values in top_k_matches.cpu().numpy():
            for sub_val in values:
                if sub_val in listed_dico:
                    n_relevant += 1
                    break
                    
        print("listed_dico :", len(listed_dico))
        print("top_k_matches :", len(top_k_matches))
        print("n_relevant :", n_relevant)

        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).numpy()
        print("_matching ", len(_matching))

        # allow for multiple possible translation
        matching = {}
        trans_match = []
        for i, src_id in enumerate(dico[:, 0]):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
            trans_match.append((src_id, min(matching.get(src_id, 0) + _matching[i], 1)))

        print("matching ", len(matching))
        print("trans_match ", len(trans_match))
        matching_at_k[k] = trans_match

        # evaluate presicion@k
        #precision_at_k = 100 * np.mean(list(matching.values()))
        # save precision value at results
        #results.append(('precision at k={}: {}'.format(k, precision_at_k)))

        # evaluate recall@k
        #recall_at_k = 100 * np.sum(list(matching.values()))/n_gold_std
        # save recall value at results
        #results.append(('recall at k={}: {}'.format(k, recall_at_k)))

        # evaluate precision@k
        precision_at_k = 100 * np.sum(list(matching.values())) / n_relevant
        #logger.info("%i source words - %s - Precision at k = %i: %f" %
        #            (len(matching), method, k, precision_at_k))
        results.append(('recall_at_%i' % k, precision_at_k))

        # evaluate recall@k
        recall_at_k = 100 * np.mean(list(matching.values()))
        #logger.info("%i source words - %s - Recall at k = %i: %f" %
        #            (len(matching), method, k, recall_at_k))
        results.append(('precision_at_%i' % k, recall_at_k))

        # evaluate f1-score@k
        if precision_at_k == 0 and recall_at_k == 0:
            f1score_at_k = 0
        else:
            f1score_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        # save f1-score value at results
        results.append(('f1-score at k={}: {}'.format(k, f1score_at_k)))

    return results, dico[:, 0], top_k_matches, matching_at_k

"""
EXPORT TRANSLATIONS RESULT
"""
def export_pair_translation(word2id, id2word, src_dico, top_k_matches, matching_at_k, file_wrong, file_correct):
    wrong_trans_path = file_wrong
    correct_trans_path = file_correct

    wrong_file = open(wrong_trans_path, 'w+', encoding='utf-8')
    correct_file = open(correct_trans_path, 'w+', encoding='utf-8')

    for k in [1, 5, 10]:
        wrong_file.write("Pair translation for k={}\n".format(k))
        correct_file.write("Pair translation for k={}\n".format(k))

        i = 0
        for key, val in matching_at_k[k]:
            key = int(key)
            if val == 0:
                # write to file contains all wrong predictions
                id_trans = top_k_matches[i][:k].tolist()
                wrong_file.write("{} {}\n".format(key, id_trans))

                trans = list(id2word[idx] for idx in id_trans)
                wrong_file.write("{} {}\n\n".format(id2word[key], trans))
                i += 1
            elif val == 1:
                # write to file contains all correct predictions
                id_trans = top_k_matches[i][:k].tolist()
                #print(key, id_trans)
                correct_file.write("{} {}\n".format(key, id_trans))

                trans = list(id2word[idx] for idx in id_trans)
                correct_file.write("{} {}\n\n".format(id2word[key], trans))
                i += 1

        wrong_file.write("\n\n")
        correct_file.write("\n\n")
  
    wrong_file.close()
    correct_file.close()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Help: python3 eval_pseudo_bilingual.py <emb_file> <dict_file> <file_wrong> <file_corect>')
        sys.exit()
    
    EMB_FILE = VECTORS_PATH + sys.argv[1]
    DICT_PATH = DICTIONARY_PATH + sys.argv[2]
    FILE_WRONG = RESULTS_PATH + sys.argv[3]
    FILE_CORRECT = RESULTS_PATH + sys.argv[4]

    word2id, id2word, embeddings = load_embedding(EMB_FILE)
    dico, n_gold_std = load_dictionary(DICT_PATH, word2id)
    top_matches = get_word_translation(embeddings, word2id, dico)
    results, src_dico, top_k_matches, matching_at_k = eval(top_matches, dico, n_gold_std)

    print(results)

    export_pair_translation(word2id, id2word, src_dico, top_k_matches, matching_at_k, FILE_WRONG, FILE_CORRECT)
