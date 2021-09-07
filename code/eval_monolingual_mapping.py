import sys
import torch
import numpy as np

DUMP_PATH = 'dumped/debug/'
# VECTORS_PATH = '../data/vectors/'
DICTIONARY_PATH = 'data/crosslingual/dictionaries/'
# RESULTS_PATH = '../result/'

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

def load_dictionary(path, word2id1, word2id2):
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
            if (word1 in word2id1) and (word2 in word2id2):
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    print('Not found pair words: ', not_found)
    print('Not found source words: ', not_found1)
    print('Not found target words: ', not_found2)
    print('Number of found pairs words: ', len(pairs))
    
    # sort dictionary by source words frequency
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]
    
    return dico, n_gold_std

def get_nn_avg_dist(emb, query, knn):
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1))
    all_distances = torch.cat(all_distances)

    return all_distances.numpy()

def get_word_translation(emb1, emb2, method, dico):
    """
    Get all word translations of source words using nearest neighbor method
    """
    # normalize embedding
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbor
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
    
    # cross domain score local scaling
    elif method.startswith == 'csls_knn_':
        # average distance to k nearest neighbors
        knn = int(method[len('csls_knn_'):])
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # gueries /scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])
    
    else:
        print('method invalid')

    top_matches = scores.topk(10, 1, True)[1]

    return top_matches

"""
TRANSLATION EVALUATION
"""
def eval(top_matches, dico, n_gold_std):
    results = []
    matching_at_k = {}

    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
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
        precision_at_k = 100 * np.mean(list(matching.values()))
        # save precision value at results
        results.append(('precision at k={}: {}'.format(k, precision_at_k)))

        # evaluate recall@k
        recall_at_k = 100 * np.sum(list(matching.values()))/n_gold_std
        # save recall value at results
        results.append(('recall at k={}: {}'.format(k, recall_at_k)))

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
def export_pair_translation(id2word1, id2word2, top_k_matches, matching_at_k, file_wrong, file_correct):
    wrong_trans_path = DUMP_PATH + file_wrong
    correct_trans_path = DUMP_PATH + file_correct

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

                trans = list(id2word2[idx] for idx in id_trans)
                wrong_file.write("{} {}\n\n".format(id2word1[key], trans))
                i += 1
            elif val == 1:
                # write to file contains all correct predictions
                id_trans = top_k_matches[i][:k].tolist()
                print(key, id_trans)
                correct_file.write("{} {}\n".format(key, id_trans))

                trans = list(id2word2[idx] for idx in id_trans)
                correct_file.write("{} {}\n\n".format(id2word1[key], trans))
                i += 1

        wrong_file.write("\n\n")
        correct_file.write("\n\n")
  
    wrong_file.close()
    correct_file.close()

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Help: python3 eval_monolingual_mapping.py <emb1_file> <emb2_file> <dict_file> <file_wrong> <file_corect>')
        sys.exit()
    
    EMB1_FILE = DUMP_PATH  + sys.argv[1]
    EMB2_FILE = DUMP_PATH + sys.argv[2]
    DICT_PATH = DICTIONARY_PATH + sys.argv[3]
    FILE_WRONG = sys.argv[4]
    FILE_CORRECT = sys.argv[5]

    word2id1, id2word1, embeddings1 = load_embedding(EMB1_FILE)
    word2id2, id2word2, embeddings2 = load_embedding(EMB2_FILE)
    dico, n_gold_std = load_dictionary(DICT_PATH, word2id1, word2id2)

    for method in ['nn', 'csls_knn_10']:
        top_matches = get_word_translation(embeddings1, embeddings2, method, dico)
        results, src_dico, top_k_matches, matching_at_k = eval(top_matches, dico, n_gold_std)

        print(results)

        wrong_file =  method + '-' + FILE_WRONG
        correct_file = method + '-' + FILE_CORRECT

        export_pair_translation(id2word1, id2word2, top_k_matches, matching_at_k, wrong_file, correct_file)

# python3 eval_monolingual_map.py vectors-jv.txt vectors-su.txt jv-su-test.txt wrong-trans.txt correct-trans.txt