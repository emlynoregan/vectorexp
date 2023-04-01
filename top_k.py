import numpy as np

def top_k_similar(dumb_index, vector, k):
    D = len(vector)

    #1: Change datatypes to match numpy.
    n_vector = np.array(vector) # this is a 1 X D vector

    triples = dumb_index["triples"]
    vectors = [triple[0] for triple in triples]
    # vectors is a list of 1 X D lists, each a vector
    # we want them to be a D X N matrix
    n_index_matrix = np.array(vectors).T # this is a D X N matrix

    cosine_similarities = np.dot(n_vector, n_index_matrix) # this is a 1 X N vector

    # now we want the top k indices
    ind = np.argpartition(cosine_similarities, -k)[-k:]

    # get the top k (triple, similarity) pairs
    top_k = [(triples[i], cosine_similarities[i]) for i in ind]

    # sort the top k triples
    sorted_top_k = sorted(top_k, key=lambda item: item[1], reverse=True)

    sorted_top_k_triples = [item[0] for item in sorted_top_k]

    return {
        "triples": sorted_top_k_triples,
        "paths": dumb_index["paths"],
        "file_pairs": dumb_index["file_pairs"]
    }
