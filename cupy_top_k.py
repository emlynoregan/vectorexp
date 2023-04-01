import numpy as np
import cupy as cp
import time

def cupy_top_k_similar(dumb_index, vector, k):
    start_time = time.time()
 
    def print_time(message):
        print(message + ": " + str(time.time() - start_time))

    print_time("start")

    D = len(vector)

    triples = dumb_index["triples"]
    vectors = [triple[0] for triple in triples]

    print_time("prepped")

    def cupy_top_k_similar_1024(vectors_1024):
        # process the vectors in chunks of 1024


        vector_cupy = cp.asarray(vector)

        print_time("vector_cupy")

        # vectors_np = np.array(vectors_1024)
        # vectors_np_t = vectors_np.T
        # vectors_cupy = cp.asarray(vectors_np_t)
        vectors_cupy = cp.asarray(vectors_1024).T

        print_time("vectors_cupy")

        # get the x dim of vectors_np_t
        # x_dim = vectors_np_t.shape[0]
        y_dim = vectors_cupy.shape[1]

        output_cupy = cp.zeros((y_dim, 1), dtype=np.float64)

        print_time("output_cupy")

        # with cp.cuda.Device() as dev:
        ret = vector_cupy.dot(vectors_cupy, out=output_cupy)
            # dev.synchronize()
        
        print_time("dot")
        # print (ret)

        # flattened_output_cupy = output_cupy.flatten()
        # print_time("flattened_output_cupy")
        # np_retval = flattened_output_cupy.get()
        # print_time("np_retval")
        # retval = np_retval.tolist()
        
        # # # convert output_cupy to a plain list and return
        # # list_of_lists = output_cupy.get().tolist()
        # # retval = [item[0] for item in list_of_lists]

        # print_time("retval")
        # print ("retval:" + retval)

        return output_cupy.flatten().get()

    # # break vectors in blocks of 1024
    # vector_blocks = [vectors[i:i + 1024] for i in range(0, len(vectors), 1024)]

    # # process each block
    # cosine_similarities = []
    # for vector_block in vector_blocks:
    #     block_cosine_similarities = cupy_top_k_similar_1024(vector_block)
    #     cosine_similarities.extend(block_cosine_similarities)

    cosine_similarities = cupy_top_k_similar_1024(vectors)

    # now we want the top k indices
    ind = np.argpartition(cosine_similarities, -k)[-k:]
    print_time("ind")

    # get the top k (triple, similarity) pairs
    top_k = [(triples[i], cosine_similarities[i]) for i in ind]
    print_time("top_k")

    # sort the top k triples
    sorted_top_k = sorted(top_k, key=lambda item: item[1], reverse=True)
    print_time("sorted_top_k")

    sorted_top_k_triples = [item[0] for item in sorted_top_k]
    print_time("sorted_top_k_triples")

    return {
        "triples": sorted_top_k_triples,
        "paths": dumb_index["paths"],
        "file_pairs": dumb_index["file_pairs"]
    }
