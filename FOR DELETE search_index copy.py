# In this program we open an index, sort it by similarity to a query, then
# return the top N chunks.

import argparse
from dumb_vector_s3 import bytes_to_vector, C_VECTORTYPE_INT8, read_dumb_index_from_file, \
    get_chunks_from_dumb_index, read_chunk_from_s3, sort_dumb_index_by_similarity
from top_k import top_k_similar
from cupy_top_k import cupy_top_k_similar
import openai
from openai.embeddings_utils import get_embedding
import base64
import json
import boto3
import time
import os

def time_function(func):
    def timed(*args, **kw):
        ts = time.time()
        try:
            result = func(*args, **kw)
        finally:
            te = time.time()

            print ('%r  %2.2f sec' % \
                (func.__name__, te-ts))
        return result

    return timed

def read_credentials():
    with open('credentials.json', 'r') as f:
        return json.load(f)
    

def main():
    # usage: python search_index.py index_filename query [num_results]

    parser = argparse.ArgumentParser()

    parser.add_argument('index_filename', help='the name of the index file to use')
    parser.add_argument('query', help='the query to search for')
    # default to 20 results
    parser.add_argument('num_results', help='the number of results to return', nargs='?', default=20)

    args = parser.parse_args()

    index_filename = args.index_filename
    query = args.query
    num_results = int(args.num_results)

    # read the credentials
    credentials = read_credentials()

    s3_session = boto3.Session(
        aws_access_key_id=credentials['aws_access_key_id'],
        aws_secret_access_key=credentials['aws_secret_access_key'],
        region_name=credentials['region_name'],
    )
    s3_bucket = credentials['s3_bucket']
    openai.api_key = credentials['openaikey']

    # read the index
    index = time_function(read_dumb_index_from_file)(index_filename)

    # number of triples in the index
    print ("num triples:", len(index.get('triples')))

    # get the embedding for the query
    embedding = time_function(get_embedding)(query, engine="text-embedding-ada-002")

    # # get chunks to cache the chunks
    # time_function(get_chunks_from_dumb_index)(s3_session, s3_bucket, index, 0, 1)

    # do the basic search
    sorted_index1 = time_function(sort_dumb_index_by_similarity)(index, embedding)

    # now do numpy based top-k
    sorted_index2 = time_function(top_k_similar)(index, embedding, num_results)

    # now do cupy based top-k
    sorted_index3 = time_function(cupy_top_k_similar)(index, embedding, num_results)

    # compare the results
    triples_1 = sorted_index1.get('triples')[:num_results]
    triples_2 = sorted_index2.get('triples')
    triples_3 = sorted_index3.get('triples')

    for ix, (t1, t2, t3) in enumerate(zip(triples_1, triples_2, triples_3)):
        if t1 != t2:
            print ("mismatch at", ix, t1, t2)
        if t1 != t3:
            print ("mismatch at", ix, t1, t3)

    # get the chunks
    chunks = time_function(get_chunks_from_dumb_index)(s3_session, s3_bucket, sorted_index3, 0, num_results)

    # again to get cached timing
    chunks = time_function(get_chunks_from_dumb_index)(s3_session, s3_bucket, sorted_index3, 0, num_results)

    # print the results
    # just print "text" for each chunk
    for ix, chunk in enumerate(chunks):
        print (ix, chunk['text'])

    print ("done")

if __name__ == '__main__':
    main()

