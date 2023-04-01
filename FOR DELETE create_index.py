# In this program we open some chunk files and create indices from them. We then write the indices to S3.

import argparse
from dumbvector import bytes_to_vector, C_VECTORTYPE_INT8, create_dumb_index, write_dumb_index_to_file
import openai
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
    # usage: python create_index.py index_filename <s3_chunk_path>, <s3_chunk_path>, ..., <s3_chunk_path>

    parser = argparse.ArgumentParser()

    parser.add_argument('index_filename', help='the name of the index file to create')
    parser.add_argument('s3_chunk_path', help='the S3 chunk paths to process', nargs='+')

    args = parser.parse_args()

    index_filename = args.index_filename
    s3_chunk_paths = args.s3_chunk_path

    # read the credentials
    credentials = read_credentials()

    s3_session = boto3.Session(
        aws_access_key_id=credentials['aws_access_key_id'],
        aws_secret_access_key=credentials['aws_secret_access_key'],
        region_name=credentials['region_name'],
    )
    s3_bucket = credentials['s3_bucket']
    openai.api_key = credentials['openaikey']

    # create the index
    def get_vector(chunk):
        embedding_bytes_b64 = chunk['embedding']
        embedding_bytes = base64.b64decode(embedding_bytes_b64)
        vector_type = chunk['vector_type']
        embedding = bytes_to_vector(embedding_bytes, vector_type)
        return embedding

    index = time_function(create_dumb_index)(s3_session, s3_bucket, s3_chunk_paths, get_vector)

    triples = index.get('triples')

    if not triples:
        print ("no triples found")
        return
    
    num_dimensions = len(triples[0][0])

    # write the index to local file
    write_dumb_index_to_file(index_filename, index, C_VECTORTYPE_INT8, num_dimensions)

    print ("done")

if __name__ == '__main__':
    main()

