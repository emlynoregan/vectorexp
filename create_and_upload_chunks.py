# In this program we open a file specified on the command line, break it into paragraphs, and then create chunks from the paragraphs.  
# We then write the chunks to S3.  

import argparse
from dumbvector import write_chunks_to_s3, vector_to_bytes, C_VECTORTYPE_INT8
from openai.embeddings_utils import get_embedding
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

@time_function
def create_chunks_from_paragraphs(paragraphs):
    # create chunks from the paragraphs
    chunks = []
    for index, p in enumerate(paragraphs):
        print (f'creating chunk {index}')
        # create a chunk from the paragraph and embedding
        chunk = {
            'text': p
        }

        # get the embedding for the paragraph
        embedding = get_embedding(p, engine="text-embedding-ada-002")

        # we need to shrink the embedding to single byte elements
        embedding_bytes = vector_to_bytes(embedding, C_VECTORTYPE_INT8)                
        chunk["embedding"] = base64.b64encode(embedding_bytes).decode('utf-8')
        chunk["vector_type"] = C_VECTORTYPE_INT8        

        chunks.append(chunk)

    return chunks

def read_credentials():
    with open('credentials.json', 'r') as f:
        return json.load(f)

@time_function
def create_and_upload_chunks(filename, s3_path, s3_session, s3_bucket):
    # remove any path info from the filename
    filename_no_path = os.path.basename(filename)

    # read the file (utf-8)
    print (f'reading {filename}')
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # break it into paragraphs. They are separated by two newlines
    print (f'breaking {filename} into paragraphs')
    paragraphs = text.split('\n\n')

    # break up any paragraphs that are too long (longer than 2000 characters)
    print (f'breaking up any paragraphs that are too long')
    new_paragraphs = []
    for p in paragraphs:
        if len(p) > 2000:
            print (f'paragraph is too long: {len(p)} characters')
            # now break it into chunks of 2000 characters and the remainder
            while p:
                new_paragraphs.append(p[:2000])
                p = p[2000:]
        else:
            new_paragraphs.append(p)

    # remove any empty paragraphs
    paragraphs = [p for p in new_paragraphs if p]
    print (f'found {len(paragraphs)} paragraphs')

    chunks_filename = f'chunks_{filename_no_path}.json'
    chunks_path = os.path.join('chunks', chunks_filename)
    chunks_file_exists = os.path.exists(chunks_path)

    if not chunks_file_exists:
        # create chunks from the paragraphs
        chunks = create_chunks_from_paragraphs(paragraphs)

        # write the chunks to data/chunks_<filename>.json
        print (f'writing {len(chunks)} chunks to file')
        with open(chunks_path, 'w') as f:
            json.dump(chunks, f)

        # write the chunks to S3
        print (f'writing {len(chunks)} chunks to S3')
        time_function(write_chunks_to_s3)(s3_session, s3_bucket, s3_path, chunks_filename, chunks)
    else:
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)

    return chunks

def main():
    # usage: python create_and_upload_chunks.py [<filename>] [<s3_path>]

    parser = argparse.ArgumentParser()

    parser.add_argument('filename', help='name of file to read')
    parser.add_argument('s3_path', help='path to write chunks to S3', nargs='?', default='')

    args = parser.parse_args()

    filename = args.filename
    s3_path = args.s3_path or "vectorexp/chunks"

    # read the credentials
    credentials = read_credentials()

    s3_session = boto3.Session(
        aws_access_key_id=credentials['aws_access_key_id'],
        aws_secret_access_key=credentials['aws_secret_access_key'],
        region_name=credentials['region_name'],
    )
    s3_bucket = credentials['s3_bucket']
    openai.api_key = credentials['openaikey']

    filename_is_dir = os.path.isdir(filename)

    if filename_is_dir:
        # get all the files in the directory
        filenames = [os.path.join(filename, f) for f in os.listdir(filename)]
        for filename in filenames:
            create_and_upload_chunks(filename, s3_path, s3_session, s3_bucket)
    else:
        create_and_upload_chunks(filename, s3_path, s3_session, s3_bucket)

    print ("done")

if __name__ == '__main__':
    main()

