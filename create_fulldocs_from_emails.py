# This program opens the large "emails.csv" file containing the enron emails, and creates a docs file for each chunk of 1000 emails.
# The columns are "file" and "message", these will be placed in the docs.
# We also add an sbert embedding, although it's probably not very useful.

import argparse
from dumbvector.docs import file_docs_exists, make_docs_v1, get_docs_file_writer
from openai.embeddings_utils import get_embedding
import json
import time
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

C_MODEL = None

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
def create_doclist_from_paragraphs(paragraphs):
    # create chunks from the paragraphs
    doclist = []
    for index, p in enumerate(paragraphs):
        print (f'creating doc {index}')
        # create a chunk from the paragraph and embedding
        doc = {
            'text': p,
            'ix': index,
            'embedding': np.array(get_embedding(p, engine="text-embedding-ada-002"))
        }

        doclist.append(doc)

    return doclist

def read_credentials():
    with open('credentials.json', 'r') as f:
        return json.load(f)

@time_function
def create_docs(filename, docs_path):
    # remove any path info from the filename
    docs_name = os.path.basename(filename)
    print (f'creating docs {docs_name} from file {filename}')

    docs_exists = file_docs_exists(docs_name, docs_path)

    if docs_exists:
        print (f'{docs_name} already exists')
        return

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

    # create doclist from the paragraphs
    doclist = create_doclist_from_paragraphs(paragraphs)

    # create the Docs object
    print (f'creating docs object')
    d = make_docs_v1(docs_name, doclist)

    # write the Docs object to file
    print (f'writing docs to file')
    writer = get_docs_file_writer(docs_path)
    writer(d)

def main():
    # usage: python create_fulldocs_for_emails.py <emails_filename> <docs_path>

    parser = argparse.ArgumentParser()

    parser.add_argument('emails_filename', help='the filename of the emails.csv file')
    parser.add_argument('docs_path', help='the path to the docs directory')

    args = parser.parse_args()

    emails_filename = args.emails_filename
    docs_path = args.docs_path

    global C_MODEL
    C_MODEL = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='modelcache')

    # read the emails file as a csv
    print (f'reading {emails_filename}')
    emails = pd.read_csv(emails_filename, encoding='utf-8')
    # import csv
    # with open(emails_filename, 'r', encoding='utf-8') as f:
    #     reader = csv.reader(f)
    #     emails = list(reader)

    C_SIZE = 1000
    t_start = time.time()
    for i in range(0, len(emails), C_SIZE):
        docname = f'emails-{i}'

        if not file_docs_exists(docname, docs_path):
            print (f'processing emails {i}')
            nextemails = emails.iloc[i:i+C_SIZE]

            if i > 0:
                # calculate the eta
                t_now = time.time()
                t_elapsed = t_now - t_start
                t_remaining = (t_elapsed / i) * (len(emails) - i)
                print (f'elapsed: {t_elapsed} seconds, remaining: {t_remaining} seconds')

            # print (f'nextemails: {len(nextemails)}: {nextemails.iloc[0]}')

            embeddings = C_MODEL.encode(nextemails['message'].to_numpy())
            # print (f'embeddings: {embeddings}')

            doclist = doclist = [{
                'file': f,
                'message': m,
                'embedding': e
            } for f, m, e in zip(nextemails['file'], nextemails['message'], embeddings)]

            # print (f'doclist: {len(doclist)}: {doclist[0]}')
            # exit()

            # create the Docs object
            print (f'creating docs object')
            d = make_docs_v1(f'emails-{i}', doclist)

            # write the Docs object to file
            print (f'writing docs to file')
            writer = get_docs_file_writer(docs_path)
            writer(d)

    print ("done")

if __name__ == '__main__':
    main()

