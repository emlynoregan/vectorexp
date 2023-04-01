# In this program we create a dumb_index Docs object from a file specified on the command line, by breaking it into paragraphs, 
# and then creating embeddings from the paragraphs. We write this to file.

import argparse
from dumbvector.docs import file_docs_exists, make_docs_v1, get_docs_file_writer, file_to_docs
from dumbvector.util import time_function
from dumbvector.bsonutil import shrink_ndarrays
import base64
import json
import time
import os
import numpy as np

@time_function
def update_docs(filename, docs_path, shrink):
    # remove any path info from the filename
    print (f'filename: {filename}')
    docs_name = os.path.basename(filename)
    # remove the extension
    docs_name = os.path.splitext(docs_name)[0]
    source_path = os.path.dirname(filename)
    docs = file_to_docs(docs_name, source_path)
    # print (f'file: {filename} docs: {docs}')

    if shrink:
        print (f'shrinking docs')
        docs = shrink_ndarrays(docs)

    print (f'writing docs to file')
    writer = get_docs_file_writer(docs_path)
    writer(docs)

def main():
    # usage: python update_docs.py <source_path> <docs_path> [shrink]

    parser = argparse.ArgumentParser()

    parser.add_argument('source_path', help='path to the source file or directory')
    parser.add_argument('docs_path', help='path to the docs directory for writing the docs files')
    parser.add_argument('--shrink', help='shrink the docs to int8', action='store_true')

    args = parser.parse_args()

    source_path = args.source_path
    docs_path = args.docs_path
    shrink = args.shrink

    sourcepath_is_dir = os.path.isdir(source_path)

    if sourcepath_is_dir:
        # get all the files in the directory
        filenames = [os.path.join(source_path, f) for f in os.listdir(source_path)]
        for filename in filenames:
            update_docs(filename, docs_path, shrink)
    else:
        update_docs(source_path, docs_path, shrink)

    print ("done")

if __name__ == '__main__':
    main()

