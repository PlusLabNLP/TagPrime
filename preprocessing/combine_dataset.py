import argparse
import json
from collections import Counter, defaultdict
import os
import ipdb
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_folder", type=str, required=True)
parser.add_argument("--file_prefix", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
args = parser.parse_args()


all_data = []
for f_name in glob.glob('{}/{}_*'.format(args.input_folder, args.file_prefix)):
    print(f_name)
    i = [json.loads(line) for line in open(f_name, 'r')]
    all_data.extend(i)

with open(args.output_path, 'w') as f:
    for doc in all_data:
        f.write(json.dumps(doc) + '\n')