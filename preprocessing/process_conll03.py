from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
from transformers import BertTokenizer, RobertaTokenizer, BartTokenizer, T5Tokenizer,AutoTokenizer
import time
import ipdb


def tag_path_to_span(path):
    mentions = []
    cur_mention = None
    for j, tag in enumerate(path):
        if tag == 'O':
            prefix = tag = 'O'
        else:
            prefix, tag = tag.split('-', 1)
        if prefix == 'B':
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = [j, j + 1, tag]
        elif prefix == 'I':
            if cur_mention is None:
                # treat it as B-*
                cur_mention = [j, j + 1, tag]
            elif cur_mention[-1] == tag:
                cur_mention[1] = j + 1
            else:
                # treat it as B-*
                mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
        else:
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = None
    if cur_mention:
        mentions.append(cur_mention)
    #ipdb.set_trace()
    return mentions

def read_file(file_path, captalize):
    samples = []
    tokens = []
    tags = []
    with open(file_path, 'r') as fb:
        for line in fb:
            line = line.strip('\n')

            if line == '-DOCSTART- -X- -X- O':
                pass
            elif line =='':
                if len(tokens) != 0:
                    samples.append((tokens, tags))
                    tokens = []
                    tags = []
            else:
                contents = line.split(' ')
                if captalize == 'Original':
                    tokens.append(contents[0])
                elif captalize == 'Uncased':
                    tokens.append(contents[0].lower())
                elif captalize == 'Cased_first':
                    tokens.append(contents[0][0]+contents[0][1:].lower())
                tags.append(contents[-1])
    
    return samples

def convert(data, output_path, tokenizer, valid_types, name):
    dic_type={}
    w = open(output_path+".json", 'w', encoding='utf-8')
  
    for i in range(len(data)):
        wnd_tokens = data[i][0] # the first element of a tuple
        pieces = [tokenizer.tokenize(t) for t in wnd_tokens]
        word_lens = [len(p) for p in pieces]

        # convert tag to span
        wnd_entities = tag_path_to_span(data[i][1])
        wnd_entities_ = []
        cnt = 0
        for j, (start, end, entity_type) in enumerate(wnd_entities):
            if entity_type not in valid_types:
                continue
            entity_id = '{}-{}-E{}'.format(name, i, cnt)
            text=' '.join(wnd_tokens[start:end])
            if(len(text)>0):
                cnt += 1
                if entity_type not in dic_type:
                    dic_type[entity_type]=1
                else:
                    dic_type[entity_type]=dic_type[entity_type]+1
                entity = {
                    'id': entity_id,
                    'start': start, 'end': end,
                    'entity_type': entity_type,
            
                    # Mention types are not included in DyGIE++'s format
                    'mention_type': 'UNK',
                    'text': text}
                wnd_entities_.append(entity)
            else:
                ipdb.set_trace()

        wnd_ = {
            'doc_id': name,
            'wnd_id': "{}-{}".format(name, i),
            'entity_mentions': wnd_entities_,
            'relation_mentions': [],
            'event_mentions': [],
            'entity_coreference': [],
            'event_coreference': [],
            'tokens': wnd_tokens,
            'pieces': [p for w in pieces for p in w],
            'token_lens': word_lens,
            'sentence': ' '.join(wnd_tokens),
            'sentence_starts': [0],
            'valid_types': valid_types          
        }
        w.write(json.dumps(wnd_) + '\n')
    with open (output_path+"_type.json", 'w', encoding='utf-8') as w1:
        w1.write(json.dumps(dic_type))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-b', '--bert', help='BERT model name', default='bert-large-cased')
    parser.add_argument('-i', '--input', help='Input CoNLL file', required=True)
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('--valid_types', nargs='+',
        choices=["ORG", "MISC", "PER", "LOC"], default=["ORG", "MISC", "PER", "LOC"])
    parser.add_argument('--captalize', type=str, default="Original", choices=["Original", "Uncased", "Cased_first"])
    args = parser.parse_args()
    model_name = args.bert
    if model_name.startswith('bert-'):
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    elif model_name.startswith('roberta-'):
        bert_tokenizer = RobertaTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False,
                                                       add_prefix_space=True)
    elif model_name.startswith("facebook/bart-"):
        bert_tokenizer = BartTokenizer.from_pretrained(args.bert)
    elif model_name.startswith("t5-"):
        bert_tokenizer = T5Tokenizer.from_pretrained(args.bert)
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, do_lower_case=False, do_fast=False)
    
    data = read_file(args.input, args.captalize)
    output_path = args.output
    convert(data, output_path, bert_tokenizer, args.valid_types, args.name)
