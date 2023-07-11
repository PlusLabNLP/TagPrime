from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
import time
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
import ipdb

def convert(input_file, output_file, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as f:                  
        with open(output_file, 'w',encoding='utf-8') as f1:
            for line in f.readlines():
                json_read=json.loads(line)
                doc_id = json_read["doc_key"]
                if json_read["sentences"]:
                    assert len(json_read["ners"]) == len(json_read["sentences"])
                    for k in range(len(json_read["ners"])):
                        wnd_entities=[]
                        wnd_id = "{}-S{}".format(doc_id, k)
                        for q in range(len(json_read["ners"][k])): #每一个句子对应多少个entity
                            wnd_entities_={}
                            wnd_entities_["id"]=str(wnd_id+"-E"+ str(q))
                            start=json_read["ners"][k][q][0]
                            end=json_read["ners"][k][q][1]+1
                            wnd_entities_["start"]=start
                            wnd_entities_["end"]=end
                            wnd_entities_["entity_type"]=json_read["ners"][k][q][2]
                            wnd_entities_["mention_type"]=str("UNK")
                            text1= [str(json_read["sentences"][k][i]) for i in range(start,end)]
                            wnd_entities_["text"]=' '.join(text1)
                            wnd_entities.append(wnd_entities_)

                        wnd_tokens=json_read["sentences"][k]
                        pieces1 = [tokenizer.tokenize(t) for t in wnd_tokens]
                        word_lens = [len(p) for p in pieces1] 

                        wnd_ = {
                                'doc_id': doc_id,
                                'wnd_id': wnd_id,
                                'entity_mentions': wnd_entities,
                                'relation_mentions':[],
                                'event_mentions':[],
                                'entity_coreference': [],
                                'event_coreference': [],
                                'tokens': wnd_tokens,
                                'pieces': [p for w in pieces1 for p in w],
                                'token_lens': word_lens,
                                'sentence': ' '.join(wnd_tokens),
                                'sentence_starts': [0],  
                        }
                        f1.write(json.dumps(wnd_) + '\n')
                else:
                    ipdb.set_trace()
        f1.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file')
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-b', '--bert', help='BERT model name', default='bert-large-cased')
    args = parser.parse_args()
    model_name = args.bert
    if model_name.startswith('bert-'):
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert,
                                                       do_lower_case=False)
    elif model_name.startswith('roberta-'):
        bert_tokenizer = RobertaTokenizer.from_pretrained(args.bert, do_lower_case=False, add_prefix_space=True)
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert, do_lower_case=False, use_fast=False)
    
    convert(args.input, args.output, bert_tokenizer)