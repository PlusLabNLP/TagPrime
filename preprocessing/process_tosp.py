from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
from collections import defaultdict
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

def read_file(input_dir, split):
    
    with open(os.path.join(input_dir, split, "label")) as fp:
        lines = fp.readlines()
    intents = list(map(lambda x: x.strip(), lines))
    
    with open(os.path.join(input_dir, split, "seq.in")) as fp:
        lines = fp.readlines()
    inputs = list(map(lambda x: x.strip(), lines))
        
    with open(os.path.join(input_dir, split, "seq.out")) as fp:
        lines = fp.readlines()
    outputs = list(map(lambda x: x.strip(), lines))
    
    assert len(intents) == len(inputs) == len(outputs)
    
    if split == "dev" or split == "test":
        with open(os.path.join(input_dir, split, "pred.txt")) as fp:
            lines = fp.readlines()
        preds = list(map(lambda x: x.split()[0], lines))
        assert len(intents) == len(preds)
    else:
        preds = intents

    data = list(zip(intents, inputs, outputs, preds))
    return data
    
def convert(data, output_path, split):
    
    slot_type_dict = defaultdict(int)
    intent_set = set()
    pattern_dict = defaultdict(set)
    with open(output_path+".json", 'w', encoding='utf-8') as fp:
        for idx, (intent_, input_, output_, pred_) in enumerate(data):
            
            tokens = input_.split()
            labels = output_.split()
            assert len(tokens) == len(labels)
            
            slots = tag_path_to_span(labels)
            
            intent_set.add(intent_)
            for _, _, t, in slots:
                slot_type_dict[t] += 1
                pattern_dict[intent_].add(t)
            
            ex = {
                'doc_id': f"{split}-{idx}",
                'wnd_id': f"{split}-{idx}-0",
                'tokens': tokens, 
                'labels': labels, 
                'intent': intent_, 
                'p_intent': pred_, 
                'slots': slots, 
                'sentence': input_, 
                'label': output_, 
            }
            fp.write(json.dumps(ex) + '\n')
    
    
    with open (output_path+"_intent.json", 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(sorted(intent_set), indent=4))
    
    with open (output_path+"_slot_type.json", 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(slot_type_dict, indent=4))
    
    pattern_dict = {t: sorted(pattern_dict[t]) for t in sorted(pattern_dict)}
    with open (output_path+"_pattern.json", 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(pattern_dict, indent=4))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='Input mtop dir', required=True)
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-s', '--split', required=True)
    args = parser.parse_args()
    
    data = read_file(args.input_dir, args.split)
    convert(data,  args.output, args.split)
