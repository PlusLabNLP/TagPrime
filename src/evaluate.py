import os, sys, json, logging, time, pprint, tqdm
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, AdamW,
                          RobertaTokenizer,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup)
from model import StructuralModel
from data import EAEDataset, REStage1Dataset, REDataset, EDDataset
from scorer import *
import copy
import ipdb
from pattern import *
from pathlib import Path
from eval_RE.run_eval import get_REscore

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True )
parser.add_argument('-w', '--weight_path', required=True)

parser.add_argument('-t', '--test_file', type=str, required=True)
parser.add_argument('--eval_batch', type=int)
parser.add_argument('--max_length', type=int)
parser.add_argument('--use_ner_filter', default=False, action='store_true')
parser.add_argument('--write_file_name', type=str, default='predictions_test.json')

args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config = Namespace(**config)

if args.eval_batch is not None:
    config.eval_batch_size = args.eval_batch
if args.max_length is not None:
    config.max_length = args.max_length

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)

# set GPU device
if config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# load weight
map_location = f'cuda:{config.gpu_device}'
print(f"Loading model from {args.weight_path}")
state = torch.load(args.weight_path, map_location=map_location)
model_folder = Path(args.weight_path).parent.absolute()
gpu_device = config.gpu_device

tokenizer = state['tokenizer']

# load dataset
print('==============Prepare Test Set=================')
if config.task == 'eae':
    config.use_unified_label = getattr(config, "use_unified_label", True)
    test_set = EAEDataset(args.test_file, tokenizer, max_length=config.max_length)
elif config.task == 're_stage1':
    test_set = REStage1Dataset(args.test_file, tokenizer, max_length=config.max_length, use_pred_ner=True)
elif config.task == 're':
    test_set = REDataset(args.test_file, tokenizer, max_length=config.max_length, use_pred_ner=True)
elif config.task == 'ed':
    test_set = EDDataset(args.test_file, tokenizer, max_length=config.max_length)

vocabs = state['vocabs']

test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = StructuralModel(config, tokenizer, vocabs, dataset=config.dataset)
model.load_state_dict(state['model'])
model.cuda(device=gpu_device)
model.eval()

# Evaluation 
progress = tqdm.tqdm(total=test_batch_num, ncols=75, 
                    desc='Test')
write_output = []
test_gold_arg_id, test_gold_arg_cls, test_pred_arg_id, test_pred_arg_cls = [], [], [], []
for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                        shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    if config.task == 'eae':
        preds, input_text = model.predict(batch, use_ner_filter=args.use_ner_filter)
        for wid, tri, role, pred, text, token in zip(batch.wnd_ids, batch.triggers, batch.roles, preds, input_text, batch.tokens):
            gold_arg_id = [(wid, tri[2], r[1][0], r[1][1]) for r in role]
            gold_arg_cls = [(wid, tri[2], r[1][0], r[1][1], r[1][2]) for r in role]
            pred_arg_id = [(wid, tri[2], r[0], r[1]) for r in pred]
            pred_arg_cls = [(wid, tri[2], r[0], r[1], r[2]) for r in pred]
            
            test_gold_arg_id.extend(gold_arg_id)
            test_gold_arg_cls.extend(gold_arg_cls)
            test_pred_arg_id.extend(pred_arg_id)
            test_pred_arg_cls.extend(pred_arg_cls)
            write_output.append({
                "input_text": text,
                "gold_argument": gold_arg_cls,
                "pred_argument": pred_arg_cls,
                "trigger": tri,
                "window_id": wid
            })
    elif config.task == 're_stage1':
        preds, input_text = model.predict(batch)
        for docid, wid, ts_offset, ents, all_relations, pred, text in zip(batch.doc_ids, batch.wnd_ids, batch.token_start_offsets, batch.entities, batch.all_relations, preds, input_text):
            gold_arg_id = [(wid,)+(r[0]+ts_offset, r[1]+ts_offset-1, r[2]+ts_offset, r[3]+ts_offset-1) for r in all_relations]
            gold_arg_cls = [(wid,)+(r[0]+ts_offset, r[1]+ts_offset-1, r[2]+ts_offset, r[3]+ts_offset-1, r[4]) for r in all_relations]
            test_gold_arg_id.extend(gold_arg_id)
            test_gold_arg_cls.extend(gold_arg_cls)

            pred_arg_id = [(wid, r[0]+ts_offset, r[1]+ts_offset-1, r[3]+ts_offset, r[4]+ts_offset-1) for r in pred]
            pred_arg_cls = [(wid, r[0]+ts_offset, r[1]+ts_offset-1, r[3]+ts_offset, r[4]+ts_offset-1, r[6]) for r in pred]
            test_pred_arg_id.extend(pred_arg_id)
            test_pred_arg_cls.extend(pred_arg_cls)
            write_output.append({
                "input_text": text,
                "gold_relations": gold_arg_cls,
                "pred_relations": pred_arg_cls,
                "given entities": ents,
                "window_id": wid, 
                "doc_id": docid,
            })
    elif config.task == 're':
        preds, input_text = model.predict(batch)
        for docid, wid, ts_offset, subj, rels, all_relations, pred, text in zip(batch.doc_ids, batch.wnd_ids, batch.token_start_offsets, batch.subjects, batch.relations, batch.all_relations, preds, input_text):
            true_gold_arg_id = [(wid,)+(r[0]+ts_offset, r[1]+ts_offset-1, r[2]+ts_offset, r[3]+ts_offset-1) for r in all_relations]
            true_gold_arg_cls = [(wid,)+(r[0]+ts_offset, r[1]+ts_offset-1, r[2]+ts_offset, r[3]+ts_offset-1, r[4]) for r in all_relations]
            test_gold_arg_id.extend(true_gold_arg_id)
            test_gold_arg_cls.extend(true_gold_arg_cls)

            gold_arg_id = [(wid, r[0][0]+ts_offset, r[0][1]+ts_offset-1, r[1][0]+ts_offset, r[1][1]+ts_offset-1) for r in rels]
            gold_arg_cls = [(wid, r[0][0]+ts_offset, r[0][1]+ts_offset-1, r[1][0]+ts_offset, r[1][1]+ts_offset-1, r[2]) for r in rels]
            # true_gold_arg_cls is different from gold_arg_cls because the if we use pred entity, then batch.relations cannot cover all relations.

            pred_arg_id = [(wid, subj[0]+ts_offset, subj[1]+ts_offset-1, r[0]+ts_offset, r[1]+ts_offset-1) for r in pred]
            pred_arg_cls = [(wid, subj[0]+ts_offset, subj[1]+ts_offset-1, r[0]+ts_offset, r[1]+ts_offset-1, r[2]) for r in pred]
            test_pred_arg_id.extend(pred_arg_id)
            test_pred_arg_cls.extend(pred_arg_cls)
            write_output.append({
                "input_text": text,
                "gold_relations (for the given entity)": gold_arg_cls,
                "pred_relations": pred_arg_cls,
                "subject_entity": subj,
                "window_id": wid, 
                "doc_id": docid,
            })
    elif config.task == 'ed':
        preds, input_text = model.predict(batch)
        pred_tris, entities = preds
        for wid, tri, pred, entity, text, token in zip(batch.wnd_ids, batch.triggers, pred_tris, entities, input_text, batch.tokens):
            gold_arg_id = [(wid, r[0], r[1]) for r in tri]
            gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in tri]
            pred_arg_id = [(wid, r[0], r[1]) for r in pred]
            pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]

            if config.use_ner:
                pred_entities = [(wid, r[0], r[1], r[2]) for r in entity]
                write_output.append({
                    "input_text": text,
                    "gold_trigger": gold_arg_cls,
                    "pred_trigger": pred_arg_cls,
                    "pred_entity": pred_entities,
                    "window_id": wid,
                    "tokens": token
                })
            else:
                write_output.append({
                    "input_text": text,
                    "gold_trigger": gold_arg_cls,
                    "pred_trigger": pred_arg_cls,
                    "window_id": wid,
                    "tokens": token
                })
            
            test_gold_arg_id.extend(gold_arg_id)
            test_gold_arg_cls.extend(gold_arg_cls)
            test_pred_arg_id.extend(pred_arg_id)
            test_pred_arg_cls.extend(pred_arg_cls)

progress.close()
test_gold_arg_id = set(test_gold_arg_id)
test_gold_arg_cls = set(test_gold_arg_cls)
test_pred_arg_id = set(test_pred_arg_id)
test_pred_arg_cls = set(test_pred_arg_cls)

gold_arg_id_num = len(test_gold_arg_id)
gold_arg_cls_num = len(test_gold_arg_cls)
match_arg_id_num = len(test_gold_arg_id & test_pred_arg_id)

pred_arg_id_num = len(test_pred_arg_id)
pred_arg_cls_num = len(test_pred_arg_cls)
match_arg_cls_num = len(test_gold_arg_cls & test_pred_arg_cls)

# calculate scores
test_scores = {
    'arg_id': compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
    'arg_cls': compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num)
}

# print scores
print("--Test score----------------------------------------------------------")
print('Identification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['arg_id'][0] * 100.0, match_arg_id_num, pred_arg_id_num, 
    test_scores['arg_id'][1] * 100.0, match_arg_id_num, gold_arg_id_num, test_scores['arg_id'][2] * 100.0))
print('Classification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['arg_cls'][0] * 100.0, match_arg_cls_num, pred_arg_cls_num, 
    test_scores['arg_cls'][1] * 100.0, match_arg_cls_num, gold_arg_cls_num, test_scores['arg_cls'][2] * 100.0))
print("---------------------------------------------------------------------")

if (config.task == 're_stage1' or config.task == 're') and (args.write_file_name is not None):
    pred_relations = dict()
    for obj in write_output:
        if obj['doc_id'] not in pred_relations.keys():
            pred_relations[obj['doc_id']] = dict()
        doc_container = pred_relations[obj['doc_id']]
        if obj['window_id'] not in doc_container.keys():
            doc_container[obj['window_id']] = list()
        doc_container[obj['window_id']].extend([[pred[1], pred[2], pred[3], pred[4], pred[5]] for pred in obj['pred_relations']])
    with open(args.test_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()

    writes = []
    for line in lines:
        document = json.loads(line)
        d_id = document['doc_key']
        output_relations = []
        for sent_id in range(len(document['sentences'])):
            if (d_id in pred_relations.keys()) and ('{}-w{}'.format(d_id, sent_id) in pred_relations[d_id].keys()):
                output_relations.append(pred_relations[d_id]['{}-w{}'.format(d_id, sent_id)])
            else:
                output_relations.append([])
        document['predicted_relations'] = output_relations
        writes.append(json.dumps(document))
    with open(os.path.join(model_folder,args.write_file_name), 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(writes))
    get_REscore(os.path.join(model_folder,args.write_file_name))

if (config.task == 'ed') and (args.write_file_name is not None):
    if config.use_ner:
        pred_triggers = dict()
        pred_ents = dict()
        for obj in write_output:
            if obj['window_id'] not in pred_triggers.keys():
                pred_triggers[obj['window_id']] = []
                pred_ents[obj['window_id']] = []
            for pred in obj['pred_trigger']:
                pred_triggers[obj['window_id']].append({
                    "event_type": pred[3],
                    "id": "{}-EV{}".format(obj['window_id'], len(pred_triggers[obj['window_id']])),
                    "trigger":{
                        "start": pred[1],
                        "end": pred[2],
                        "text": " ".join(obj['tokens'][pred[1]:pred[2]])
                    },
                    "arguments": []
                })
            for pred in obj['pred_entity']:
                pred_ents[obj['window_id']].append({
                    "id": "{}-E{}".format(obj['window_id'], len(pred_ents[obj['window_id']])),
                    "start": pred[1],
                    "end": pred[2],
                    "text": " ".join(obj['tokens'][pred[1]:pred[2]]),
                    "entity_type": pred[3],
                    "mention_type": "UNK",
                    "entity_subtype": "UNK"
                })
        
        with open(args.test_file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()

        writes = []
        for line in lines:
            sentence = json.loads(line)
            w_id = sentence['wnd_id']
            assert w_id in pred_triggers.keys()
            sentence['event_mentions'] = pred_triggers[w_id]
            sentence['entity_mentions'] = pred_ents[w_id]
            writes.append(json.dumps(sentence))
        with open(os.path.join(model_folder, args.write_file_name), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join(writes))
    else:
        pred_triggers = dict()
        for obj in write_output:
            if obj['window_id'] not in pred_triggers.keys():
                pred_triggers[obj['window_id']] = []
            for pred in obj['pred_trigger']:
                pred_triggers[obj['window_id']].append({
                    "event_type": pred[3],
                    "id": "{}-EV{}".format(obj['window_id'], len(pred_triggers[obj['window_id']])),
                    "trigger":{
                        "start": pred[1],
                        "end": pred[2],
                        "text": " ".join(obj['tokens'][pred[1]:pred[2]])
                    },
                    "arguments": []
                })
        
        with open(args.test_file, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()

        writes = []
        for line in lines:
            sentence = json.loads(line)
            w_id = sentence['wnd_id']
            assert w_id in pred_triggers.keys()
            sentence['event_mentions'] = pred_triggers[w_id]
            writes.append(json.dumps(sentence))
        with open(os.path.join(model_folder, args.write_file_name), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join(writes))