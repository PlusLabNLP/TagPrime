import os, sys, json, logging, time, pprint, tqdm
import random
from argparse import ArgumentParser, Namespace
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, AdamW,
                          RobertaTokenizer,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup)
from model import StructuralModel
from data import EAEDataset, REStage1Dataset, REDataset, SPDataset
from scorer import *
import copy
import ipdb
from pattern import *
from pathlib import Path

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config', required=True )
parser.add_argument('-w', '--weight_path', required=True)
parser.add_argument('-t', '--test_file', type=str, required=True)
parser.add_argument('-p', '--pred_file', type=str, required=True)
parser.add_argument('--ignore_first_header', default=False, action='store_true')
parser.add_argument('--use_ner_filter', default=False, action='store_true')
parser.add_argument('--eval_batch', type=int)
parser.add_argument('--max_length', type=int)
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
    pred_set = EAEDataset(args.pred_file, tokenizer, max_length=config.max_length)

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
test_gold_tri_id, test_gold_tri_cls, test_pred_tri_id, test_pred_tri_cls = [], [], [], []
test_gold_arg_id, test_gold_arg_cls, test_pred_arg_id, test_pred_arg_cls = [], [], [], []

for batch in DataLoader(pred_set, batch_size=config.eval_batch_size,
                        shuffle=False, collate_fn=pred_set.collate_fn):
    progress.update(1)
    preds, input_text = model.predict(batch, use_ner_filter=args.use_ner_filter)
    if config.task == 'eae':
        for wid, tri, role, pred, text, token in zip(batch.wnd_ids, batch.triggers, batch.roles, preds, input_text, batch.tokens):
            pred_arg_id = [(wid, tri[2], r[0], r[1]) for r in pred]
            pred_arg_cls = [(wid, tri[2], r[0], r[1], r[2]) for r in pred]
            if (args.ignore_first_header) and (int(wid.split('-')[-1]) < 4):
                pass
            else:
                test_pred_arg_id.extend(pred_arg_id)
                test_pred_arg_cls.extend(pred_arg_cls)
                test_pred_tri_id.append((wid, tri[0], tri[1]))
                test_pred_tri_cls.append((wid, tri[0], tri[1], tri[2]))
#             write_output.append({
#                 "input_text": text,
#                 "gold_argument": gold_arg_cls,
#                 "pred_argument": pred_arg_cls,
#                 "trigger": tri,
#                 "window_id": wid
#             })

for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                        shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    if config.task == 'eae':
        for wid, tri, role, token in zip(batch.wnd_ids, batch.triggers, batch.roles, batch.tokens):
            gold_arg_id = [(wid, tri[2], r[1][0], r[1][1]) for r in role]
            gold_arg_cls = [(wid, tri[2], r[1][0], r[1][1], r[1][2]) for r in role]
            
            test_gold_arg_id.extend(gold_arg_id)
            test_gold_arg_cls.extend(gold_arg_cls)
            test_gold_tri_id.append((wid, tri[0], tri[1]))
            test_gold_tri_cls.append((wid, tri[0], tri[1], tri[2]))
            

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

test_gold_tri_id = set(test_gold_tri_id)
test_gold_tri_cls = set(test_gold_tri_cls)
test_pred_tri_id = set(test_pred_tri_id)
test_pred_tri_cls = set(test_pred_tri_cls)

gold_tri_id_num = len(test_gold_tri_id)
gold_tri_cls_num = len(test_gold_tri_cls)
match_tri_id_num = len(test_gold_tri_id & test_pred_tri_id)

pred_tri_id_num = len(test_pred_tri_id)
pred_tri_cls_num = len(test_pred_tri_cls)
match_tri_cls_num = len(test_gold_tri_cls & test_pred_tri_cls)

# calculate scores
test_scores = {
    'arg_id': compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
    'arg_cls': compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num), 
    'tri_id': compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
    'tri_cls': compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num), 
}

# print scores
print("--Test score----------------------------------------------------------")
print('Identification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['tri_id'][0] * 100.0, match_tri_id_num, pred_tri_id_num, 
    test_scores['tri_id'][1] * 100.0, match_tri_id_num, gold_tri_id_num, test_scores['tri_id'][2] * 100.0))
print('Classification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['tri_cls'][0] * 100.0, match_tri_cls_num, pred_tri_cls_num, 
    test_scores['tri_cls'][1] * 100.0, match_tri_cls_num, gold_tri_cls_num, test_scores['tri_cls'][2] * 100.0))
print("---------------------------------------------------------------------")
print('Identification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['arg_id'][0] * 100.0, match_arg_id_num, pred_arg_id_num, 
    test_scores['arg_id'][1] * 100.0, match_arg_id_num, gold_arg_id_num, test_scores['arg_id'][2] * 100.0))
print('Classification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
    test_scores['arg_cls'][0] * 100.0, match_arg_cls_num, pred_arg_cls_num, 
    test_scores['arg_cls'][1] * 100.0, match_arg_cls_num, gold_arg_cls_num, test_scores['arg_cls'][2] * 100.0))
print("---------------------------------------------------------------------")

