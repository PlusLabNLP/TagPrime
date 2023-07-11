import os, sys, json, logging, time, pprint, tqdm
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, AdamW,
                          XLMRobertaTokenizer,
                          RobertaTokenizer,
                          AlbertTokenizer,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup)
from model import StructuralModel
from data import EDDataset, EAEDataset, NERDataset, REDataset, REStage1Dataset, SPDataset
from scorer import *
import copy
import ipdb
from util import (Summarizer, generate_ed_vocabs, generate_eae_vocabs, generate_ner_vocabs, generate_re_vocabs, generate_sp_vocabs)
from pattern import *

# configuration
parser = ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('--seed', type=int, required=False)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
# config.update(args.__dict__)
config = Namespace(**config)

if args.seed is not None:
    config.seed = args.seed

# fix random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.backends.cudnn.enabled = False

# set GPU device
if config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

best_role_model = os.path.join(output_dir, 'best.role.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

model_name = config.bert_model_name
if model_name.startswith('bert-'):
    tokenizer = BertTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)
elif model_name.startswith('roberta-'):
    tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir,
                                            add_prefix_space=True)
elif model_name.startswith('xlm-'):
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir,
                                            add_prefix_space=True)
elif model_name.startswith('albert-'):
    tokenizer = AlbertTokenizer.from_pretrained(model_name,
                                            cache_dir=config.bert_cache_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir=config.bert_cache_dir,
                                          do_lower_case=False,
                                          use_fast=False)

special_tokens = []
if config.task == 'eae':
    if config.trigger_type:
        special_tokens += special_tags[config.dataset]["trigger:"+config.trigger_type]
    if config.role_type:
        special_tokens += special_tags[config.dataset]["role:"+config.role_type]
elif config.task == 'sp':
    if config.intent_type:
        special_tokens += special_tags[config.dataset]["intent:"+config.intent_type]
elif config.task == 're_stage1':
    if config.subject_type:
        special_tokens += special_tags[config.dataset]["subj:"+config.subject_type]
    if config.tag_type:
        special_tokens += special_tags[config.dataset]["tag:"+config.tag_type]
elif config.task == 're':
    if config.subject_type:
        special_tokens += special_tags[config.dataset]["subj:"+config.subject_type]
    if config.relation_type:
        special_tokens += special_tags[config.dataset]["rel:"+config.relation_type]
    if config.tag_type:
        special_tokens += special_tags[config.dataset]["tag:"+config.tag_type]

tokenizer.add_tokens(special_tokens)

# datasets
if config.task == 'eae':
    config.use_unified_label = getattr(config, "use_unified_label", True)
    print('==============Prepare Training Set=================')
    train_set = EAEDataset(config.train_file, tokenizer, max_length=config.max_length)
    print('==============Prepare Dev Set=================')
    dev_set = EAEDataset(config.dev_file, tokenizer, max_length=config.max_length)
    print('==============Prepare Test Set=================')
    test_set = EAEDataset(config.test_file, tokenizer, max_length=config.max_length)
    vocabs = generate_eae_vocabs([train_set, dev_set, test_set])
elif config.task == 're_stage1':
    print('==============Prepare Training Set=================')
    train_set = REStage1Dataset(config.train_file, tokenizer, max_length=config.max_length, use_pred_ner=False)
    print('==============Prepare Dev Set=================')
    dev_set = REStage1Dataset(config.dev_file, tokenizer, max_length=config.max_length, use_pred_ner=config.use_pred_ner)
    print('==============Prepare Test Set=================')
    test_set = REStage1Dataset(config.test_file, tokenizer, max_length=config.max_length, use_pred_ner=config.use_pred_ner)
    # test_set = REStage1Dataset(config.test_file, tokenizer, max_length=config.max_length, use_pred_ner=True)
    vocabs = generate_re_vocabs([train_set, dev_set, test_set])
elif config.task == 're':
    print('==============Prepare Training Set=================')
    train_set = REDataset(config.train_file, tokenizer, max_length=config.max_length, use_pred_ner=False)
    print('==============Prepare Dev Set=================')
    dev_set = REDataset(config.dev_file, tokenizer, max_length=config.max_length, use_pred_ner=config.use_pred_ner)
    print('==============Prepare Test Set=================')
    test_set = REDataset(config.test_file, tokenizer, max_length=config.max_length, use_pred_ner=config.use_pred_ner)
    # test_set = REDataset(config.test_file, tokenizer, max_length=config.max_length, use_pred_ner=True)
    vocabs = generate_re_vocabs([train_set, dev_set, test_set])    
elif config.task == 'ed':
    print('==============Prepare Training Set=================')
    train_set = EDDataset(config.train_file, tokenizer, max_length=config.max_length)
    print('==============Prepare Dev Set=================')
    dev_set = EDDataset(config.dev_file, tokenizer, max_length=config.max_length)
    print('==============Prepare Test Set=================')
    test_set = EDDataset(config.test_file, tokenizer, max_length=config.max_length)
    vocabs = generate_ed_vocabs([train_set, dev_set, test_set])
elif config.task == 'ner':
    print('==============Prepare Training Set=================')
    train_set = NERDataset(config.train_file, tokenizer, max_length=config.max_length)
    print('==============Prepare Dev Set=================')
    dev_set = NERDataset(config.dev_file, tokenizer, max_length=config.max_length)
    print('==============Prepare Test Set=================')
    test_set = NERDataset(config.test_file, tokenizer, max_length=config.max_length)
    vocabs = generate_ner_vocabs([train_set, dev_set, test_set])
elif config.task == 'sp':
    config.use_unified_label = getattr(config, "use_unified_label", True)
    print('==============Prepare Training Set=================')
    train_set = SPDataset(config.train_file, tokenizer, max_length=config.max_length, use_pred_intent=False)
    print('==============Prepare Dev Set=================')
    dev_set = SPDataset(config.dev_file, tokenizer, max_length=config.max_length, use_pred_intent=config.use_pred_intent)
    print('==============Prepare Test Set=================')
    test_set = SPDataset(config.test_file, tokenizer, max_length=config.max_length, use_pred_intent=True)
    vocabs = generate_sp_vocabs([train_set, dev_set, test_set])    


# save config
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)

batch_num = len(train_set) // config.batch_size + \
    (len(train_set) % config.batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + \
    (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)


# initialize the model
model = StructuralModel(config, tokenizer, vocabs, dataset=config.dataset)
model.cuda(device=config.gpu_device)
# print(model)

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if (n.startswith('model.bert.type_feature') or n.startswith('model.bert.role_feature'))],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if ((n.startswith('model.bert')) and not((n.startswith('model.bert.type_feature') or n.startswith('model.bert.role_feature'))))],
        'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if (not n.startswith('model.bert')) and (not n.startswith('model.binding_layer'))],
        'lr': config.learning_rate, 'weight_decay': config.weight_decay
    },
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('model.binding_layer')],
        'lr': 5e-04, 'weight_decay': 1e-05
    },
]

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*config.warmup_epoch,
                                           num_training_steps=batch_num*config.max_epoch)

# model state
state = dict(model=model.state_dict(),
             config=config,
             tokenizer=tokenizer,
             vocabs=vocabs)

# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'arg_id': (0.0, 0.0, 0.0),
    'arg_cls': (0.0, 0.0, -0.01)
}
print('================Start Training================')
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    
    # training step
    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    cummulate_loss = 0.
    for batch_idx, batch in enumerate(DataLoader(
            train_set, batch_size=config.batch_size // config.accumulate_step,
            shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):
        loss = model(batch)
        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1

        loss = loss * (1 / config.accumulate_step)
        cummulate_loss += loss
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
    progress.close()
    logger.info({"average training loss": (cummulate_loss / batch_idx).data})

    if (config.debug) or (epoch >= 10):
        best_dev_flag = False
        # dev set
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                            desc='Dev {}'.format(epoch))
        best_dev_role_model = False
        write_output = []
        dev_gold_arg_id, dev_gold_arg_cls, dev_pred_arg_id, dev_pred_arg_cls = [], [], [], []
        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)
            preds, input_text = model.predict(batch)
            if config.task == 'eae':
                for wid, tri, role, pred, text in zip(batch.wnd_ids, batch.triggers, batch.roles, preds, input_text):
                    gold_arg_id = [(wid, tri[2], r[1][0], r[1][1]) for r in role]
                    gold_arg_cls = [(wid, tri[2], r[1][0], r[1][1], r[1][2]) for r in role]
                    pred_arg_id = [(wid, tri[2], r[0], r[1]) for r in pred]
                    pred_arg_cls = [(wid, tri[2], r[0], r[1], r[2]) for r in pred]
                    
                    dev_gold_arg_id.extend(gold_arg_id)
                    dev_gold_arg_cls.extend(gold_arg_cls)
                    dev_pred_arg_id.extend(pred_arg_id)
                    dev_pred_arg_cls.extend(pred_arg_cls)
                    write_output.append({
                        "input_text": text,
                        "gold_argument": gold_arg_cls,
                        "pred_argument": pred_arg_cls,
                        "trigger": tri,
                        "window_id": wid
                    })
            elif config.task == 're_stage1':
                for wid, ents, all_relations, pred, text in zip(batch.wnd_ids, batch.entities, batch.all_relations, preds, input_text):
                    gold_arg_id = [(wid,)+r[:-1] for r in all_relations]
                    gold_arg_cls = [(wid,)+r for r in all_relations]
                    dev_gold_arg_id.extend(gold_arg_id)
                    dev_gold_arg_cls.extend(gold_arg_cls)

                    pred_arg_id = [(wid, r[0], r[1], r[3], r[4]) for r in pred]
                    pred_arg_cls = [(wid, r[0], r[1], r[3], r[4], r[6]) for r in pred]
                    dev_pred_arg_id.extend(pred_arg_id)
                    dev_pred_arg_cls.extend(pred_arg_cls)
                    write_output.append({
                        "input_text": text,
                        "gold_relations": gold_arg_cls,
                        "pred_relations": pred_arg_cls,
                        "given entities": ents,
                        "window_id": wid
                    })
            elif config.task == 're':
                for wid, subj, rels, all_relations, pred, text in zip(batch.wnd_ids, batch.subjects, batch.relations, batch.all_relations, preds, input_text):
                    true_gold_arg_id = [(wid,)+r[:-1] for r in all_relations]
                    true_gold_arg_cls = [(wid,)+r for r in all_relations]
                    dev_gold_arg_id.extend(true_gold_arg_id)
                    dev_gold_arg_cls.extend(true_gold_arg_cls)

                    gold_arg_id = [(wid, r[0][0], r[0][1], r[1][0], r[1][1]) for r in rels]
                    gold_arg_cls = [(wid, r[0][0], r[0][1], r[1][0], r[1][1], r[2]) for r in rels]
                    # true_gold_arg_cls is different from gold_arg_cls because the if we use pred entity, then batch.relations cannot cover all relations.

                    pred_arg_id = [(wid, subj[0], subj[1], r[0], r[1]) for r in pred]
                    pred_arg_cls = [(wid, subj[0], subj[1], r[0], r[1], r[2]) for r in pred]
                    dev_pred_arg_id.extend(pred_arg_id)
                    dev_pred_arg_cls.extend(pred_arg_cls)
                    write_output.append({
                        "input_text": text,
                        "gold_relations (for the given entity)": gold_arg_cls,
                        "pred_relations": pred_arg_cls,
                        "subject_entity": subj,
                        "window_id": wid
                    })
            elif config.task == 'ed':
                pred_tris, entities = preds
                for wid, tri, pred, entity, text in zip(batch.wnd_ids, batch.triggers, pred_tris, entities, input_text):
                    gold_arg_id = [(wid, r[0], r[1]) for r in tri]
                    gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in tri]
                    pred_arg_id = [(wid, r[0], r[1]) for r in pred]
                    pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]
                    
                    dev_gold_arg_id.extend(gold_arg_id)
                    dev_gold_arg_cls.extend(gold_arg_cls)
                    dev_pred_arg_id.extend(pred_arg_id)
                    dev_pred_arg_cls.extend(pred_arg_cls)
                    write_output.append({
                        "input_text": text,
                        "gold_trigger": gold_arg_cls,
                        "pred_trigger": pred_arg_cls,
                        "window_id": wid
                    })
            elif config.task == 'ner':
                for wid, tri, pred, text in zip(batch.wnd_ids, batch.entities, preds, input_text):
                    gold_arg_id = [(wid, r[0], r[1]) for r in tri]
                    gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in tri]
                    pred_arg_id = [(wid, r[0], r[1]) for r in pred]
                    pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]
                    
                    dev_gold_arg_id.extend(gold_arg_id)
                    dev_gold_arg_cls.extend(gold_arg_cls)
                    dev_pred_arg_id.extend(pred_arg_id)
                    dev_pred_arg_cls.extend(pred_arg_cls)
                    write_output.append({
                        "input_text": text,
                        "gold_entities": gold_arg_cls,
                        "pred_entities": pred_arg_cls,
                        "window_id": wid
                    })
            elif config.task == 'sp':
                for wid, itt, slot, pred, text in zip(batch.wnd_ids, batch.intents, batch.slots, preds, input_text):
                    gold_arg_id = [(wid, r[0], r[1]) for r in slot]
                    gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in slot]
                    pred_arg_id = [(wid, r[0], r[1]) for r in pred]
                    pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]
                    
                    dev_gold_arg_id.extend(gold_arg_id)
                    dev_gold_arg_cls.extend(gold_arg_cls)
                    dev_pred_arg_id.extend(pred_arg_id)
                    dev_pred_arg_cls.extend(pred_arg_cls)
                    write_output.append({
                        "input_text": text,
                        "gold_slot": gold_arg_cls,
                        "pred_slot": pred_arg_cls,
                        "intent": itt,
                        "window_id": wid
                    })

        progress.close()
        dev_gold_arg_id = set(dev_gold_arg_id)
        dev_gold_arg_cls = set(dev_gold_arg_cls)
        dev_pred_arg_id = set(dev_pred_arg_id)
        dev_pred_arg_cls = set(dev_pred_arg_cls)

        gold_arg_id_num = len(dev_gold_arg_id)
        gold_arg_cls_num = len(dev_gold_arg_cls)
        match_arg_id_num = len(dev_gold_arg_id & dev_pred_arg_id)

        pred_arg_id_num = len(dev_pred_arg_id)
        pred_arg_cls_num = len(dev_pred_arg_cls)
        match_arg_cls_num = len(dev_gold_arg_cls & dev_pred_arg_cls)
        
        # calculate scores
        dev_scores = {
            'arg_id': compute_f1(pred_arg_id_num, gold_arg_id_num, match_arg_id_num),
            'arg_cls': compute_f1(pred_arg_cls_num, gold_arg_cls_num, match_arg_cls_num)
        }

        # print scores
        print("--Dev score----------------------------------------------------------")
        print('Identification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
            dev_scores['arg_id'][0] * 100.0, match_arg_id_num, pred_arg_id_num, 
            dev_scores['arg_id'][1] * 100.0, match_arg_id_num, gold_arg_id_num, dev_scores['arg_id'][2] * 100.0))
        print('Classification     - P: {:5.2f} ({:4d}/{:4d}), R: {:5.2f} ({:4d}/{:4d}), F: {:5.2f}'.format(
            dev_scores['arg_cls'][0] * 100.0, match_arg_cls_num, pred_arg_cls_num, 
            dev_scores['arg_cls'][1] * 100.0, match_arg_cls_num, gold_arg_cls_num, dev_scores['arg_cls'][2] * 100.0))
        print("---------------------------------------------------------------------")
        
        # check best dev model
        if dev_scores['arg_cls'][2] > best_dev_scores['arg_cls'][2]:
            best_dev_flag = True

        if best_dev_flag:
            # save best model
            logger.info('Saving best model')
            torch.save(state, best_role_model)
            best_dev_scores = dev_scores
            best_dev_epoch = epoch
            
            # save dev result
            with open(dev_prediction_path, 'w') as fp:
                json.dump(write_output, fp, indent=4)

            # eval test set
            progress = tqdm.tqdm(total=test_batch_num, ncols=75, 
                                desc='Test {}'.format(epoch))
            write_output = []
            test_gold_arg_id, test_gold_arg_cls, test_pred_arg_id, test_pred_arg_cls = [], [], [], []
            for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn):
                progress.update(1)
                preds, input_text = model.predict(batch)
                if config.task == 'eae':
                    for wid, tri, role, pred, text in zip(batch.wnd_ids, batch.triggers, batch.roles, preds, input_text):
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
                    for wid, ents, all_relations, pred, text in zip(batch.wnd_ids, batch.entities, batch.all_relations, preds, input_text):
                        gold_arg_id = [(wid,)+r[:-1] for r in all_relations]
                        gold_arg_cls = [(wid,)+r for r in all_relations]
                        test_gold_arg_id.extend(gold_arg_id)
                        test_gold_arg_cls.extend(gold_arg_cls)

                        pred_arg_id = [(wid, r[0], r[1], r[3], r[4]) for r in pred]
                        pred_arg_cls = [(wid, r[0], r[1], r[3], r[4], r[6]) for r in pred]
                        test_pred_arg_id.extend(pred_arg_id)
                        test_pred_arg_cls.extend(pred_arg_cls)
                        write_output.append({
                            "input_text": text,
                            "gold_relations": gold_arg_cls,
                            "pred_relations": pred_arg_cls,
                            "given entities": ents,
                            "window_id": wid
                        })
                elif config.task == 're':
                    for wid, subj, rels, all_relations, pred, text in zip(batch.wnd_ids, batch.subjects, batch.relations, batch.all_relations, preds, input_text):
                        true_gold_arg_id = [(wid,)+r[:-1] for r in all_relations]
                        true_gold_arg_cls = [(wid,)+r for r in all_relations]
                        test_gold_arg_id.extend(true_gold_arg_id)
                        test_gold_arg_cls.extend(true_gold_arg_cls)

                        gold_arg_id = [(wid, r[0][0], r[0][1], r[1][0], r[1][1]) for r in rels]
                        gold_arg_cls = [(wid, r[0][0], r[0][1], r[1][0], r[1][1], r[2]) for r in rels]
                        # true_gold_arg_cls is different from gold_arg_cls because the if we use pred entity, then batch.relations cannot cover all relations.

                        pred_arg_id = [(wid, subj[0], subj[1], r[0], r[1]) for r in pred]
                        pred_arg_cls = [(wid, subj[0], subj[1], r[0], r[1], r[2]) for r in pred]
                        test_pred_arg_id.extend(pred_arg_id)
                        test_pred_arg_cls.extend(pred_arg_cls)
                        write_output.append({
                            "input_text": text,
                            "gold_relations (for the given entity)": gold_arg_cls,
                            "pred_relations": pred_arg_cls,
                            "subject_entity": subj,
                            "window_id": wid
                        })
                elif config.task == 'ed':
                    for wid, tri, pred, text in zip(batch.wnd_ids, batch.triggers, preds, input_text):
                        gold_arg_id = [(wid, r[0], r[1]) for r in tri]
                        gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in tri]
                        pred_arg_id = [(wid, r[0], r[1]) for r in pred]
                        pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]
                        
                        test_gold_arg_id.extend(gold_arg_id)
                        test_gold_arg_cls.extend(gold_arg_cls)
                        test_pred_arg_id.extend(pred_arg_id)
                        test_pred_arg_cls.extend(pred_arg_cls)
                        write_output.append({
                            "input_text": text,
                            "gold_trigger": gold_arg_cls,
                            "pred_trigger": pred_arg_cls,
                            "window_id": wid
                        })
                elif config.task == 'ner':
                    for wid, tri, pred, text in zip(batch.wnd_ids, batch.entities, preds, input_text):
                        gold_arg_id = [(wid, r[0], r[1]) for r in tri]
                        gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in tri]
                        pred_arg_id = [(wid, r[0], r[1]) for r in pred]
                        pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]
                        
                        test_gold_arg_id.extend(gold_arg_id)
                        test_gold_arg_cls.extend(gold_arg_cls)
                        test_pred_arg_id.extend(pred_arg_id)
                        test_pred_arg_cls.extend(pred_arg_cls)
                        write_output.append({
                            "input_text": text,
                            "gold_entities": gold_arg_cls,
                            "pred_entities": pred_arg_cls,
                            "window_id": wid
                        })
                elif config.task == 'sp':
                    for wid, itt, slot, pred, text in zip(batch.wnd_ids, batch.intents, batch.slots, preds, input_text):
                        gold_arg_id = [(wid, r[0], r[1]) for r in slot]
                        gold_arg_cls = [(wid, r[0], r[1], r[2]) for r in slot]
                        pred_arg_id = [(wid, r[0], r[1]) for r in pred]
                        pred_arg_cls = [(wid, r[0], r[1], r[2]) for r in pred]

                        test_gold_arg_id.extend(gold_arg_id)
                        test_gold_arg_cls.extend(gold_arg_cls)
                        test_pred_arg_id.extend(pred_arg_id)
                        test_pred_arg_cls.extend(pred_arg_cls)
                        write_output.append({
                            "input_text": text,
                            "gold_slot": gold_arg_cls,
                            "pred_slot": pred_arg_cls,
                            "intent": itt,
                            "window_id": wid
                        })

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
            
            # save test result
            with open(test_prediction_path, 'w') as fp:
                json.dump(write_output, fp, indent=4)
                
        logger.info({"epoch": epoch, "dev_scores": dev_scores})
        if best_dev_flag:
            logger.info({"epoch": epoch, "test_scores": test_scores})
        logger.info("Current best")
        logger.info({"best_epoch": best_dev_epoch, "best_scores": best_dev_scores})
    else:
        torch.save(model.state_dict(), best_model_path)
        
logger.info(log_path)
logger.info("Done!")
