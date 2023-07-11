import ipdb
from argparse import ArgumentParser
from collections import defaultdict
import json
from pprint import pprint

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def convert_arguments_id(entity_maps, events):
    args = set()
    for event in events:
        for r in event['arguments']:
            trigger_label = event['event_type']
            argu = entity_maps[r['entity_id']]
            arg_start = argu[0]
            arg_end = argu[1]
            args.add((arg_start, arg_end, trigger_label))
    return args

def convert_arguments_cls(entity_maps, events):
    args = set()
    for event in events:
        for r in event['arguments']:
            trigger_label = event['event_type']
            argu = entity_maps[r['entity_id']]
            arg_start = argu[0]
            arg_end = argu[1]
            role = r['role']
            args.add((arg_start, arg_end, trigger_label, role))
    return args

def score_ee_graphs(gold_graphs, pred_graphs):
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_trigger_id_num = pred_trigger_id_num = trigger_idn_num = 0
    gold_trigger_cls_num = pred_trigger_cls_num = trigger_cls_num = 0
    gold_eve_type_num = pred_eve_type_num = eve_type_match_num = 0

    gold_arg_id_num = pred_arg_id_num = arg_idn_num = 0
    gold_arg_cls_num = pred_arg_cls_num =  arg_cls_num = 0

    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):

        gold_entity_maps = dict()
        pred_entity_maps = dict()

        # Entity
        gold_entities = []
        for e in gold_graph['entity_mentions']:
            gold_entities.append((e['start'], e['end'], e['entity_type']))
            gold_entity_maps[e['id']] = (e['start'], e['end'], e['entity_type'])

        gold_entities = list(set(gold_entities))
        gold_entities = [[e[0], e[1], e[2]] for e in gold_entities]

        pred_entities = []
        for e in pred_graph['entity_mentions']:
            pred_entities.append((e['start'], e['end'], e['entity_type']))
            pred_entity_maps[e['id']] = (e['start'], e['end'], e['entity_type'])

        pred_entities = list(set(pred_entities))
        pred_entities = [[e[0], e[1], e[2]] for e in pred_entities]

        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])

        # Trigger ID
        gold_trigger_ids = [(t['trigger']['start'], t['trigger']['end']) for t in gold_graph['event_mentions']]
        gold_trigger_ids = list(set(gold_trigger_ids))
        gold_trigger_ids = [[t[0], t[1]] for t in gold_trigger_ids]

        pred_trigger_ids = [(t['trigger']['start'], t['trigger']['end']) for t in pred_graph['event_mentions']]
        pred_trigger_ids = list(set(pred_trigger_ids))
        pred_trigger_ids = [[t[0], t[1]] for t in pred_trigger_ids]

        gold_trigger_id_num += len(gold_trigger_ids)
        pred_trigger_id_num += len(pred_trigger_ids)
        for pred_trig in pred_trigger_ids:
            matched = [item for item in gold_trigger_ids
                       if item[0] == pred_trig[0] and item[1] == pred_trig[1]]
            if matched:
                trigger_idn_num += 1

        # Trigger CLS
        gold_trigger_cls = [(t['trigger']['start'], t['trigger']['end'], t['event_type']) for t in gold_graph['event_mentions']]
        gold_trigger_cls = list(set(gold_trigger_cls))
        gold_trigger_cls = [[t[0], t[1], t[2]] for t in gold_trigger_cls]
        pred_trigger_cls = [(t['trigger']['start'], t['trigger']['end'], t['event_type']) for t in pred_graph['event_mentions']]
        pred_trigger_cls = list(set(pred_trigger_cls))
        pred_trigger_cls = [[t[0], t[1], t[2]] for t in pred_trigger_cls]

        gold_trigger_cls_num += len(gold_trigger_cls)
        pred_trigger_cls_num += len(pred_trigger_cls)
        for pred_trig in pred_trigger_cls:
            matched = [item for item in gold_trigger_cls
                       if item[0] == pred_trig[0] and item[1] == pred_trig[1] and item[2] == pred_trig[2]]
            if matched:
                trigger_cls_num += 1

        # Event Type
        gold_eve_type = list(set([t['event_type'] for t in gold_graph['event_mentions']]))
        pred_eve_type = list(set([t['event_type'] for t in pred_graph['event_mentions']]))
        gold_eve_type_num += len(gold_eve_type)
        pred_eve_type_num += len(pred_eve_type)
        for pred_eve in pred_eve_type:
            matched = [item for item in gold_eve_type if item == pred_eve]
            if matched:
                eve_type_match_num += 1


        # Argument ID
        gold_args = convert_arguments_id(gold_entity_maps, gold_graph['event_mentions'])
        pred_args = convert_arguments_id(pred_entity_maps, pred_graph['event_mentions'])
        gold_arg_id_num += len(gold_args)
        pred_arg_id_num += len(pred_args)
        for pred_arg in pred_args:
            gold_idn = [item for item in gold_args
                        if item==pred_arg]
            if gold_idn:
                arg_idn_num += 1
        
        # Argument CLS
        gold_args = convert_arguments_cls(gold_entity_maps, gold_graph['event_mentions'])
        pred_args = convert_arguments_cls(pred_entity_maps, pred_graph['event_mentions'])
        gold_arg_cls_num += len(gold_args)
        pred_arg_cls_num += len(pred_args)
        for pred_arg in pred_args:
            gold_cls = [item for item in gold_args
                        if item==pred_arg]
            if gold_cls:
                arg_cls_num += 1

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)
    trigger_id_prec, trigger_id_rec, trigger_id_f = compute_f1(
        pred_trigger_id_num, gold_trigger_id_num, trigger_idn_num)
    trigger_prec, trigger_rec, trigger_f = compute_f1(
        pred_trigger_cls_num, gold_trigger_cls_num, trigger_cls_num)
    
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_id_num, gold_arg_id_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_cls_num, gold_arg_cls_num, arg_cls_num)
    
    eve_type_prec, eve_type_rec, eve_type_f = compute_f1(
        pred_eve_type_num, gold_eve_type_num, eve_type_match_num)
    
    print("---------------------------------------------------------------------")
    print('Entity     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        entity_prec * 100.0, ent_match_num, pred_ent_num, 
        entity_rec * 100.0, ent_match_num, gold_ent_num, entity_f * 100.0))
    
    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        trigger_id_prec * 100.0, trigger_idn_num, pred_trigger_id_num, 
        trigger_id_rec * 100.0, trigger_idn_num, gold_trigger_id_num, trigger_id_f * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        trigger_prec * 100.0, trigger_cls_num, pred_trigger_cls_num,
        trigger_rec * 100.0, trigger_cls_num, gold_trigger_cls_num, trigger_f * 100.0))
    
    print("---------------------------------------------------------------------")
    print('Event Type - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        eve_type_prec * 100.0, eve_type_match_num, pred_eve_type_num, 
        eve_type_rec * 100.0, eve_type_match_num, gold_eve_type_num, eve_type_f * 100.0))
    
    print("---------------------------------------------------------------------")
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        role_id_prec * 100.0, arg_idn_num, pred_arg_id_num, 
        role_id_rec * 100.0, arg_idn_num, gold_arg_id_num, role_id_f * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        role_prec * 100.0, arg_cls_num, pred_arg_cls_num, 
        role_rec * 100.0, arg_cls_num, gold_arg_cls_num, role_f * 100.0))
    
    print("---------------------------------------------------------------------")
    
    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f},
        'trigger': {'prec': trigger_prec, 'rec': trigger_rec, 'f': trigger_f},
        'trigger_id': {'prec': trigger_id_prec, 'rec': trigger_id_rec,
                       'f': trigger_id_f},
        'role': {'prec': role_prec, 'rec': role_rec, 'f': role_f},
        'role_id': {'prec': role_id_prec, 'rec': role_id_rec, 'f': role_id_f},
        'event_type': {'prec': eve_type_prec, 'rec': eve_type_rec,
                       'f': eve_type_f}
    }

    return scores

# configuration
parser = ArgumentParser()
parser.add_argument('-p', '--pred_path', required=True)
parser.add_argument('-g', '--gold_path', required=True)
args = parser.parse_args()

predictions = [json.loads(line) for line in open(args.pred_path, 'r', encoding='utf-8').readlines()]
golds = [json.loads(line) for line in open(args.gold_path, 'r', encoding='utf-8').readlines()]

predictions = {pred['wnd_id']: pred for pred in predictions}
golds = {gold['wnd_id']: gold for gold in golds}

assert len(golds) == len(predictions)

pred_graphs = []
gold_graphs = []
for key, pred_graph in predictions.items():
    gold_graph = golds[key]
    # ipdb.set_trace()
    pred_graphs.append(pred_graph)
    gold_graphs.append(gold_graph)


full_scores = score_ee_graphs(gold_graphs, pred_graphs)