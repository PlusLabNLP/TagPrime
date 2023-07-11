import copy
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
from itertools import combinations 
import ipdb

ed_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'token_lens', 'token_num', 'trigger', 'entity', 'text']
EDInstance = namedtuple('EDInstance', field_names=ed_instance_fields, defaults=[None] * len(ed_instance_fields))

ed_batch_fields = ['doc_ids', 'wnd_ids', 'tokens', 'pieces', 'token_lens', 'token_nums','triggers', 'entities', 'texts']
EDBatch = namedtuple('EDBatch', field_names=ed_batch_fields, defaults=[None] * len(ed_batch_fields))

eae_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'token_lens', 'token_num', 'trigger', 'roles', 'text', 'entity']
EAEInstance = namedtuple('EAEInstance', field_names=eae_instance_fields, defaults=[None] * len(eae_instance_fields))

eae_batch_fields = ['doc_ids', 'wnd_ids', 'tokens', 'pieces', 'token_lens', 'token_nums','triggers', 'roles', 'texts', 'entities']
EAEBatch = namedtuple('EAEBatch', field_names=eae_batch_fields, defaults=[None] * len(eae_batch_fields))

ner_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'token_lens', 'token_num', 'entity', 'text']
NERInstance = namedtuple('NERInstance', field_names=ner_instance_fields, defaults=[None] * len(ner_instance_fields))

ner_batch_fields = ['doc_ids', 'wnd_ids', 'tokens', 'pieces', 'token_lens', 'token_nums','entities', 'texts']
NERBatch = namedtuple('NERBatch', field_names=ner_batch_fields, defaults=[None] * len(ner_batch_fields))

re_instance_fields = ['doc_id', 'wnd_id', 'token_start_offset', 'tokens', 'pieces', 'token_lens', 'use_token_masks', 'token_num', 'subject', 'relation', 'all_entities', 'all_relations', 'text', 'sub_tagged_pieces', 'sub_tagged_token_lens', 'sub_tagged_masks']
REInstance = namedtuple('REInstance', field_names=re_instance_fields, defaults=[None] * len(re_instance_fields))

re_batch_fields = ['doc_ids', 'wnd_ids', 'token_start_offsets', 'tokens', 'pieces', 'token_lens', 'use_token_masks', 'token_nums', 'subjects', 'relations', 'all_entities', 'all_relations', 'texts', 'sub_tagged_pieces', 'sub_tagged_token_lens', 'sub_tagged_masks', 'ner_tagged_pieces', 'ner_tagged_token_lens', 'ner_tagged_masks']
REBatch = namedtuple('REBatch', field_names=re_batch_fields, defaults=[None] * len(re_batch_fields))

re_stage1_instance_fields = ['doc_id', 'wnd_id', 'token_start_offset', 'tokens', 'pieces', 'token_lens', 'use_token_masks', 'ner_tagged_pieces', 'ner_tagged_token_lens', 'ner_tagged_masks', 'token_num', 'entities', 'all_relations', 'relation_map', 'text', 'ner_feat']
REStage1Instance = namedtuple('REStage1Instance', field_names=re_stage1_instance_fields, defaults=[None] * len(re_stage1_instance_fields))

re_stage1_batch_fields = ['doc_ids', 'wnd_ids', 'token_start_offsets', 'tokens', 'pieces', 'token_lens', 'use_token_masks', 'ner_tagged_pieces', 'ner_tagged_token_lens', 'ner_tagged_masks', 'token_nums', 'entities', 'all_relations', 'relation_maps', 'texts', 'ner_feats']
REStage1Batch = namedtuple('REStage1Batch', field_names=re_stage1_batch_fields, defaults=[None] * len(re_stage1_batch_fields))

sp_instance_fields = ['doc_id', 'wnd_id', 'tokens', 'pieces', 'token_lens', 'token_num', 'intent', 'g_intent', 'slots', 'text']
SPInstance = namedtuple('SPInstance', field_names=sp_instance_fields, defaults=[None] * len(sp_instance_fields))

sp_batch_fields = ['doc_ids', 'wnd_ids', 'tokens', 'pieces', 'token_lens', 'token_nums', 'intents', 'g_intents', 'slots', 'texts']
SPBatch = namedtuple('SPBatch', field_names=sp_batch_fields, defaults=[None] * len(sp_batch_fields))


def get_entity_labels(entities, token_num):
    """Convert entity mentions in a sentence to an entity label sequence with
    the length of token_num
    :param entities (list): a list of entity mentions.
    :param token_num (int): the number of tokens.
    :return:a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    count = 0
    for entity in entities:
        start, end = entity["start"], entity["end"]
        if end > token_num:
            continue
        entity_type = entity["entity_type"]
        if any([labels[i] != 'O' for i in range(start, end)]):
            count += 1
            continue
        labels[start] = 'B-{}'.format(entity_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(entity_type)
    if count:
        print('cannot cover {} entities due to span overlapping'.format(count))
    return labels

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        break_flag = False
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                break_flag = True
        if break_flag:
            continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    if len(id_map) > 0:
        print("{}/{} of entities are removed due to overlapping".format(len(id_map), len(entities)))
    return entities_, id_map

def get_role_list(entities, events, id_map):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(arg['entity_id'], arg['entity_id'])]
            
            # This will automatically remove multi role scenario
            if visited[i][entity_idx[0]] == 0:
                # ((trigger start, trigger end, trigger type), (argument start, argument end, role type))
                temp = ((event['trigger']['start'], event['trigger']['end'], event['event_type'], event['trigger']['text']),
                        (entity_idx[1]['start'], entity_idx[1]['end'], arg['role']))
                role_list.append(temp)
                visited[i][entity_idx[0]] = 1
    role_list.sort(key=lambda x: (x[0][0], x[1][0]))
    return role_list

class EAEDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, fair_compare=True):
        self.path = path
        self.tokenizer = tokenizer
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.fair_compare = fair_compare
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    # @property
    # def entity_type_set(self):
    #     type_set = set()
    #     for inst in self.insts:
    #         for entity in inst['entities']:
    #             type_set.add(entity["entity_type"])
    #     return type_set

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    @property
    def token_set(self):
        type_set = set()
        for inst in self.data:
            for token in inst['tokens']:
                type_set.add(token)
        return type_set

    @property
    def char_set(self):
        type_set = set()
        for inst in self.data:
            for token in inst['chars']:
                for char in token:
                    type_set.add(char)
        return type_set

    def load_data(self):
        """Load data from file."""
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                print("over max length with sub-token length {}".format(inst_len))
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']

            entities = inst['entity_mentions']
            if self.fair_compare:
                entities, entity_id_map = remove_overlap_entities(entities)
            else:
                entities = entities
                entity_id_map = {}
                
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            triggers = [(e['trigger']['start'], e['trigger']['end'], e['event_type'], e['trigger']['text']) for e in events]
            roles = get_role_list(entities, events, entity_id_map)
            
            for trigger in triggers:
                role = []
                for r in roles:
                    if r[0] == trigger:
                        role.append(r)

                instance = EAEInstance(
                    doc_id=doc_id,
                    wnd_id=wnd_id,
                    tokens=tokens,
                    pieces=pieces,
                    token_lens=token_lens,
                    token_num=token_num,
                    trigger=trigger,
                    roles=role,
                    text=" ".join(tokens),
                    entity=entities
                )
                self.data.append(instance)

        print('Loaded {} instances from {}'.format(len(self), self.path))

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        triggers = [inst.trigger for inst in batch]
        roles = [inst.roles for inst in batch]
        texts = [inst.text for inst in batch]
        token_nums = torch.cuda.LongTensor([inst.token_num for inst in batch])
        entities = [inst.entity for inst in batch]

        return EAEBatch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            tokens=tokens,
            pieces=pieces,
            token_lens=token_lens,
            token_nums=token_nums,
            triggers=triggers,
            roles=roles,
            texts=texts,
            entities=entities
        )

class REDataset(Dataset):
    ###
    # This is designed for data format that follows DyGIE format
    ###
    def __init__(self, path, tokenizer, max_length, use_pred_ner=False):
        self.path = path
        self.tokenizer = tokenizer
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.use_pred_ner = use_pred_ner
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.insts:
            for entity in inst['entity_mentions']:
                type_set.add(entity["entity_type"])
        return type_set

    @property
    def relation_type_set(self):
        type_set = set()
        for inst in self.insts:
            for arg in inst['relations']:
                type_set.add(arg[-1])
        return type_set

    def load_data(self):
        """Load data from file."""
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            document = json.loads(line)
            offset = 0
            all_tokens = [tok for sent in document['sentences'] for tok in sent]
            for sent_id in range(len(document['sentences'])):
                tokens = document['sentences'][sent_id]
                pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in tokens]
                token_lens = [len(x) for x in pieces]
                if 0 in token_lens:
                    raise ValueError
                # pieces = [p for ps in pieces for p in ps]

                if len([p for ps in pieces for p in ps]) > self.max_length:
                    print("over max length with sub-token length {}".format(len(pieces)))
                    offset += len(tokens)
                    continue

                if not self.use_pred_ner:
                    entities = [{"start":ne[0]-offset, 
                                 "end":ne[1]-offset+1, 
                                 "entity_type":ne[2], 
                                 "text":' '.join(all_tokens[ne[0]:(ne[1]+1)])} 
                        for ne in document['ner'][sent_id]]
                else:
                    entities = [{"start":ne[0]-offset, 
                                 "end":ne[1]-offset+1, 
                                 "entity_type":ne[2], 
                                 "text":' '.join(all_tokens[ne[0]:(ne[1]+1)])} 
                        for ne in document['predicted_ner'][sent_id]]

                ent_maps = {(ent["start"], ent["end"]): ent["entity_type"] for ent in entities}

                relations = [(re[0]-offset, re[1]-offset+1, re[2]-offset, re[3]-offset+1, re[4]) 
                        for re in document['relations'][sent_id]]

                rel_maps = {}
                for re in relations:
                    if (re[0], re[1]) not in rel_maps.keys():
                        rel_maps[(re[0], re[1])] = []
                    rel_maps[(re[0], re[1])].append(re)
                
                aggregate_rels = []
                for ner in entities:
                    key = (ner["start"], ner['end'])
                    if key in rel_maps.keys():
                        rels = rel_maps[key]
                        arguments = []
                        for rel in rels:
                            obj_start = rel[2]
                            obj_end = rel[3]
                            r = rel[4]
                            # if the argument is in entity list
                            if (obj_start, obj_end) in ent_maps.keys():
                                obj_type = ent_maps[(obj_start, obj_end)]
                                arguments.append((
                                    (ner["start"], ner["end"], ner['entity_type']),
                                    (obj_start, obj_end, obj_type),
                                    r
                                ))
                        arguments = sorted(arguments, key=lambda x: x[1][1]-x[1][0])
                    else:
                        arguments = []
                    aggregate_rels.append({
                        "subj_start": ner["start"],
                        "subj_end": ner['end'],
                        "subj_ent_type": ner['entity_type'],
                        "subj_text": ner['text'],
                        "arguments": arguments
                    })

                self.insts.append({
                    "doc_id": document['doc_key'],
                    "wnd_id": "{}-w{}".format(document['doc_key'], sent_id),
                    "token_start_offset": offset,
                    "tokens": tokens,
                    "pieces": pieces,
                    "token_lens": token_lens,
                    "entity_mentions": entities,
                    "relations": relations, # this is used to evaluate
                    "relation_mentions": aggregate_rels, # this is used to create our train/eval instance
                })
                offset += len(tokens)
        
        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            entities = inst['entity_mentions']
            
            for rel_seq in inst['relation_mentions']:
                subject = (rel_seq["subj_start"], rel_seq["subj_end"], rel_seq["subj_ent_type"], rel_seq["subj_text"])
                
                # tag all ner in the sequence
                sub_tagged_pieces = []
                sub_tagged_token_lens = []
                sub_tagged_masks = []
                for index, piece in enumerate(pieces):
                    # add start token if possible
                    if index == subject[0]:
                        added_token = self.tokenizer.tokenize('<S:{}>'.format(subject[2]))
                        sub_tagged_pieces.append(added_token)
                        sub_tagged_token_lens.append(len(added_token))
                        sub_tagged_masks.append(0)
                    sub_tagged_pieces.append(piece)
                    sub_tagged_token_lens.append(len(piece))
                    sub_tagged_masks.append(1)
                    # add end token if possible
                    if index == subject[1]-1:
                        added_token = self.tokenizer.tokenize('</S:{}>'.format(subject[2]))
                        sub_tagged_pieces.append(added_token)
                        sub_tagged_token_lens.append(len(added_token))
                        sub_tagged_masks.append(0)

                instance = REInstance(
                    doc_id=doc_id,
                    wnd_id=wnd_id,
                    token_start_offset=inst['token_start_offset'],
                    tokens=tokens,
                    pieces=[p for ps in pieces for p in ps],
                    token_lens=token_lens,
                    use_token_masks=[1]*token_num,
                    sub_tagged_pieces=[p for ps in sub_tagged_pieces for p in ps],
                    sub_tagged_token_lens=sub_tagged_token_lens,
                    sub_tagged_masks=sub_tagged_masks,
                    token_num=token_num,
                    subject=subject,
                    relation=rel_seq['arguments'],
                    all_entities=entities,
                    all_relations=inst['relations'],
                    text=" ".join(tokens),
                )
                self.data.append(instance)

        print('Loaded {} instances from {}'.format(len(self), self.path))

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        token_start_offsets = [inst.token_start_offset for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        use_token_masks = [inst.use_token_masks for inst in batch]
        subjects = [inst.subject for inst in batch]
        relations = [inst.relation for inst in batch]
        all_entities = [inst.all_entities for inst in batch]
        all_relations = [inst.all_relations for inst in batch]
        texts = [inst.text for inst in batch]
        token_nums = torch.cuda.LongTensor([inst.token_num for inst in batch])
        
        sub_tagged_pieces = [inst.sub_tagged_pieces for inst in batch]
        sub_tagged_token_lens = [inst.sub_tagged_token_lens for inst in batch]
        sub_tagged_masks = [inst.sub_tagged_masks for inst in batch]

        return REBatch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            token_start_offsets=token_start_offsets,
            tokens=tokens,
            pieces=pieces,
            token_lens=token_lens,
            use_token_masks=use_token_masks,
            token_nums=token_nums,
            subjects=subjects,
            relations=relations,
            all_entities=all_entities,
            all_relations=all_relations,
            texts=texts,
            sub_tagged_pieces=sub_tagged_pieces,
            sub_tagged_token_lens=sub_tagged_token_lens,
            sub_tagged_masks=sub_tagged_masks,
        )

class REStage1Dataset(Dataset):
    ###
    # This is designed for data format that follows DyGIE format
    ###
    def __init__(self, path, tokenizer, max_length, use_pred_ner=False):
        self.path = path
        self.tokenizer = tokenizer
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.use_pred_ner = use_pred_ner
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.insts:
            for entity in inst['entity_mentions']:
                type_set.add(entity["entity_type"])
        return type_set

    @property
    def relation_type_set(self):
        type_set = set()
        for inst in self.insts:
            for arg in inst['relations']:
                type_set.add(arg[-1])
        return type_set

    @property
    def valid_pair_set(self):
        subj_rel_counter = Counter()
        for inst in self.insts:
            ent_map = inst['entity_maps']
            for rel in inst['relations']:
                subj = (rel[0], rel[1])
                obj = (rel[2], rel[3])
                retype = rel[4]
                subjtype = ent_map[subj]
                objtype = ent_map[obj]
                subj_rel_counter[(subjtype, objtype, retype)] += 1
        return subj_rel_counter

    def load_data(self):
        """Load data from file."""
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            document = json.loads(line)
            offset = 0
            all_tokens = [tok for sent in document['sentences'] for tok in sent]
            for sent_id in range(len(document['sentences'])):
                tokens = document['sentences'][sent_id]
                pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in tokens]
                token_lens = [len(x) for x in pieces]
                if 0 in token_lens:
                    raise ValueError
                # pieces = [p for ps in pieces for p in ps]

                if len([p for ps in pieces for p in ps]) > self.max_length:
                    print("over max length with sub-token length {}".format(len(pieces)))
                    continue

                if not self.use_pred_ner:
                    entities = [{"start":ne[0]-offset, 
                                 "end":ne[1]-offset+1, 
                                 "entity_type":ne[2], 
                                 "text":' '.join(all_tokens[ne[0]:(ne[1]+1)])} 
                        for ne in document['ner'][sent_id]]
                else:
                    entities = [{"start":ne[0]-offset, 
                                 "end":ne[1]-offset+1, 
                                 "entity_type":ne[2], 
                                 "text":' '.join(all_tokens[ne[0]:(ne[1]+1)])} 
                        for ne in document['predicted_ner'][sent_id]]
                

                ent_maps = {(ent["start"], ent["end"]): ent["entity_type"] for ent in entities}

                relations = [(re[0]-offset, re[1]-offset+1, re[2]-offset, re[3]-offset+1, re[4]) 
                        for re in document['relations'][sent_id]]
                
                rel_maps = {}
                for re in relations:
                    if (re[0], re[1]) not in rel_maps.keys():
                        rel_maps[(re[0], re[1])] = []
                    rel_maps[(re[0], re[1])].append(re)

                # to make the training more stable, if we are not doing final testing, we will remove sentences without entities:
                if self.use_pred_ner or len(entities)>0:
                    self.insts.append({
                        "doc_id": document['doc_key'],
                        "wnd_id": "{}-w{}".format(document['doc_key'], sent_id),
                        "token_start_offset": offset,
                        "tokens": tokens,
                        "pieces": pieces,
                        "token_lens": token_lens,
                        "entity_mentions": entities,
                        "entity_maps": ent_maps,
                        "relations": relations, # this is used to evaluate
                        "relation_map": rel_maps, # this is for fast creation of training data
                    })
                offset += len(tokens)
        
        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            entities = inst['entity_mentions']
            
            # tag all ner in the sequence
            ner_tagged_pieces = []
            ner_tagged_token_lens = []
            ner_tagged_masks = []
            ner_feature = []
            for index, piece in enumerate(pieces):
                # add start token if possible
                flag = False
                for ent in entities:
                    if index == ent['start']:
                        added_token = self.tokenizer.tokenize('<ent:{}>'.format(ent['entity_type']))
                        ner_tagged_pieces.append(added_token)
                        ner_tagged_token_lens.append(len(added_token))
                        ner_tagged_masks.append(0)
                    if ent['start'] <= index < ent['end']:
                        flag = True
                ner_tagged_pieces.append(piece)
                ner_tagged_token_lens.append(len(piece))
                ner_tagged_masks.append(1)
                if flag:
                    ner_feature.append(2) # is NER
                else:
                    ner_feature.append(1) # not NER
                # add end token if possible
                for ent in reversed(entities):
                    if index == ent['end']-1:
                        added_token = self.tokenizer.tokenize('</ent:{}>'.format(ent['entity_type']))
                        ner_tagged_pieces.append(added_token)
                        ner_tagged_token_lens.append(len(added_token))
                        ner_tagged_masks.append(0)

            instance = REStage1Instance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                token_start_offset=inst['token_start_offset'],
                tokens=tokens,
                pieces=[p for ps in pieces for p in ps],
                token_lens=token_lens,
                use_token_masks=[1]*token_num,
                ner_tagged_pieces=[p for ps in ner_tagged_pieces for p in ps],
                ner_tagged_token_lens=ner_tagged_token_lens,
                ner_tagged_masks=ner_tagged_masks,
                token_num=token_num,
                entities=entities,
                all_relations=inst['relations'],
                relation_map=inst['relation_map'],
                text=" ".join(tokens),
                ner_feat=ner_feature,
            )
            self.data.append(instance)

        print('Loaded {} instances from {}'.format(len(self), self.path))

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        token_start_offsets = [inst.token_start_offset for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        use_token_masks = [inst.use_token_masks for inst in batch]
        ner_tagged_pieces = [inst.ner_tagged_pieces for inst in batch]
        ner_tagged_token_lens = [inst.ner_tagged_token_lens for inst in batch]
        ner_tagged_masks = [inst.ner_tagged_masks for inst in batch]
        
        entities = [inst.entities for inst in batch]
        all_relations = [inst.all_relations for inst in batch]
        relation_maps = [inst.relation_map for inst in batch]
        texts = [inst.text for inst in batch]
        token_nums = torch.cuda.LongTensor([inst.token_num for inst in batch])
        ner_feats = [inst.ner_feat for inst in batch]
        
        return REStage1Batch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            token_start_offsets=token_start_offsets,
            tokens=tokens,
            pieces=pieces,
            token_lens=token_lens,
            use_token_masks=use_token_masks,
            ner_tagged_pieces=ner_tagged_pieces,
            ner_tagged_token_lens=ner_tagged_token_lens,
            ner_tagged_masks=ner_tagged_masks,
            token_nums=token_nums,
            entities=entities,
            all_relations=all_relations,
            relation_maps=relation_maps,
            texts=texts,
            ner_feats=ner_feats,
        )

class EDDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.path = path
        self.tokenizer = tokenizer
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['entity_mentions']:
                type_set.add(event['entity_type'])
        return type_set

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                print("over max length with sub-token length {}".format(inst_len))
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            
            entities = inst['entity_mentions']
            entities.sort(key=lambda x: x['end']-x['start'], reverse=True)
            
            events = inst['event_mentions']
            events.sort(key=lambda x: x['trigger']['start'])
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            triggers = [(e['trigger']['start'], e['trigger']['end'], e['event_type'], e['trigger']['text']) for e in events]
            
            ents = [(e['start'], e['end'], e['entity_type'], e['text']) for e in entities]
            
            instance = EDInstance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                token_lens=token_lens,
                token_num=token_num,
                trigger=triggers,
                entity=ents,
                text=" ".join(tokens)
            )
            self.data.append(instance)

        print('Loaded {} instances from {}'.format(len(self), self.path))

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        triggers = [inst.trigger for inst in batch]
        entities = [inst.entity  for inst in batch]
        texts = [inst.text for inst in batch]
        token_nums = torch.cuda.LongTensor([inst.token_num for inst in batch])

        return EDBatch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            tokens=tokens,
            pieces=pieces,
            token_lens=token_lens,
            token_nums=token_nums,
            triggers=triggers,
            entities=entities,
            texts=texts
        )

class NERDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, remove_overlap=True):
        self.path = path
        self.tokenizer = tokenizer
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.remove_overlap = remove_overlap
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.insts:
            for event in inst['entity_mentions']:
                type_set.add(event['entity_type'])
        return type_set

    def load_data(self):
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            if inst_len > self.max_length:
                print("over max length with sub-token length {}".format(inst_len))
                continue
            self.insts.append(inst)

        for inst in self.insts:
            doc_id = inst['doc_id']
            wnd_id = inst['wnd_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
              
            entities = inst['entity_mentions']
            entities.sort(key=lambda x: x['end']-x['start'], reverse=True)

            if self.remove_overlap:
                entities, entity_id_map = remove_overlap_entities(entities)
            
            token_num = len(tokens)
            token_lens = inst['token_lens']
            
            ents = [(e['start'], e['end'], e['entity_type'], e['text']) for e in entities]
            
            instance = NERInstance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                token_lens=token_lens,
                token_num=token_num,
                entity=ents,
                text=" ".join(tokens)
            )
            self.data.append(instance)

        print('Loaded {} instances from {}'.format(len(self), self.path))

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        entities = [inst.entity for inst in batch]
        texts = [inst.text for inst in batch]
        token_nums = torch.cuda.LongTensor([inst.token_num for inst in batch])

        return NERBatch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            tokens=tokens,
            pieces=pieces,
            token_lens=token_lens,
            token_nums=token_nums,
            entities=entities,
            texts=texts
        )
    
class SPDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, use_pred_intent=False):
        self.path = path
        self.tokenizer = tokenizer
        self.data = []
        self.insts = []
        self.max_length = max_length
        self.use_pred_intent = use_pred_intent
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def intent_type_set(self):
        type_set = set()
        for inst in self.insts:
            type_set.add(inst["intent"])
        return type_set

    @property
    def slot_type_set(self):
        type_set = set()
        for inst in self.insts:
            for slot in inst['slots']:
                type_set.add(slot[-1])
        return type_set

    def load_data(self):
        """Load data from file."""
        with open(self.path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        self.insts = []
        for line in lines:
            inst = json.loads(line)
            self.insts.append(inst)
            
        for inst in self.insts:
            
            tokens = inst['tokens']
            token_num = len(tokens)
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in tokens]
            pieces = [x if len(x)>0 else [self.tokenizer.unk_token] for x in pieces] # handle unkown words
            token_lens = [len(x) for x in pieces]
            pieces = [p for ps in pieces for p in ps]
            
            if len(pieces) > self.max_length:
                print("over max length with sub-token length {}".format(len(pieces)))
                continue
            
            instance = SPInstance(
                doc_id=inst['doc_id'],
                wnd_id=inst['wnd_id'],
                tokens=inst['tokens'],
                pieces=pieces,
                token_lens=token_lens,
                token_num=token_num,
                intent=inst['p_intent'] if self.use_pred_intent else inst['intent'],
                g_intent=inst['intent'],
                slots=inst['slots'],
                text=inst['sentence'],
            )
            self.data.append(instance)

        print('Loaded {} instances from {}'.format(len(self), self.path))

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        pieces = [inst.pieces for inst in batch]
        token_lens = [inst.token_lens for inst in batch]
        intents = [inst.intent for inst in batch]
        g_intents = [inst.g_intent for inst in batch]
        slots = [inst.slots for inst in batch]
        texts = [inst.text for inst in batch]
        token_nums = torch.cuda.LongTensor([inst.token_num for inst in batch])

        return SPBatch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            tokens=tokens,
            pieces=pieces,
            token_lens=token_lens,
            token_nums=token_nums,
            intents=intents,
            g_intents=g_intents,
            slots=slots,
            texts=texts
        )
    