import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, RobertaModel, AutoModel, BertConfig, RobertaConfig, AlbertModel, AlbertConfig
import copy
import ipdb

from pattern import *
from model_utils import (token_lens_to_offsets_masked, token_lens_to_idxs_masked, tag_paths_to_spans, get_relation_seqlabels_stage1, get_relation_seqlabels, get_entity_embedding, CRF, Linears)

from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class REStage1Model(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.entity_type_stoi = self.vocabs['entity_type']
        self.entity_type_num = len(self.entity_type_stoi)

        self.relation_label_stoi = vocabs['relation_label']
        self.relation_type_stoi = vocabs['relation_type']
        self.relation_label_itos = {i:s for s, i in self.relation_label_stoi.items()}
        self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        
        self.relation_label_num = len(self.relation_label_stoi)
        self.relation_type_num = len(self.relation_type_stoi)

        # BERT encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
            self.bos_token = "[CLS]"
            self.eos_token = "[SEP]"
            self.sep_token = self.tokenizer.sep_token
        elif self.bert_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = self.tokenizer.sep_token
        else:
            raise ValueError

        if self.config.model_type.startswith("early"):
            # binding Transformer
            self.bert_binding = copy.deepcopy(self.bert.encoder.layer[self.config.cut_layer:])

            # cut layer
            assert self.config.cut_layer < len(self.bert.encoder.layer)
            self.bert.encoder.layer = self.bert.encoder.layer[:self.config.cut_layer]

            # BERT label encoder
            self.bert_label = self.bert
        
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        if self.config.use_span_feature:
            feature_dim = self.bert_dim*2
        else:
            feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.entity_type_num, 100)

        if self.config.use_ner_feature:
            feature_dim += 10
            self.ner_feature_module = nn.Embedding(3, 10)

        self.relation_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.relation_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.relation_crf = CRF(self.relation_label_stoi, bioes=False)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        
        ## Info in this part is used for self.bert
        enc_idxs = []
        enc_attn = []

        # Note: if we do not us binding, then the below three list will not be use
        ## Info in this part is used for self.bert_label
        prompt_label_idxs = []
        prompt_label_attn = []
        ## We need this to perform duplication for the output of self.bert
        offsets = []
        
        relation_seqidxs = []
        subject_types = []
        ner_feats = []
        token_lens = []
        token_masks = []
        token_nums = []
        subjects = []
        max_token_num = max([len(tokens) for tokens, entities in zip(batch.tokens, batch.entities) if len(entities)>0])
        
        for tokens, piece, token_mask, token_len, ner_tag_piece, ner_token_len, ner_token_mask, entities, relations, relation_map, token_num, ner_feat in zip(batch.tokens, batch.pieces, batch.use_token_masks,  batch.token_lens, batch.ner_tagged_pieces, batch.ner_tagged_token_lens, batch.ner_tagged_masks, batch.entities, batch.all_relations, batch.relation_maps, batch.token_nums, batch.ner_feats):
            if self.config.model_type.startswith("early"): 
                if len(entities)>0:
                    if self.config.model_type == "early+passage+Subjprompt+Typeprompt":
                        use_piece = piece
                        use_token_len = token_len
                        use_token_mask = token_mask
                    
                    elif self.config.model_type == "early+nertag+Subjprompt+Typeprompt":
                        use_piece = ner_tag_piece
                        use_token_len = ner_token_len
                        use_token_mask = ner_token_mask

                    else:
                        raise ValueError

                    piece_id = self.tokenizer.convert_tokens_to_ids(use_piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.bos_token)] + \
                                piece_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.eos_token)]
                    
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    
                ner_map = ner_tags[self.dataset][self.config.subject_type]
                # enumerate all entity
                for subject in entities:
                    prompt_label = "{} {} {}".format(subject['text'], self.sep_token, ner_map[subject['entity_type']])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                    rel_seq = get_relation_seqlabels_stage1(subject, relation_map, len(tokens))
                    if self.config.use_crf:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    subject_types.append(self.entity_type_stoi[subject['entity_type']])
                    token_lens.append(use_token_len)
                    token_masks.append(use_token_mask)
                    token_nums.append(token_num)
                    subjects.append((subject['start'], subject['end'], subject['entity_type'], subject['text']))
                    ner_feats.append(ner_feat + [0]*(max_token_num-len(ner_feat)))
                    offsets.append(len(enc_idxs)-1)
                
            elif self.config.model_type.startswith("sep"):
                # enumerate all entity
                ner_map = ner_tags[self.dataset][self.config.subject_type]
                for subject in entities:
                    if self.config.model_type == "sep+passage+Subjprompt+Typeprompt":
                        use_token_len = token_len
                        use_token_mask = token_mask
                        prompt = "{} {} {} {}".format(self.sep_token, subject['text'], 
                                                self.sep_token, ner_map[subject['entity_type']])
                        piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                        prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False, is_split_into_words=False)
                        enc_idx = [self.tokenizer.convert_tokens_to_ids(self.bos_token)] + \
                                    piece_id + \
                                    prompt_id + \
                                    [self.tokenizer.convert_tokens_to_ids(self.eos_token)]
                   
                    elif self.config.model_type == "sep+nertag+Subjprompt+Typeprompt":
                        use_token_len = ner_token_len
                        use_token_mask = ner_token_mask
                        prompt = "{} {} {} {}".format(self.sep_token, subject['text'], 
                                                self.sep_token, ner_map[subject['entity_type']])
                        piece_id = self.tokenizer.convert_tokens_to_ids(ner_tag_piece)
                        prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False, is_split_into_words=False)
                        enc_idx = [self.tokenizer.convert_tokens_to_ids(self.bos_token)] + \
                                    piece_id + \
                                    prompt_id +\
                                    [self.tokenizer.convert_tokens_to_ids(self.eos_token)]

                    elif self.config.model_type == "sep+passage":
                        use_token_len = token_len
                        use_token_mask = token_mask
                        piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                        enc_idx = [self.tokenizer.convert_tokens_to_ids(self.bos_token)] + \
                                    piece_id + [self.tokenizer.convert_tokens_to_ids(self.eos_token)]
                   
                    elif self.config.model_type == "sep+nertag":
                        use_token_len = ner_token_len
                        use_token_mask = ner_token_mask
                        piece_id = self.tokenizer.convert_tokens_to_ids(ner_tag_piece)
                        enc_idx = [self.tokenizer.convert_tokens_to_ids(self.bos_token)] + \
                                    piece_id + [self.tokenizer.convert_tokens_to_ids(self.eos_token)]

                    else:
                        raise ValueError
                    
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    rel_seq = get_relation_seqlabels_stage1(subject, relation_map, len(tokens))
                    if self.config.use_crf:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    subject_types.append(self.entity_type_stoi[subject['entity_type']])
                    token_lens.append(use_token_len)
                    token_masks.append(use_token_mask)
                    token_nums.append(token_num)
                    ner_feats.append(ner_feat + [0]*(max_token_num-len(ner_feat)))
                    subjects.append((subject['start'], subject['end'], subject['entity_type'], subject['text']))
        
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])

        if len(prompt_label_idxs) > 0:
            max_len = max([len(prompt_label_idx) for prompt_label_idx in prompt_label_idxs])
            prompt_label_idxs = torch.LongTensor([prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(prompt_label_idx)) for prompt_label_idx in prompt_label_idxs])
            prompt_label_attn = torch.LongTensor([prompt_label_att + [0]*(max_len-len(prompt_label_att)) for prompt_label_att in prompt_label_attn])
            
            prompt_label_idxs = prompt_label_idxs.cuda()
            prompt_label_attn = prompt_label_attn.cuda()
            offsets = (torch.LongTensor(offsets).unsqueeze(1).unsqueeze(2)).cuda()

        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        subject_types = torch.cuda.LongTensor(subject_types)
        ner_feats = torch.cuda.LongTensor(ner_feats)
        relation_seqidxs = torch.cuda.LongTensor(relation_seqidxs)
        return enc_idxs, enc_attn, token_masks, prompt_label_idxs, prompt_label_attn, offsets, relation_seqidxs, subject_types, token_lens, torch.cuda.LongTensor(token_nums), subjects, ner_feats

    def binding(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        for i, layer_module in enumerate(self.bert_binding):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

    def encode(self, piece_idxs, attention_masks, token_mask, token_lens, prime_idxs, prime_attn_masks, index_map):
        # encode the passage part
        all_passage_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        if self.config.model_type.startswith("early"):
            passage_outputs = all_passage_outputs[0]
            # augment back using index_map
            index_map_feat = index_map.expand(-1, passage_outputs.size(1), passage_outputs.size(2))
            passage_outputs = torch.gather(passage_outputs, dim=0, index=index_map_feat)
            index_map_attn = (index_map.squeeze(1)).expand(-1, passage_outputs.size(1))
            passage_attn_mask = torch.gather(attention_masks, dim=0, index=index_map_attn)

            # encode the priming part
            all_prime_outputs = self.bert_label(prime_idxs, attention_mask=prime_attn_masks)
            prime_outputs = all_prime_outputs[0]

            # pass through binding layer
            transformer_mask = torch.cat([passage_attn_mask, prime_attn_masks], dim=1)
            bert_outputs = self.binding((torch.cat([passage_outputs, prime_outputs], dim=1)), transformer_mask)
        else:
            bert_outputs = all_passage_outputs[0]
        
        batch_size = bert_outputs.size(0)
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets_masked(token_lens, token_mask)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs_masked(token_lens, token_mask)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def span_id(self, bert_outputs, token_nums, target=None, predict=False):
        loss = 0.0
        entities = None
        entity_label_scores = self.relation_label_ffn(bert_outputs)
        if self.config.use_crf:
            entity_label_scores_ = self.relation_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.relation_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.relation_label_stoi,
                                            self.relation_type_stoi)
            else: 
                entity_label_loglik = self.relation_crf.loglik(entity_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= entity_label_loglik.mean()
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.relation_label_stoi,
                                            self.relation_type_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.relation_label_num), target.view(-1))
 
        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, enc_token_mask, prompt_label_idxs, prompt_label_attn, offsets, rel_seqidxs, subject_types, token_lens, token_nums, subjects, ner_feats = self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, enc_token_mask, token_lens, prompt_label_idxs, prompt_label_attn, offsets)
        if self.config.use_span_feature:
            # get span embedding
            subject_vec = get_entity_embedding(bert_outputs, subjects)
            extend_subj_vec = subject_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_subj_vec), dim=-1)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(subject_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        if self.config.use_ner_feature:
            ner_feature = self.ner_feature_module(ner_feats)
            bert_outputs = torch.cat((bert_outputs, ner_feature), dim=-1)

        span_id_loss, _ = self.span_id(bert_outputs, token_nums, rel_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            if len([entities for tokens, entities in zip(batch.tokens, batch.entities) if len(entities)>0]) == 0:
                self.train()
                return [[] for _ in range(len(batch.entities))], [[] for _ in range(len(batch.entities))]
            # process data
            enc_idxs, enc_attn, enc_token_mask, prompt_label_idxs, prompt_label_attn, offsets, rel_seqidxs, subject_types, token_lens, token_nums, subjects, ner_feats = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, enc_token_mask, token_lens, prompt_label_idxs, prompt_label_attn, offsets)
            if self.config.use_span_feature:
                # get span embedding
                subject_vec = get_entity_embedding(bert_outputs, subjects)
                extend_subj_vec = subject_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_subj_vec), dim=-1)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(subject_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            if self.config.use_ner_feature:
                ner_feature = self.ner_feature_module(ner_feats)
                bert_outputs = torch.cat((bert_outputs, ner_feature), dim=-1)
            _, rel_candidates = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            cnt = 0
            new_relations = []
            new_input_texts = []
            for b_idx, entities in enumerate(batch.entities):
                new_sub_relations = []
                new_sub_input_texts = []
                ent_span2type = {(ent['start'], ent['end']): ent['entity_type'] for ent in entities}
                for subject in entities:
                    for rel in rel_candidates[cnt]:
                        if (rel[0], rel[1]) in ent_span2type: # check it's an entity
                            obj_type = ent_span2type[(rel[0], rel[1])]
                            if ((rel[0], rel[1]) == (subject['start'], subject['end'])) :
                                continue
                            if (subject["entity_type"], rel[2]) in type_constraint[self.dataset].keys():
                                if obj_type in type_constraint[self.dataset][(subject["entity_type"], rel[2])]: # entity-type-relation constraint
                                    new_sub_relations.append((subject['start'], subject['end'], subject['entity_type'], rel[0], rel[1], obj_type, rel[2]))
                    if self.config.model_type.startswith("early"):
                        new_sub_input_texts.append(input_texts[b_idx])
                    else:
                        new_sub_input_texts.append(input_texts[cnt])
                    cnt += 1
                new_relations.append(new_sub_relations)
                new_input_texts.append(new_sub_input_texts)
            assert cnt == bert_outputs.size(0)
        self.train()
        return new_relations, new_input_texts

class REModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.entity_type_stoi = self.vocabs['entity_type']
        self.entity_type_num = len(self.entity_type_stoi)

        if self.config.model_type.startswith("sep") or self.config.model_type.startswith("early"):
            self.relation_label_stoi = vocabs['unified_label']
            self.relation_type_stoi = vocabs['unified_type']
        else:
            self.relation_label_stoi = vocabs['relation_label']
            self.relation_type_stoi = vocabs['relation_type']

        self.relation_label_itos = {i:s for s, i in self.relation_label_stoi.items()}
        self.relation_type_itos = {i: s for s, i in self.relation_type_stoi.items()}
        
        self.relation_label_num = len(self.relation_label_stoi)
        self.relation_type_num = len(self.relation_type_stoi)

        # BERT encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
            self.start_token = "[CLS]"
            self.end_token = "[SEP]"
            self.sep_token = self.tokenizer.sep_token
        elif self.bert_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
            self.start_token = "<s>"
            self.end_token = "</s>"
            self.sep_token = self.tokenizer.sep_token
        elif self.bert_model_name.startswith('albert-'):
            self.bert = AlbertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = AlbertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
            self.start_token = "[CLS]"
            self.end_token = "[SEP]"
            self.sep_token = self.tokenizer.sep_token
        elif self.bert_model_name.startswith('allenai/scibert'):
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
            self.start_token = "[CLS]"
            self.end_token = "[SEP]"
            self.sep_token = self.tokenizer.sep_token
        else:
            raise ValueError

        if self.config.model_type.startswith("early"):
            # binding Transformer
            self.bert_binding = copy.deepcopy(self.bert.encoder.layer[self.config.cut_layer:])

            # cut layer
            assert self.config.cut_layer < len(self.bert.encoder.layer)
            self.bert.encoder.layer = self.bert.encoder.layer[:self.config.cut_layer]

            # BERT label encoder
            self.bert_label = self.bert

        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        if self.config.use_span_feature:
            feature_dim = self.bert_dim*2
        else:
            feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.entity_type_num, 100)

        self.rel_type_feature_stoi = vocabs['relation_type']
        if self.config.use_rel_feature:
            assert self.config.model_type.startswith("sep")
            feature_dim += 100
            self.rel_feature_module = nn.Embedding(len(self.rel_type_feature_stoi), 100)

        self.relation_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.relation_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.relation_crf = CRF(self.relation_label_stoi, bioes=False)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        
        ## Info in this part is used for self.bert
        enc_idxs = []
        enc_attn = []
        
        # Note: if we do not us binding, then the below three list will not be use
        ## Info in this part is used for self.bert_label
        prompt_label_idxs = []
        prompt_label_attn = []
        ## We need this to perform duplication for the output of self.bert
        offsets = []
        
        relation_seqidxs = []
        subject_types = []
        relation_types = [] # will only be used for sep model
        token_lens = []
        token_masks = []
        token_nums = []
        subjects = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        
        for tokens, piece, subject, relation, token_len, token_mask, token_num, sub_tag_piece, sub_tag_token_len, sub_tag_mask in zip(batch.tokens, batch.pieces, batch.subjects, batch.relations, batch.token_lens, batch.use_token_masks, batch.token_nums, batch.sub_tagged_pieces, batch.sub_tagged_token_lens, batch.sub_tagged_masks):
            if self.config.model_type.startswith("sep"):
                valid_roles = sorted(patterns[self.dataset][subject[2]])
                for candidate in valid_roles:
                    if self.config.model_type == "sep+Subjprompt+Typeprompt+Relprompt":
                        relation_map = relation_tags[self.dataset][self.config.relation_type]
                        ner_map = ner_tags[self.dataset][self.config.subject_type]
                        prompt = "{} {} {} {} {} {}".format(self.sep_token, subject[3],
                                                self.sep_token, ner_map[subject[2]], 
                                                self.sep_token, relation_map[candidate])
                        use_piece = piece
                        use_token_len = token_len
                        use_token_mask = token_mask
                    elif self.config.model_type == "sep+subtag+Subjprompt+Typeprompt+Relprompt":
                        relation_map = relation_tags[self.dataset][self.config.relation_type]
                        ner_map = ner_tags[self.dataset][self.config.subject_type]
                        prompt = "{} {} {} {} {} {}".format(self.sep_token, subject[3],
                                                self.sep_token, ner_map[subject[2]], 
                                                self.sep_token, relation_map[candidate])
                        use_piece = sub_tag_piece
                        use_token_len = sub_tag_token_len
                        use_token_mask = sub_tag_mask
                    elif self.config.model_type == "sep+Subjprompt+Typeprompt":
                        ner_map = ner_tags[self.dataset][self.config.subject_type]
                        prompt = "{} {} {} {}".format(self.sep_token, subject[3],
                                                self.sep_token, ner_map[subject[2]])
                        use_piece = piece
                        use_token_len = token_len
                        use_token_mask = token_mask
                    elif self.config.model_type == "sep+subtag+Subjprompt+Typeprompt":
                        ner_map = ner_tags[self.dataset][self.config.subject_type]
                        prompt = "{} {} {} {}".format(self.sep_token, subject[3],
                                                self.sep_token, ner_map[subject[2]])
                        use_piece = sub_tag_piece
                        use_token_len = sub_tag_token_len
                        use_token_mask = sub_tag_mask
                    elif self.config.model_type == "sep":
                        prompt = ""
                        use_piece = piece
                        use_token_len = token_len
                        use_token_mask = token_mask
                    else:
                        raise ValueError

                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False, is_split_into_words=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(use_piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.start_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.end_token)]
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    rel_seq = get_relation_seqlabels(relation, len(tokens), specify_relation=candidate)
                    subject_types.append(self.entity_type_stoi[subject[2]])
                    relation_types.append(self.rel_type_feature_stoi[candidate])
                    if self.config.use_crf:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [-100] * (max_token_num-len(tokens)))
                    token_lens.append(use_token_len)
                    token_nums.append(token_num)
                    token_masks.append(use_token_mask)
                    subjects.append(subject)
            elif self.config.model_type.startswith("early"):
                valid_roles = sorted(patterns[self.dataset][subject[2]])
                relation_map = relation_tags[self.dataset][self.config.relation_type]
                ner_map = ner_tags[self.dataset][self.config.subject_type]
                prompt = "{} {} {} {}".format(self.sep_token, subject[3],
                                            self.sep_token, ner_map[subject[2]])
                if self.config.model_type == "early+Subjprompt+Typeprompt+Relprompt":
                    use_piece = piece
                    use_token_len = token_len
                    use_token_mask = token_mask
                elif self.config.model_type == "early+subtag+Subjprompt+Typeprompt+Relprompt":
                    use_piece = sub_tag_piece
                    use_token_len = sub_tag_token_len
                    use_token_mask = sub_tag_mask
                else:
                    raise ValueError
                prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False, is_split_into_words=False)      
                piece_id = self.tokenizer.convert_tokens_to_ids(use_piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.start_token)] + \
                            piece_id + \
                            prompt_id + \
                            [self.tokenizer.convert_tokens_to_ids(self.end_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    prompt_label = "{}".format(relation_map[candidate])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))
                    rel_seq = get_relation_seqlabels(relation, len(tokens), specify_relation=candidate)
                    if self.config.use_crf:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [-100] * (max_token_num-len(tokens)))

                    token_lens.append(use_token_len)
                    token_nums.append(token_num)
                    token_masks.append(use_token_mask)
                    subject_types.append(self.entity_type_stoi[subject[2]])
                    subjects.append(subject)
                    offsets.append(len(enc_idxs)-1)
            else:
                if self.config.model_type == "Subjprompt+Typeprompt":
                    ner_map = ner_tags[self.dataset][self.config.subject_type]
                    prompt = "{} {} {} {}".format(self.sep_token, subject[3],
                                            self.sep_token, ner_map[subject[2]])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False, is_split_into_words=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.start_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.end_token)]
                    use_token_len = token_len
                    use_token_mask = token_mask
                
                elif self.config.model_type == "subtag+Subjprompt+Typeprompt":
                    ner_map = ner_tags[self.dataset][self.config.subject_type]
                    prompt = "{} {} {} {}".format(self.sep_token, subject[3],
                                            self.sep_token, ner_map[subject[2]])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False, is_split_into_words=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(sub_tag_piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.start_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.end_token)]
                    use_token_len = sub_tag_token_len
                    use_token_mask = sub_tag_mask
                
                elif self.config.model_type == "noprompt":
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.start_token)] + \
                                piece_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.end_token)]
                    use_token_len = token_len
                    use_token_mask = token_mask
                
                elif self.config.model_type == "subtag+noprompt":
                    piece_id = self.tokenizer.convert_tokens_to_ids(sub_tag_piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.start_token)] + \
                                piece_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.end_token)]
                    use_token_len = sub_tag_token_len
                    use_token_mask = sub_tag_mask

                else:
                    raise ValueError
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))  
                rel_seq = get_relation_seqlabels(relation, len(tokens))
                subject_types.append(self.entity_type_stoi[subject[2]])
                token_lens.append(use_token_len)
                token_nums.append(token_num)
                token_masks.append(use_token_mask)
                subjects.append(subject)
                if self.config.use_crf:
                    relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    relation_seqidxs.append([self.relation_label_stoi[s] for s in rel_seq] + [-100] * (max_token_num-len(tokens)))
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        
        if len(prompt_label_idxs) > 0:
            max_len = max([len(prompt_label_idx) for prompt_label_idx in prompt_label_idxs])
            prompt_label_idxs = torch.LongTensor([prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(prompt_label_idx)) for prompt_label_idx in prompt_label_idxs])
            prompt_label_attn = torch.LongTensor([prompt_label_att + [0]*(max_len-len(prompt_label_att)) for prompt_label_att in prompt_label_attn])
            
            prompt_label_idxs = prompt_label_idxs.cuda()
            prompt_label_attn = prompt_label_attn.cuda()
            offsets = (torch.LongTensor(offsets).unsqueeze(1).unsqueeze(2)).cuda()        

        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        subject_types = torch.cuda.LongTensor(subject_types)
        relation_types = torch.cuda.LongTensor(relation_types)
        relation_seqidxs = torch.cuda.LongTensor(relation_seqidxs)
        return enc_idxs, enc_attn, token_masks, prompt_label_idxs, prompt_label_attn, offsets, relation_seqidxs, subject_types, relation_types, token_lens, torch.cuda.LongTensor(token_nums), subjects

    def binding(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        for i, layer_module in enumerate(self.bert_binding):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

    def encode(self, piece_idxs, attention_masks, token_mask, token_lens, prime_idxs, prime_attn_masks, index_map):
        # encode the passage part
        all_passage_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        if self.config.model_type.startswith("early"):
            passage_outputs = all_passage_outputs[0]
            # augment back using index_map
            index_map_feat = index_map.expand(-1, passage_outputs.size(1), passage_outputs.size(2))
            passage_outputs = torch.gather(passage_outputs, dim=0, index=index_map_feat)
            index_map_attn = (index_map.squeeze(1)).expand(-1, passage_outputs.size(1))
            passage_attn_mask = torch.gather(attention_masks, dim=0, index=index_map_attn)

            # encode the priming part
            all_prime_outputs = self.bert_label(prime_idxs, attention_mask=prime_attn_masks)
            prime_outputs = all_prime_outputs[0]

            # pass through binding layer
            transformer_mask = torch.cat([passage_attn_mask, prime_attn_masks], dim=1)
            bert_outputs = self.binding((torch.cat([passage_outputs, prime_outputs], dim=1)), transformer_mask)
        else:
            bert_outputs = all_passage_outputs[0]
        
        batch_size = bert_outputs.size(0)
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets_masked(token_lens, token_mask)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs_masked(token_lens, token_mask)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            masks = bert_outputs.new(masks).unsqueeze(-1)
            bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
            bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
            bert_outputs = bert_outputs.sum(2)
        else:
            raise ValueError('Unknown multi-piece token handling strategy: {}'
                             .format(self.multi_piece))
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def span_id(self, bert_outputs, token_nums, target=None, predict=False):
        loss = 0.0
        entities = None
        entity_label_scores = self.relation_label_ffn(bert_outputs)
        if self.config.use_crf:
            entity_label_scores_ = self.relation_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.relation_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.relation_label_stoi,
                                            self.relation_type_stoi)
            else: 
                entity_label_loglik = self.relation_crf.loglik(entity_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= entity_label_loglik.mean()
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.relation_label_stoi,
                                            self.relation_type_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.relation_label_num), target.view(-1))
 
        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, enc_token_mask, prompt_label_idxs, prompt_label_attn, offsets, rel_seqidxs, subject_types, relation_types, token_lens, token_nums, subjects = self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, enc_token_mask, token_lens, prompt_label_idxs, prompt_label_attn, offsets)
        if self.config.use_span_feature:
            # get span embedding
            subject_vec = get_entity_embedding(bert_outputs, subjects)
            extend_subj_vec = subject_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_subj_vec), dim=-1)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(subject_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        if self.config.use_rel_feature:
            rel_feature = self.rel_feature_module(relation_types)
            extend_rel_vec = rel_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_rel_vec), dim=-1)
        
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, rel_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, enc_token_mask, prompt_label_idxs, prompt_label_attn, offsets, rel_seqidxs, subject_types, relation_types, token_lens, token_nums, subjects = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, enc_token_mask, token_lens, prompt_label_idxs, prompt_label_attn, offsets)
            if self.config.use_span_feature:
                # get span embedding
                subject_vec = get_entity_embedding(bert_outputs, subjects)
                extend_subj_vec = subject_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_subj_vec), dim=-1)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(subject_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            if self.config.use_rel_feature:
                rel_feature = self.rel_feature_module(relation_types)
                extend_rel_vec = rel_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_rel_vec), dim=-1)
            _, rel_candidates = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("sep") or self.config.model_type.startswith("early"):
                # decompose predicted arguments
                cnt = 0
                new_relations = []
                new_input_texts = []
                for b_idx, (all_entity, subject, relation) in enumerate(zip(batch.all_entities, batch.subjects, batch.relations)):
                    valid_roles = sorted(patterns[self.dataset][subject[2]])
                    ent_span2type = {(ent['start'], ent['end']): ent['entity_type'] for ent in all_entity}
                    new_sub_relations = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        for rel in rel_candidates[cnt]:
                            if ((rel[0], rel[1]) == (subject[0], subject[1])) :
                                continue
                            if (rel[0], rel[1]) in ent_span2type:
                                obj_type = ent_span2type[(rel[0], rel[1])]
                                if obj_type in type_constraint[self.dataset][(subject[2], candidate)]:
                                    new_sub_relations.append([rel[0], rel[1], candidate])
                        if self.config.model_type.startswith("early"):
                            new_sub_input_texts.append(input_texts[b_idx])
                        else:
                            new_sub_input_texts.append(input_texts[cnt])
                        cnt += 1
                    new_relations.append(new_sub_relations)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == bert_outputs.size(0)
                return new_relations, new_input_texts
            else:
                assert len(rel_candidates) == len(batch.all_entities)
                new_relations = []
                for all_entity, subject, relation in zip(batch.all_entities, batch.subjects, rel_candidates):
                    ent_span2type = {(ent['start'], ent['end']): ent['entity_type'] for ent in all_entity}
                    new_sub_relations = []
                    for rel in relation:
                        if ((rel[0], rel[1]) == (subject[0], subject[1])) :
                            continue
                        if (rel[0], rel[1]) in ent_span2type:
                            obj_type = ent_span2type[(rel[0], rel[1])]
                            if ((subject[2], rel[2]) in type_constraint[self.dataset]) and (obj_type in type_constraint[self.dataset][(subject[2], rel[2])]):
                                new_sub_relations.append(rel)
                    new_relations.append(new_sub_relations)
        self.train()
        return new_relations, input_texts