import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, RobertaModel, XLMRobertaModel, AutoModel, BertConfig, RobertaConfig, XLMRobertaConfig
import copy, random
import ipdb

from pattern import *
from model_utils import (token_lens_to_offsets, token_lens_to_idxs, tag_paths_to_spans, get_role_seqlabels, get_slot_seqlabels, get_trigger_embedding, CRF, Linears, filter_use_ner, get_type_embedding_from_bert)
from sub_models import EDModel, NERModel
from model_re import REStage1Model, REModel
from model_eaefeat import EAEfeatModel

from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

class StructuralModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset='ace05e'):
        super().__init__()
        if config.task == 'eae':
            if config.model_type.startswith("late"):
                self.model = EAELateBind(config, tokenizer, vocabs, dataset)
            elif config.model_type.startswith("early"):
                self.model = EAEEarlyBind_Position(config, tokenizer, vocabs, dataset)
                # self.model = EAEEarlyBind(config, tokenizer, vocabs, dataset)
            elif config.model_type.startswith('feat'):
                self.model = EAEfeatModel(config, tokenizer, vocabs, dataset)
            else:
                self.model = EAEModel(config, tokenizer, vocabs, dataset)
        elif config.task == 're_stage1':
            self.model = REStage1Model(config, tokenizer, vocabs, dataset)
        elif config.task == 're':
            self.model = REModel(config, tokenizer, vocabs, dataset)
        elif config.task == 'ed':
            self.model = EDModel(config, tokenizer, vocabs, dataset)
        elif config.task == 'ner':
            self.model = NERModel(config, tokenizer, vocabs, dataset)
        elif config.task == 'sp':
            self.model = SPModel(config, tokenizer, vocabs, dataset)
        self.tt = tokenizer
    
    def forward(self, batch):
        return self.model(batch)
    
    def predict(self, batch, **kwargs):
        return self.model.predict(batch, **kwargs)

class EAEModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.trigger_type_stoi = self.vocabs['trigger_type']
        self.trigger_type_num = len(self.trigger_type_stoi)

        if not self.config.model_type.startswith("sep"):
            self.role_label_stoi = vocabs['role_label']
            self.role_type_stoi = vocabs['role_type']
        else:
            if self.config.use_unified_label:
                self.role_label_stoi = vocabs['unified_label']
                self.role_type_stoi = vocabs['unified_type']
            else:
                self.role_label_stoi = vocabs['role_label']
                self.role_type_stoi = vocabs['role_type']
        self.role_label_itos = {i:s for s, i in self.role_label_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        
        self.role_label_num = len(self.role_label_stoi)
        self.role_type_num = len(self.role_type_stoi)

        # BERT encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('xlm-'):
            self.bert = XLMRobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = XLMRobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        
        else:
            raise ValueError
        
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        if self.config.use_trigger_feature:
            feature_dim = self.bert_dim*2
        else:
            feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.trigger_type_num, 100)
        elif self.config.use_init_type_feature:
            feature_dim += self.bert_dim
            # create initialization embedding
            weight = get_type_embedding_from_bert(self.bert, tokenizer, self.trigger_type_stoi)
            self.type_feature_module = nn.Embedding.from_pretrained(weight)

        self.role_type_feature_stoi = vocabs['role_type']
        if self.config.use_role_feature:
            assert self.config.model_type.startswith("sep")
            feature_dim += 100
            self.role_feature_module = nn.Embedding(len(self.role_type_feature_stoi), 100)
        elif self.config.use_init_role_feature:
            feature_dim += self.bert_dim
            weight = get_type_embedding_from_bert(self.bert, tokenizer, self.role_type_feature_stoi)
            self.role_feature_module = nn.Embedding.from_pretrained(weight)

        self.role_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.role_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.role_crf = CRF(self.role_label_stoi, bioes=False)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        enc_idxs = []
        enc_attn = []
        role_seqidxs = []
        trigger_types = []
        role_types = [] # will only be used for sep model
        token_lens = []
        token_nums = []
        triggers = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        
        for tokens, piece, trigger, role, token_len, token_num in zip(batch.tokens, batch.pieces, batch.triggers, batch.roles, batch.token_lens, batch.token_nums):
            # separate model
            if self.config.model_type.startswith("sep"):
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                for candidate in valid_roles:
                    if self.config.model_type == "sep+CRF+Triprompt+Typeprompt":
                        evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                        role_map = role_tags[self.dataset][self.config.role_type]
                        prompt = "{} {} {} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                    self.tokenizer.sep_token, trigger[3],
                                                    self.tokenizer.sep_token, role_map[candidate])

                    elif self.config.model_type == "sep+CRF+Triprompt":
                        role_map = role_tags[self.dataset][self.config.role_type]
                        prompt = "{} {} {} {}".format(self.tokenizer.sep_token, trigger[3],
                                                    self.tokenizer.sep_token, role_map[candidate])

                    elif self.config.model_type == "sep+CRF":
                        role_map = role_tags[self.dataset][self.config.role_type]
                        prompt = "{} {}".format(self.tokenizer.sep_token, role_map[candidate])
                    elif self.config.model_type == "sep+CRF+rolefeat":
                        prompt = ""
                    else:
                        raise ValueError
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    role_types.append(self.role_type_feature_stoi[candidate])
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    
            # joint model
            else:
                if self.config.model_type == "CRF+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt = "{} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                elif self.config.model_type == "CRF+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt = "{} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]                
                elif self.config.model_type == "CRF+Triprompt+Typeprompt+Rolehint":
                    valid_roles = sorted(list(role_maps[self.dataset][trigger[2]].keys()))
                    prompt = "{} {} {} {} {} {}".format(self.tokenizer.sep_token, evetype_tags[trigger[2]], 
                                            self.tokenizer.sep_token, trigger[3], self.tokenizer.sep_token, 
                                            f" {self.tokenizer.sep_token} ".join([role_tags[r] for r in valid_roles]))
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]

                elif self.config.model_type == "CRF+Triprompt":
                    prompt = "{} {}".format(self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                elif self.config.model_type == "CRF":
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                else:
                    raise ValueError
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))  
                role_seq = get_role_seqlabels(role, len(tokens))
                trigger_types.append(self.trigger_type_stoi[trigger[2]])
                token_lens.append(token_len)
                token_nums.append(token_num)
                triggers.append(trigger)
                if self.config.use_crf:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        trigger_types = torch.LongTensor(trigger_types)
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        trigger_types = trigger_types.cuda()
        role_types = torch.cuda.LongTensor(role_types)
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        return enc_idxs, enc_attn, role_seqidxs, trigger_types, role_types, token_lens, torch.cuda.LongTensor(token_nums), triggers
    
    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
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
        entity_label_scores = self.role_label_ffn(bert_outputs)
        if self.config.use_crf:
            entity_label_scores_ = self.role_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.role_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else: 
                entity_label_loglik = self.role_crf.loglik(entity_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= entity_label_loglik.mean()
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.role_label_num), target.view(-1))

        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, role_seqidxs, trigger_types, role_types, token_lens, token_nums, triggers = self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        if self.config.use_trigger_feature:
            # get trigger embedding
            trigger_vec = get_trigger_embedding(bert_outputs, triggers)
            extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
        if self.config.use_type_feature or self.config.use_init_type_feature:
            type_feature = self.type_feature_module(trigger_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        if self.config.use_role_feature or self.config.use_init_role_feature:
            role_feature = self.role_feature_module(role_types)
            extend_role_vec = role_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_role_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, role_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch, use_ner_filter=False):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, trigger_types, role_types, token_lens, token_nums, triggers = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            if self.config.use_trigger_feature:
                # get trigger embedding
                trigger_vec = get_trigger_embedding(bert_outputs, triggers)
                extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
            if self.config.use_type_feature or self.config.use_init_type_feature:
                type_feature = self.type_feature_module(trigger_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            if self.config.use_role_feature or self.config.use_init_role_feature:
                role_feature = self.role_feature_module(role_types)
                extend_role_vec = role_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_role_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("sep"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx, (trigger, role) in enumerate(zip(batch.triggers, batch.roles)):
                    valid_roles = sorted(patterns[self.dataset][trigger[2]])
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        if self.config.use_unified_label:
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                        else:
                            # new_sub_arguments.extend([[argu[0], argu[1], argu[2]] for argu in arguments[cnt]]) 
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                            # TODO: not sure whether need to change here if use_unified_label = True
                        new_sub_input_texts.append(input_texts[cnt])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == enc_idxs.size(0)
                if use_ner_filter:
                    new_arguments = filter_use_ner(new_arguments, batch.entities)
                return new_arguments, new_input_texts
            if use_ner_filter:
                arguments = filter_use_ner(arguments, batch.entities)
        self.train()
        return arguments, input_texts

class EAELateBind(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.trigger_type_stoi = self.vocabs['trigger_type']
        self.trigger_type_num = len(self.trigger_type_stoi)

        if self.config.model_type.startswith("late+trigger+T"):
            self.role_label_stoi = vocabs['role_label']
            self.role_type_stoi = vocabs['role_type']
        else:
            if self.config.use_unified_label:
                self.role_label_stoi = vocabs['unified_label']
                self.role_type_stoi = vocabs['unified_type']
            else:
                self.role_label_stoi = vocabs['role_label']
                self.role_type_stoi = vocabs['role_type']

        self.role_label_itos = {i:s for s, i in self.role_label_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        
        self.role_label_num = len(self.role_label_stoi)
        self.role_type_num = len(self.role_type_stoi)

        # BERT sentence encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        else:
            raise ValueError

        # BERT label encoder
        if self.config.share_bert:
            self.bert_label = self.bert
        else:
            if self.bert_model_name.startswith('bert-'):
                self.bert_label = BertModel.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir,
                                                    output_hidden_states=True)
            elif self.bert_model_name.startswith('roberta-'):
                self.bert_label = RobertaModel.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir,
                                                    output_hidden_states=True)
            else:
                raise ValueError        
        
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_label.resize_token_embeddings(len(self.tokenizer))
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # Late binding Transformer
        encoder_layer = TransformerEncoderLayer(self.bert_dim, 8)
        encoder_norm = LayerNorm(self.bert_dim)
        self.binding_layer = TransformerEncoder(encoder_layer, self.config.binding_layer, encoder_norm)

        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        if self.config.use_trigger_feature:
            feature_dim = self.bert_dim*2
        else:
            feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.trigger_type_num, 100)

        self.role_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.role_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.role_crf = CRF(self.role_label_stoi, bioes=False)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        
        ## Info in this part is used for self.bert
        enc_idxs = []
        enc_attn = []
        ## Info in this part is used for self.bert_label
        prompt_label_idxs = []
        prompt_label_attn = []
        ## We need this to perform duplication for the output of self.bert
        offsets = []
        ## These are the same as EAE model
        role_seqidxs = []
        trigger_types = []
        token_lens = []
        token_nums = []
        triggers = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        for tokens, piece, trigger, role, token_len, token_num in zip(batch.tokens, batch.pieces, batch.triggers, batch.roles, batch.token_lens, batch.token_nums):
            if self.config.model_type.startswith("late+role"): # for this part, we only consider role part priming in bert_label
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                role_map = role_tags[self.dataset][self.config.role_type]
                if self.config.model_type == "late+role+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt = "{} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)

                elif self.config.model_type == "late+role+Triprompt":
                    prompt = "{} {}".format(self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    
                elif self.config.model_type == "late+role":
                    prompt_id = []

                else:
                    raise ValueError
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + \
                            prompt_id + \
                            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    prompt_label = "{}".format(role_map[candidate])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    offsets.append(len(enc_idxs)-1)
                
            elif self.config.model_type.startswith("late+trigger+role"): # for this part, we consider priming trigger+role in bert_label    
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                role_map = role_tags[self.dataset][self.config.role_type]
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    if self.config.model_type == "late+trigger+role+Triprompt+Typeprompt":
                        evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                        prompt_label = "{} {} {} {} {}".format(evetype_map[trigger[2]], self.tokenizer.sep_token, trigger[3],
                                                        self.tokenizer.sep_token, role_map[candidate])
                        prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                        prompt_label_idxs.append(prompt_label_idx)
                        prompt_label_attn.append([1]*len(prompt_label_idx))

                    elif self.config.model_type == "late+trigger+role+Triprompt":
                        prompt_label = "{} {} {}".format(trigger[3], self.tokenizer.sep_token, role_map[candidate])
                        prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                        prompt_label_idxs.append(prompt_label_idx)
                        prompt_label_attn.append([1]*len(prompt_label_idx))

                    else:
                        raise ValueError

                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    offsets.append(len(enc_idxs)-1)

            elif self.config.model_type.startswith("late+trigger"): # for this part, we consider trigger part only priming in bert_label
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))

                if self.config.model_type == "late+trigger+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt_label = "{} {} {}".format(evetype_map[trigger[2]], self.tokenizer.sep_token, trigger[3])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))
                
                elif self.config.model_type == "late+trigger+Triprompt+Typepromp+Rolehint":
                    valid_roles = sorted(list(role_maps[self.dataset][trigger[2]].keys()))
                    prompt_label = "{} {} {} {} {}".format(evetype_tags[trigger[2]], self.tokenizer.sep_token, trigger[3], self.tokenizer.sep_token, 
                                            f" {self.tokenizer.sep_token} ".join([role_tags[r] for r in valid_roles]))
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                elif self.config.model_type == "late+trigger+Triprompt":
                    prompt_label = "{}".format(trigger[3])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                else:
                    raise ValueError
                role_seq = get_role_seqlabels(role, len(tokens))
                trigger_types.append(self.trigger_type_stoi[trigger[2]])
                token_lens.append(token_len)
                token_nums.append(token_num)
                triggers.append(trigger)
                offsets.append(len(enc_idxs)-1)
                if self.config.use_crf:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
        
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])

        max_len = max([len(prompt_label_idx) for prompt_label_idx in prompt_label_idxs])
        prompt_label_idxs = torch.LongTensor([prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(prompt_label_idx)) for prompt_label_idx in prompt_label_idxs])
        prompt_label_attn = torch.LongTensor([prompt_label_att + [0]*(max_len-len(prompt_label_att)) for prompt_label_att in prompt_label_attn])
        
        offsets = (torch.LongTensor(offsets).unsqueeze(1).unsqueeze(2)).cuda()
        trigger_types = torch.LongTensor(trigger_types)
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        prompt_label_idxs = prompt_label_idxs.cuda()
        prompt_label_attn = prompt_label_attn.cuda()
        trigger_types = trigger_types.cuda()
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        return enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, role_seqidxs, trigger_types, token_lens, torch.cuda.LongTensor(token_nums), triggers
    
    def encode(self, piece_idxs, attention_masks, prime_idxs, prime_attn_masks, index_map, token_lens):
        batch_size, _ = prime_idxs.size()
        # encode the passage part
        all_passage_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
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
        transformer_mask = ~(torch.cat([passage_attn_mask, prime_attn_masks], dim=1).bool())
        bert_outputs = self.binding_layer((torch.cat([passage_outputs, prime_outputs], dim=1)).transpose(0,1), src_key_padding_mask=transformer_mask)
        bert_outputs = torch.transpose(bert_outputs, 0, 1)
        
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
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
        entity_label_scores = self.role_label_ffn(bert_outputs)
        if self.config.use_crf:
            entity_label_scores_ = self.role_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.role_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else: 
                entity_label_loglik = self.role_crf.loglik(entity_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= entity_label_loglik.mean()
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.role_label_num), target.view(-1))

        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, rol_seqidxs, trigger_types, token_lens, token_nums, triggers = self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, token_lens)
        if self.config.use_trigger_feature:
            # get trigger embedding
            trigger_vec = get_trigger_embedding(bert_outputs, triggers)
            extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(trigger_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, rol_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch, use_ner_filter=False):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, rol_seqidxs, trigger_types, token_lens, token_nums, triggers = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, token_lens)
            if self.config.use_trigger_feature:
                # get trigger embedding
                trigger_vec = get_trigger_embedding(bert_outputs, triggers)
                extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(trigger_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("late+role") or self.config.model_type.startswith("late+trigger+role"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx, (trigger, role) in enumerate(zip(batch.triggers, batch.roles)):
                    valid_roles = sorted(patterns[self.dataset][trigger[2]])
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        if self.config.use_unified_label:
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                        else:
                            # new_sub_arguments.extend([[argu[0], argu[1], argu[2]] for argu in arguments[cnt]]) 
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                            # TODO: not sure whether need to change here if use_unified_label = True
                        new_sub_input_texts.append(input_texts[offsets[cnt, 0, 0]])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == prompt_label_idxs.size(0)
                if use_ner_filter:
                    new_arguments = filter_use_ner(new_arguments, batch.entities)
                return new_arguments, new_input_texts
            if use_ner_filter:
                arguments = filter_use_ner(arguments, batch.entities)
        self.train()
        return arguments, input_texts

class EAEEarlyBind(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.trigger_type_stoi = self.vocabs['trigger_type']
        self.trigger_type_num = len(self.trigger_type_stoi)

        if self.config.model_type.startswith("early+trigger+T"):
            self.role_label_stoi = vocabs['role_label']
            self.role_type_stoi = vocabs['role_type']
        else:
            if self.config.use_unified_label:
                self.role_label_stoi = vocabs['unified_label']
                self.role_type_stoi = vocabs['unified_type']
            else:
                self.role_label_stoi = vocabs['role_label']
                self.role_type_stoi = vocabs['role_type']

        self.role_label_itos = {i:s for s, i in self.role_label_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        
        self.role_label_num = len(self.role_label_stoi)
        self.role_type_num = len(self.role_type_stoi)

        # BERT sentence encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        else:
            raise ValueError

        # binding Transformer
        self.bert_binding = copy.deepcopy(self.bert.encoder.layer[self.config.cut_layer:])

        # cut layer
        assert self.config.cut_layer < len(self.bert.encoder.layer)
        self.bert.encoder.layer = self.bert.encoder.layer[:self.config.cut_layer]

        # BERT label encoder
        if self.config.share_bert:
            self.bert_label = self.bert
        else:
            self.bert_label = copy.deepcopy(self.bert)
        
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_label.resize_token_embeddings(len(self.tokenizer))
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy

        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        if self.config.use_trigger_feature:
            feature_dim = self.bert_dim*2
        else:
            feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.trigger_type_num, 100)

        self.role_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.role_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.role_crf = CRF(self.role_label_stoi, bioes=False)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        
        ## Info in this part is used for self.bert
        enc_idxs = []
        enc_attn = []
        ## Info in this part is used for self.bert_label
        prompt_label_idxs = []
        prompt_label_attn = []
        ## We need this to perform duplication for the output of self.bert
        offsets = []
        ## These are the same as EAE model
        role_seqidxs = []
        trigger_types = []
        token_lens = []
        token_nums = []
        triggers = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        for tokens, piece, trigger, role, token_len, token_num in zip(batch.tokens, batch.pieces, batch.triggers, batch.roles, batch.token_lens, batch.token_nums):
            if self.config.model_type.startswith("early+role"): # for this part, we only consider role part priming in bert_label
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                role_map = role_tags[self.dataset][self.config.role_type]
                if self.config.model_type == "early+role+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt = "{} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)

                elif self.config.model_type == "early+role+Triprompt":
                    prompt = "{} {}".format(self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    
                elif self.config.model_type == "early+role":
                    prompt_id = []

                else:
                    raise ValueError
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + \
                            prompt_id + \
                            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    prompt_label = "{}".format(role_map[candidate])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    # prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    offsets.append(len(enc_idxs)-1)
                
            elif self.config.model_type.startswith("early+trigger+role"): # for this part, we consider priming trigger+role in bert_label    
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                role_map = role_tags[self.dataset][self.config.role_type]
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    if self.config.model_type == "early+trigger+role+Triprompt+Typeprompt":
                        evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                        prompt_label = "{} {} {} {} {}".format(evetype_map[trigger[2]], self.tokenizer.sep_token, trigger[3],
                                                        self.tokenizer.sep_token, role_map[candidate])
                        prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                        prompt_label_idxs.append(prompt_label_idx)
                        prompt_label_attn.append([1]*len(prompt_label_idx))

                    elif self.config.model_type == "early+trigger+role+Triprompt":
                        prompt_label = "{} {} {}".format(trigger[3], self.tokenizer.sep_token, role_map[candidate])
                        prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                        prompt_label_idxs.append(prompt_label_idx)
                        prompt_label_attn.append([1]*len(prompt_label_idx))

                    else:
                        raise ValueError

                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    offsets.append(len(enc_idxs)-1)

            elif self.config.model_type.startswith("early+trigger"): # for this part, we consider trigger part only priming in bert_label
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))

                if self.config.model_type == "early+trigger+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt_label = "{} {} {}".format(evetype_map[trigger[2]], self.tokenizer.sep_token, trigger[3])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))
                
                elif self.config.model_type == "early+trigger+Triprompt+Typepromp+Rolehint":
                    valid_roles = sorted(list(role_maps[self.dataset][trigger[2]].keys()))
                    prompt_label = "{} {} {} {} {}".format(evetype_tags[trigger[2]], self.tokenizer.sep_token, trigger[3], self.tokenizer.sep_token, 
                                            f" {self.tokenizer.sep_token} ".join([role_tags[r] for r in valid_roles]))
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                elif self.config.model_type == "early+trigger+Triprompt":
                    prompt_label = "{}".format(trigger[3])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=True)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                else:
                    raise ValueError
                role_seq = get_role_seqlabels(role, len(tokens))
                trigger_types.append(self.trigger_type_stoi[trigger[2]])
                token_lens.append(token_len)
                token_nums.append(token_num)
                triggers.append(trigger)
                offsets.append(len(enc_idxs)-1)
                if self.config.use_crf:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
        
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])

        max_len = max([len(prompt_label_idx) for prompt_label_idx in prompt_label_idxs])
        prompt_label_idxs = torch.LongTensor([prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(prompt_label_idx)) for prompt_label_idx in prompt_label_idxs])
        prompt_label_attn = torch.LongTensor([prompt_label_att + [0]*(max_len-len(prompt_label_att)) for prompt_label_att in prompt_label_attn])
        
        offsets = (torch.LongTensor(offsets).unsqueeze(1).unsqueeze(2)).cuda()
        trigger_types = torch.LongTensor(trigger_types)
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        prompt_label_idxs = prompt_label_idxs.cuda()
        prompt_label_attn = prompt_label_attn.cuda()
        trigger_types = trigger_types.cuda()
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        return enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, role_seqidxs, trigger_types, token_lens, torch.cuda.LongTensor(token_nums), triggers

    def binding(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1.0 - attention_mask) * -10000.0
        for i, layer_module in enumerate(self.bert_binding):
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
        return hidden_states

    def encode(self, piece_idxs, attention_masks, prime_idxs, prime_attn_masks, index_map, token_lens):
        batch_size, _ = prime_idxs.size()
        # encode the passage part

        # TODO: parallel here, notice that we need to make piece_idxs and prime_idxs in same length if we want parallel
        all_passage_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
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
        # bert_outputs = torch.cat([passage_outputs, prime_outputs], dim=1)
        
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
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
        entity_label_scores = self.role_label_ffn(bert_outputs)
        if self.config.use_crf:
            entity_label_scores_ = self.role_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.role_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else: 
                entity_label_loglik = self.role_crf.loglik(entity_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= entity_label_loglik.mean()
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.role_label_num), target.view(-1))

        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, rol_seqidxs, trigger_types, token_lens, token_nums, triggers = self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, token_lens)
        if self.config.use_trigger_feature:
            # get trigger embedding
            trigger_vec = get_trigger_embedding(bert_outputs, triggers)
            extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(trigger_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, rol_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch, use_ner_filter=False):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, rol_seqidxs, trigger_types, token_lens, token_nums, triggers = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, offsets, token_lens)
            if self.config.use_trigger_feature:
                # get trigger embedding
                trigger_vec = get_trigger_embedding(bert_outputs, triggers)
                extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(trigger_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("early+role") or self.config.model_type.startswith("early+trigger+role"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx, (trigger, role) in enumerate(zip(batch.triggers, batch.roles)):
                    valid_roles = sorted(patterns[self.dataset][trigger[2]])
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        if self.config.use_unified_label:
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                        else:
                            # new_sub_arguments.extend([[argu[0], argu[1], argu[2]] for argu in arguments[cnt]]) 
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                            # TODO: not sure whether need to change here if use_unified_label = True
                        new_sub_input_texts.append(input_texts[offsets[cnt, 0, 0]])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == prompt_label_idxs.size(0)
                if use_ner_filter:
                    new_arguments = filter_use_ner(new_arguments, batch.entities)
                return new_arguments, new_input_texts
            if use_ner_filter:
                arguments = filter_use_ner(arguments, batch.entities)
        self.train()
        return arguments, input_texts

class EAEEarlyBind_Position(EAEEarlyBind):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__(config, tokenizer, vocabs, dataset)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        
        ## Info in this part is used for self.bert
        enc_idxs = []
        enc_attn = []
        ## Info in this part is used for self.bert_label
        prompt_label_idxs = []
        prompt_label_attn = []
        prompt_label_positions = []
        ## We need this to perform duplication for the output of self.bert
        offsets = []
        ## These are the same as EAE model
        role_seqidxs = []
        trigger_types = []
        token_lens = []
        token_nums = []
        triggers = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        for tokens, piece, trigger, role, token_len, token_num in zip(batch.tokens, batch.pieces, batch.triggers, batch.roles, batch.token_lens, batch.token_nums):
            if self.config.model_type.startswith("early+role"): # for this part, we only consider role part priming in bert_label
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                role_map = role_tags[self.dataset][self.config.role_type]
                if self.config.model_type == "early+role+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt = "{} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)

                elif self.config.model_type == "early+role+Triprompt":
                    prompt = "{} {}".format(self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    
                elif self.config.model_type == "early+role":
                    prompt_id = []

                else:
                    raise ValueError
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + prompt_id
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    prompt_label = "{} {}".format(self.tokenizer.sep_token, role_map[candidate])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)
                    prompt_label_idx = prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                    prompt_label_position = list(range(len(enc_idx), len(enc_idx)+len(prompt_label_idx)))
                    prompt_label_positions.append(prompt_label_position)
                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))

                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    offsets.append(len(enc_idxs)-1)
                
            elif self.config.model_type.startswith("early+trigger+role"): # for this part, we consider priming trigger+role in bert_label    
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                role_map = role_tags[self.dataset][self.config.role_type]
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + piece_id 
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                for candidate in valid_roles:
                    if self.config.model_type == "early+trigger+role+Triprompt+Typeprompt":
                        evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                        prompt_label = "{} {} {} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], self.tokenizer.sep_token, 
                                                trigger[3], self.tokenizer.sep_token, role_map[candidate])
                        prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)
                        prompt_label_idx = prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]

                    elif self.config.model_type == "early+trigger+role+Triprompt":
                        prompt_label = "{} {} {} {}".format(self.tokenizer.sep_token, trigger[3], self.tokenizer.sep_token, role_map[candidate])
                        prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)
                        prompt_label_idx = prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]

                    else:
                        raise ValueError

                    prompt_label_idxs.append(prompt_label_idx)
                    prompt_label_attn.append([1]*len(prompt_label_idx))
                    prompt_label_position = list(range(len(enc_idx), len(enc_idx)+len(prompt_label_idx)))
                    prompt_label_positions.append(prompt_label_position)

                    role_seq = get_role_seqlabels(role, len(tokens), specify_role=candidate, use_unified_label=self.config.use_unified_label)
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    
                    trigger_types.append(self.trigger_type_stoi[trigger[2]])
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    triggers.append(trigger)
                    offsets.append(len(enc_idxs)-1)

            elif self.config.model_type.startswith("early+trigger"): # for this part, we consider trigger part only priming in bert_label
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + piece_id
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))

                if self.config.model_type == "early+trigger+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt_label = "{} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], self.tokenizer.sep_token, trigger[3])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)
                
                elif self.config.model_type == "early+trigger+Triprompt+Typepromp+Rolehint":
                    valid_roles = sorted(list(role_maps[self.dataset][trigger[2]].keys()))
                    prompt_label = "{} {} {} {} {} {}".format(self.tokenizer.sep_token, evetype_tags[trigger[2]], self.tokenizer.sep_token, trigger[3], 
                                    self.tokenizer.sep_token, f" {self.tokenizer.sep_token} ".join([role_tags[r] for r in valid_roles]))
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)

                elif self.config.model_type == "early+trigger+Triprompt":
                    prompt_label = "{}".format(trigger[3])
                    prompt_label_idx = self.tokenizer.encode(prompt_label, add_special_tokens=False)

                else:
                    raise ValueError

                prompt_label_idx = prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                prompt_label_idxs.append(prompt_label_idx)
                prompt_label_attn.append([1]*len(prompt_label_idx))
                prompt_label_position = list(range(len(enc_idx), len(enc_idx)+len(prompt_label_idx)))
                prompt_label_positions.append(prompt_label_position)

                role_seq = get_role_seqlabels(role, len(tokens))
                trigger_types.append(self.trigger_type_stoi[trigger[2]])
                token_lens.append(token_len)
                token_nums.append(token_num)
                triggers.append(trigger)
                offsets.append(len(enc_idxs)-1)
                if self.config.use_crf:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
        
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])

        max_len = max([len(prompt_label_idx) for prompt_label_idx in prompt_label_idxs])
        prompt_label_idxs = torch.LongTensor([prompt_label_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(prompt_label_idx)) for prompt_label_idx in prompt_label_idxs])
        prompt_label_attn = torch.LongTensor([prompt_label_att + [0]*(max_len-len(prompt_label_att)) for prompt_label_att in prompt_label_attn])
        prompt_label_positions = torch.cuda.LongTensor([prompt_label_position + [0]*(max_len-len(prompt_label_position)) for prompt_label_position in prompt_label_positions])
        
        offsets = (torch.LongTensor(offsets).unsqueeze(1).unsqueeze(2)).cuda()
        trigger_types = torch.LongTensor(trigger_types)
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        prompt_label_idxs = prompt_label_idxs.cuda()
        prompt_label_attn = prompt_label_attn.cuda()
        trigger_types = trigger_types.cuda()
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        return enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, prompt_label_positions, offsets, role_seqidxs, trigger_types, token_lens, torch.cuda.LongTensor(token_nums), triggers

    def encode(self, piece_idxs, attention_masks, prime_idxs, prime_attn_masks, prime_position, index_map, token_lens):
        batch_size, _ = prime_idxs.size()
        # encode the passage part

        # TODO: parallel here, notice that we need to make piece_idxs and prime_idxs in same length if we want parallel
        all_passage_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        passage_outputs = all_passage_outputs[0]
        # augment back using index_map
        index_map_feat = index_map.expand(-1, passage_outputs.size(1), passage_outputs.size(2))
        passage_outputs = torch.gather(passage_outputs, dim=0, index=index_map_feat)
        index_map_attn = (index_map.squeeze(1)).expand(-1, passage_outputs.size(1))
        passage_attn_mask = torch.gather(attention_masks, dim=0, index=index_map_attn)

        # encode the priming part
        all_prime_outputs = self.bert_label(prime_idxs, attention_mask=prime_attn_masks, position_ids=prime_position)
        prime_outputs = all_prime_outputs[0]

        # pass through binding layer
        transformer_mask = torch.cat([passage_attn_mask, prime_attn_masks], dim=1)
        bert_outputs = self.binding((torch.cat([passage_outputs, prime_outputs], dim=1)), transformer_mask)
        # bert_outputs = torch.cat([passage_outputs, prime_outputs], dim=1)
        
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
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

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, prompt_label_positions, offsets, rol_seqidxs, trigger_types, token_lens, token_nums, triggers =\
            self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, prompt_label_positions, offsets, token_lens)
        if self.config.use_trigger_feature:
            # get trigger embedding
            trigger_vec = get_trigger_embedding(bert_outputs, triggers)
            extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(trigger_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, rol_seqidxs, predict=False)
        loss = span_id_loss
        return loss

    def predict(self, batch, use_ner_filter=False):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, prompt_label_positions, offsets, rol_seqidxs, trigger_types, token_lens, token_nums, triggers = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, prompt_label_idxs, prompt_label_attn, prompt_label_positions, offsets, token_lens)
            if self.config.use_trigger_feature:
                # get trigger embedding
                trigger_vec = get_trigger_embedding(bert_outputs, triggers)
                extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(trigger_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("early+role") or self.config.model_type.startswith("early+trigger+role"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx, (trigger, role) in enumerate(zip(batch.triggers, batch.roles)):
                    valid_roles = sorted(patterns[self.dataset][trigger[2]])
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        if self.config.use_unified_label:
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                        else:
                            # new_sub_arguments.extend([[argu[0], argu[1], argu[2]] for argu in arguments[cnt]]) 
                            new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                            # TODO: not sure whether need to change here if use_unified_label = True
                        new_sub_input_texts.append(input_texts[offsets[cnt, 0, 0]])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == prompt_label_idxs.size(0)
                if use_ner_filter:
                    new_arguments = filter_use_ner(new_arguments, batch.entities)
                return new_arguments, new_input_texts
            if use_ner_filter:
                arguments = filter_use_ner(arguments, batch.entities)
        self.train()
        return arguments, input_texts

class SPModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.intent_type_stoi = self.vocabs['intent_type']
        self.intent_type_num = len(self.intent_type_stoi)

        if not self.config.model_type.startswith("sep"):
            self.slot_label_stoi = vocabs['slot_label']
            self.slot_type_stoi = vocabs['slot_type']
        else:
            if self.config.use_unified_label:
                self.slot_label_stoi = vocabs['unified_label']
                self.slot_type_stoi = vocabs['unified_type']
            else:
                self.slot_label_stoi = vocabs['slot_label']
                self.slot_type_stoi = vocabs['slot_type']
        self.slot_label_itos = {i:s for s, i in self.slot_label_stoi.items()}
        self.slot_type_itos = {i: s for s, i in self.slot_type_stoi.items()}
        
        self.slot_label_num = len(self.slot_label_stoi)
        self.slot_type_num = len(self.slot_type_stoi)

        # BERT encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('xlm-'):
            self.bert = XLMRobertaModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = XLMRobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        else:
            raise ValueError
        
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.intent_type_num, 100)
        
        # self.slot_type_feature_stoi = vocabs['slot_type']
        self.slot_type_feature_stoi = {k: i for i, k in enumerate(["O"] + sorted(slot_tags["mtop"]["string"].keys()))}
        if self.config.use_slot_feature:
            assert self.config.model_type.startswith("sep")
            feature_dim += 100
            self.slot_feature_module = nn.Embedding(len(self.slot_type_feature_stoi), 100)

        self.slot_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.slot_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.slot_crf = CRF(self.slot_label_stoi, bioes=False)

    def process_data(self, batch):
        # base on the model type, we can design different data format here
        enc_idxs = []
        enc_attn = []
        slot_seqidxs = []
        intent_types = []
        slot_types = []
        token_lens = []
        token_nums = []
        if self.config.model_type.startswith("sep"):
            max_token_num = max([len(tokens) for tokens, intent in zip(batch.tokens, batch.intents) if len(patterns[self.dataset][intent]) > 0])
        else:
            max_token_num = max([len(tokens) for tokens in batch.tokens])
        for tokens, piece, intent, slots, token_len, token_num in zip(batch.tokens, batch.pieces, batch.intents, batch.slots, batch.token_lens, batch.token_nums):
            # separate model
            if self.config.model_type.startswith("sep"):
                valid_slots = sorted(patterns[self.dataset][intent])
                for candidate in valid_slots:
                    if self.config.model_type == "sep+CRF+Typeprompt":
                        slot_map = slot_tags[self.dataset][self.config.slot_type]
                        intent_map = intent_tags[self.config.dataset][self.config.intent_type]
                        if "#" not in intent:
                            prompt = "{} {} {} {}".format(self.tokenizer.sep_token, intent_map[intent], 
                                                        self.tokenizer.sep_token, slot_map[candidate])
                        else: # handle atis
                            sub_intents = sorted(intent.split("#"))
                            sub_intents_prompt = " ".join([intent_map[s] for s in sub_intents])
                            prompt = "{} {} {} {}".format(self.tokenizer.sep_token, sub_intents_prompt, 
                                                        self.tokenizer.sep_token, slot_map[candidate])

                    elif self.config.model_type == "sep+CRF":
                        prompt = "{} {}".format(self.tokenizer.sep_token, slot_map[candidate])
                    
                    elif self.config.model_type == "sep+CRF+slotfeat":
                        prompt = ""

                    else:
                        raise ValueError
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    slot_seq = get_slot_seqlabels(slots, len(tokens), specify_slot=candidate)
                    intent_types.append(self.intent_type_stoi[intent])
                    slot_types.append(self.slot_type_feature_stoi[candidate])
                    if self.config.use_crf:
                        slot_seqidxs.append([self.slot_label_stoi[s] for s in slot_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        slot_seqidxs.append([self.slot_label_stoi[s] for s in slot_seq] + [-100] * (max_token_num-len(tokens)))
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                    
            # joint model
            else:
                if self.config.model_type == "CRF+Typeprompt":
                    intent_map = intent_tags[self.config.dataset][self.config.intent_type]
                    if "#" not in intent:
                        prompt = "{} {}".format(self.tokenizer.sep_token, intent_map[intent])
                    else: # handle atis
                        sub_intents = sorted(intent.split("#"))
                        sub_intents_prompt = " ".join([intent_map[s] for s in sub_intents])
                        prompt = "{} {}".format(self.tokenizer.sep_token, sub_intents_prompt)
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                
                elif self.config.model_type == "CRF":
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                else:
                    raise ValueError
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))  
                slot_seq = get_slot_seqlabels(slots, len(tokens))
                intent_types.append(self.intent_type_stoi[intent])
                token_lens.append(token_len)
                token_nums.append(token_num)
                if self.config.use_crf:
                    slot_seqidxs.append([self.slot_label_stoi[s] for s in slot_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    slot_seqidxs.append([self.slot_label_stoi[s] for s in slot_seq] + [-100] * (max_token_num-len(tokens)))
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        intent_types = torch.LongTensor(intent_types)
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        intent_types = intent_types.cuda()
        slot_types = torch.cuda.LongTensor(slot_types)
        slot_seqidxs = torch.cuda.LongTensor(slot_seqidxs)
        return enc_idxs, enc_attn, slot_seqidxs, intent_types, slot_types, token_lens, torch.cuda.LongTensor(token_nums)
    
    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]
        if self.multi_piece == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            bert_outputs = torch.gather(bert_outputs, 1, offsets)
        elif self.multi_piece == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
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
        entity_label_scores = self.slot_label_ffn(bert_outputs)
        if self.config.use_crf:
            entity_label_scores_ = self.slot_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.slot_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.slot_label_stoi,
                                            self.slot_type_stoi)
            else:
                entity_label_loglik = self.slot_crf.loglik(entity_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= entity_label_loglik.mean()
                
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds,
                                            token_nums,
                                            self.slot_label_stoi,
                                            self.slot_type_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.slot_label_num), target.view(-1))

        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, slot_seqidxs, intent_types, slot_types, token_lens, token_nums = self.process_data(batch)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(intent_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)

        if self.config.use_slot_feature:
            slot_feature = self.slot_feature_module(slot_types)
            extend_slot_vec = slot_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_slot_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, slot_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, intent_types, slot_types, token_lens, token_nums = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(intent_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            if self.config.use_slot_feature:
                slot_feature = self.slot_feature_module(slot_types)
                extend_slot_vec = slot_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_slot_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("sep"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx, (intent, slot) in enumerate(zip(batch.intents, batch.slots)):
                    valid_slots = sorted(patterns[self.dataset][intent])
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_slots:
                        new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]]) 
                        new_sub_input_texts.append(input_texts[cnt])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == enc_idxs.size(0)
                return new_arguments, new_input_texts
        self.train()
        return arguments, input_texts
