import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, RobertaModel, XLMRobertaModel, AutoModel, BertConfig, RobertaConfig, XLMRobertaConfig

from pattern import evetype_tags, role_tags, intent_tags, patterns
from model_utils import (token_lens_to_offsets, token_lens_to_idxs, tag_paths_to_spans, get_role_seqlabels, get_trigger_seqlabels, get_entity_seqlabels, get_trigger_embedding, random_sample_roles, CRF,  Linears)

class EDModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.trigger_type_stoi = self.vocabs['trigger_type']
        self.trigger_type_num = len(self.trigger_type_stoi)

        if self.config.model_type.startswith("sep"):
            self.role_label_stoi = vocabs['unified_label']
            self.role_type_stoi = vocabs['unified_type']
        else:
            self.role_label_stoi = vocabs['trigger_label']
            self.role_type_stoi = vocabs['trigger_type']

        self.role_label_itos = {i:s for s, i in self.role_label_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        
        self.role_label_num = len(self.role_label_stoi)
        self.role_type_num = len(self.role_type_stoi)

        self.entity_label_stoi = vocabs['entity_label']
        self.entity_type_stoi = vocabs['entity_type']
        self.entity_label_itos = {i:s for s, i in self.entity_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi )
        self.entity_type_num = len(self.entity_type_stoi)

        # BERT encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
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

        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            if not self.config.model_type.startswith("sep"):
                raise ValueError
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.trigger_type_num, 100)

        self.role_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.role_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.role_crf = CRF(self.role_label_stoi, bioes=False)
        
        if self.config.use_ner:
            feature_dim = self.bert_dim
            self.entity_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.entity_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
            if self.config.use_crf:
                self.entity_crf = CRF(self.entity_label_stoi, bioes=False)

    def process_data(self, batch, sample=False):
        # base on the model type, we can design different data format here
        enc_idxs = []
        enc_attn = []
        role_seqidxs = []
        trigger_types = []
        token_lens = []
        token_nums = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        ent_seqidxs = []
        
        for tokens, piece, trigger, entity, token_len, token_num in zip(batch.tokens, batch.pieces, batch.triggers, batch.entities, batch.token_lens, batch.token_nums):
            if self.config.model_type.startswith("sep"):
                valid_roles = sorted(list(self.trigger_type_stoi.keys()))
                positive_type = [tri[2] for tri in trigger]
                if sample:
                    valid_roles = random_sample_roles(valid_roles, positive_type, self.config.negative_sample_ratio, self.config.sample_output_num)
                for candidate in valid_roles:
                    if candidate == 'O':
                        continue
                    prompt = "{} {}".format(self.tokenizer.sep_token, evetype_tags[candidate])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    role_seq = get_trigger_seqlabels(trigger, len(tokens), specify_role=candidate)
                    ent_seq = get_entity_seqlabels(entity, len(tokens))
                    trigger_types.append(self.trigger_type_stoi[candidate])
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                        ent_seqidxs.append([self.entity_label_stoi[s] for s in ent_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                        ent_seqidxs.append([self.entity_label_stoi[s] for s in ent_seq] + [-100] * (max_token_num-len(tokens)))
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                
            else:
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + \
                            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))  
                role_seq = get_trigger_seqlabels(trigger, len(tokens))
                ent_seq = get_entity_seqlabels(entity, len(tokens))
                trigger_types.append(0) # This is just a random padding, but shouldn't be used
                token_lens.append(token_len)
                token_nums.append(token_num)
                if self.config.use_crf:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    ent_seqidxs.append([self.entity_label_stoi[s] for s in ent_seq] + [0] * (max_token_num-len(tokens)))
                else:
                    role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    ent_seqidxs.append([self.entity_label_stoi[s] for s in ent_seq] + [-100] * (max_token_num-len(tokens)))
        
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        trigger_types = torch.LongTensor(trigger_types)
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        trigger_types = trigger_types.cuda()
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        ent_seqidxs = torch.cuda.LongTensor(ent_seqidxs)
        return enc_idxs, enc_attn, role_seqidxs, trigger_types, token_lens, torch.cuda.LongTensor(token_nums), ent_seqidxs
    
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

    def span_id(self, bert_outputs, token_nums, target=None, ent_target=None, predict=False):
        loss = 0.0
        triggers = None
        trigger_label_scores = self.role_label_ffn(bert_outputs)
        if self.config.use_crf:
            trigger_label_scores_ = self.role_crf.pad_logits(trigger_label_scores)
            if predict:
                _, trigger_label_preds = self.role_crf.viterbi_decode(trigger_label_scores_,
                                                                        token_nums)
                triggers = tag_paths_to_spans(trigger_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else: 
                trigger_label_loglik = self.role_crf.loglik(trigger_label_scores_,
                                                            target,
                                                            token_nums)
                loss -= trigger_label_loglik.mean()
        else:
            if predict:
                trigger_label_preds = torch.argmax(trigger_label_scores, dim=-1)
                triggers = tag_paths_to_spans(trigger_label_preds,
                                            token_nums,
                                            self.role_label_stoi,
                                            self.role_type_stoi)
            else:
                loss = F.cross_entropy(trigger_label_scores.view(-1, self.role_label_num), target.view(-1))
        
        if self.config.use_ner:
            entities = None
            entity_label_scores = self.entity_label_ffn(bert_outputs)
            if self.config.use_crf:
                entity_label_scores_ = self.entity_crf.pad_logits(entity_label_scores)
                if predict: 
                    _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                    entities = tag_paths_to_spans(entity_label_preds,
                                                token_nums,
                                                self.entity_label_stoi,
                                                self.entity_type_stoi)

                else:
                    entity_label_loglik = self.entity_crf.loglik(entity_label_scores_,
                                                                ent_target,
                                                                token_nums)
                    loss -= entity_label_loglik.mean()
            else:
                if predict:
                    entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                    entities = tag_paths_to_spans(entity_label_preds,
                                                token_nums,
                                                self.entity_label_stoi,
                                                self.entity_type_stoi)

                else:
                    loss += F.cross_entropy(entity_label_scores.view(-1, self.entity_label_num), ent_target.view(-1))

            return loss, triggers, entities
        else:
            return loss, triggers

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, role_seqidxs, trigger_types, token_lens, token_nums, ent_seqidxs = self.process_data(batch, sample=True)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(trigger_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        if self.config.use_ner:
            span_id_loss, _, _ = self.span_id(bert_outputs, token_nums, role_seqidxs, ent_seqidxs, predict=False)
        else:
            span_id_loss, _ = self.span_id(bert_outputs, token_nums, role_seqidxs, ent_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch):
        self.eval()
        entities = [None] * len(batch.wnd_ids)
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, trigger_types, token_lens, token_nums, ent_seqidxs = self.process_data(batch, sample=False)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(trigger_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            if self.config.use_ner:
                _, arguments, entities = self.span_id(bert_outputs, token_nums, predict=True)
            else:
                _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            
            if self.config.model_type.startswith("sep"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx in range(len(batch.tokens)):
                    valid_roles = sorted(list(self.trigger_type_stoi.keys()))
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        if candidate == 'O':
                            continue
                        new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]])
                        new_sub_input_texts.append(input_texts[cnt])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == enc_idxs.size(0)
                return new_arguments, new_input_texts, entities
                
        self.train()
        return (arguments, entities), input_texts

class NERModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.trigger_type_stoi = self.vocabs['entity_type'] # TODO: Haven't change name of "trigger" to "entity" for this model.
        self.trigger_type_num = len(self.trigger_type_stoi)

        if self.config.model_type.startswith("sep"):
            self.role_label_stoi = vocabs['unified_label']
            self.role_type_stoi = vocabs['unified_type']
        else:
            self.role_label_stoi = vocabs['entity_label']
            self.role_type_stoi = vocabs['entity_type']

        self.role_label_itos = {i:s for s, i in self.role_label_stoi.items()}
        self.role_type_itos = {i: s for s, i in self.role_type_stoi.items()}
        
        self.role_label_num = len(self.role_label_stoi)
        self.role_type_num = len(self.role_type_stoi)

        # BERT encoder
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

        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy
        
        # local classifiers
        linear_bias = config.linear_bias
        self.dropout = nn.Dropout(p=config.linear_dropout)
        feature_dim = self.bert_dim
        
        if self.config.use_type_feature:
            if not self.config.model_type.startswith("sep"):
                raise ValueError
            feature_dim += 100
            self.type_feature_module = nn.Embedding(self.trigger_type_num, 100)

        self.role_label_ffn = Linears([feature_dim, config.linear_hidden_num,
                                    self.role_label_num],
                                    dropout_prob=config.linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        if self.config.use_crf:
            self.role_crf = CRF(self.role_label_stoi, bioes=False)

    def process_data(self, batch, sample=False):
        # base on the model type, we can design different data format here
        enc_idxs = []
        enc_attn = []
        role_seqidxs = []
        trigger_types = []
        token_lens = []
        token_nums = []
        max_token_num = max([len(tokens) for tokens in batch.tokens])
        
        for tokens, piece, trigger, token_len, token_num in zip(batch.tokens, batch.pieces, batch.entities, batch.token_lens, batch.token_nums):
            if self.config.model_type.startswith("sep"):
                valid_roles = sorted(list(self.trigger_type_stoi.keys()))
                positive_type = [tri[2] for tri in trigger]
                if sample:
                    valid_roles = random_sample_roles(valid_roles, positive_type, self.config.negative_sample_ratio, self.config.sample_output_num)
                for candidate in valid_roles:
                    if candidate == 'O':
                        continue
                    prompt = "{} {}".format(self.tokenizer.sep_token, nertype_tags[candidate])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                    enc_idxs.append(enc_idx)
                    enc_attn.append([1]*len(enc_idx))
                    role_seq = get_trigger_seqlabels(trigger, len(tokens), specify_role=candidate)
                    trigger_types.append(self.trigger_type_stoi[candidate])
                    if self.config.use_crf:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [0] * (max_token_num-len(tokens)))
                    else:
                        role_seqidxs.append([self.role_label_stoi[s] for s in role_seq] + [-100] * (max_token_num-len(tokens)))
                    token_lens.append(token_len)
                    token_nums.append(token_num)
                
            else:
                piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                            piece_id + \
                            [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))  
                role_seq = get_trigger_seqlabels(trigger, len(tokens))
                trigger_types.append(0) # This is just a random padding, but shouldn't be used
                token_lens.append(token_len)
                token_nums.append(token_num)
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
        role_seqidxs = torch.cuda.LongTensor(role_seqidxs)
        return enc_idxs, enc_attn, role_seqidxs, trigger_types, token_lens, torch.cuda.LongTensor(token_nums)
    
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
        enc_idxs, enc_attn, role_seqidxs, trigger_types, token_lens, token_nums = self.process_data(batch, sample=True)
        # encoding
        bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        if self.config.use_type_feature:
            type_feature = self.type_feature_module(trigger_types)
            extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, role_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, trigger_types, token_lens, token_nums = self.process_data(batch, sample=False)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            if self.config.use_type_feature:
                type_feature = self.type_feature_module(trigger_types)
                extend_type_vec = type_feature.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_type_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            
            if self.config.model_type.startswith("sep"):
                # decompose predicted arguments
                cnt = 0
                new_arguments = []
                new_input_texts = []
                for b_idx in range(len(batch.tokens)):
                    valid_roles = sorted(list(self.trigger_type_stoi.keys()))
                    new_sub_arguments = []
                    new_sub_input_texts = []
                    for candidate in valid_roles:
                        if candidate == 'O':
                            continue
                        new_sub_arguments.extend([[argu[0], argu[1], candidate] for argu in arguments[cnt]])
                        new_sub_input_texts.append(input_texts[cnt])
                        cnt += 1
                    new_arguments.append(new_sub_arguments)
                    new_input_texts.append(new_sub_input_texts)
                assert cnt == enc_idxs.size(0)
                return new_arguments, new_input_texts
                
        self.train()
        return arguments, input_texts