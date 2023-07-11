import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig
import copy
import ipdb

from pattern import *
from model_utils import (token_lens_to_offsets, token_lens_to_idxs, tag_paths_to_spans, get_role_seqlabels, get_slot_seqlabels, get_trigger_embedding, CRF, Linears)

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class myBertModel(BertModel):
    def __init__(self, config, trigger_type_num=0, role_type_num=0, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.trigger_type_num = trigger_type_num
        self.role_type_num = role_type_num
        if trigger_type_num > 0:
            self.type_feature_module = nn.Embedding(self.trigger_type_num, config.hidden_size)
            self.type_feature_module.weight.data.normal_(0, 0.05)
        if role_type_num > 0:
            self.role_feature_module = nn.Embedding(self.role_type_num, config.hidden_size)
            self.role_feature_module.weight.data.normal_(0, 0.05)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        trigger_type_ids=None,
        role_type_ids=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if (self.trigger_type_num > 0) and (trigger_type_ids is not None):
            trigger_type_feat =  self.type_feature_module(trigger_type_ids)
            trigger_type_feat = trigger_type_feat.unsqueeze(1).repeat(1, embedding_output.size(1), 1)
            # embedding_output = self.type_feature_transform(torch.cat([embedding_output, trigger_type_feat], dim=-1))
            embedding_output = embedding_output + trigger_type_feat

        if (self.role_type_num > 0) and (role_type_ids is not None):
            role_type_feat =  self.role_feature_module(role_type_ids)
            role_type_feat = role_type_feat.unsqueeze(1).repeat(1, embedding_output.size(1), 1)
            # embedding_output = self.type_feature_transform(torch.cat([embedding_output, role_type_feat], dim=-1))
            embedding_output = embedding_output + role_type_feat

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class EAEfeatModel(nn.Module):
    def __init__(self, config, tokenizer, vocabs, dataset):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = dataset
        # vocabularies
        self.vocabs = vocabs
        self.trigger_type_stoi = self.vocabs['trigger_type']
        self.trigger_type_num = len(self.trigger_type_stoi)

        if not self.config.model_type.startswith("feat+sep"):
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

        if self.config.use_type_feature:
            used_trigger_type_num = self.trigger_type_num
        else:
            used_trigger_type_num = 0

        self.role_type_feature_stoi = vocabs['role_type']
        if self.config.use_role_feature:
            assert self.config.model_type.startswith("feat+sep")
            used_role_type_num = len(self.role_type_feature_stoi)
        else:
            used_role_type_num = 0

        # BERT encoder
        self.bert_model_name = config.bert_model_name
        self.bert_cache_dir = config.bert_cache_dir
        if self.bert_model_name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.bert = myBertModel.from_pretrained(self.bert_model_name,
                                                  cache_dir=self.bert_cache_dir,
                                                  output_hidden_states=True,
                                                  trigger_type_num=used_trigger_type_num, 
                                                  role_type_num=used_role_type_num
                                                  )
            self.bert_config = BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        # elif self.bert_model_name.startswith('roberta-'):
        #     self.bert = RobertaModel.from_pretrained(self.bert_model_name,
        #                                           cache_dir=self.bert_cache_dir,
        #                                           output_hidden_states=True)
        #     self.bert_config = RobertaConfig.from_pretrained(self.bert_model_name,
        #                                          cache_dir=self.bert_cache_dir)
        # elif self.bert_model_name.startswith('xlm-'):
        #     self.bert = XLMRobertaModel.from_pretrained(self.bert_model_name,
        #                                           cache_dir=self.bert_cache_dir,
        #                                           output_hidden_states=True)
        #     self.bert_config = XLMRobertaConfig.from_pretrained(self.bert_model_name,
        #                                          cache_dir=self.bert_cache_dir)
        
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
            if self.config.model_type.startswith("feat+sep"):
                valid_roles = sorted(patterns[self.dataset][trigger[2]])
                for candidate in valid_roles:
                    if self.config.model_type == "feat+sep+CRF+Triprompt+Typeprompt":
                        evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                        role_map = role_tags[self.dataset][self.config.role_type]
                        prompt = "{} {} {} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                    self.tokenizer.sep_token, trigger[3],
                                                    self.tokenizer.sep_token, role_map[candidate])

                    elif self.config.model_type == "feat+sep+CRF+Triprompt":
                        role_map = role_tags[self.dataset][self.config.role_type]
                        prompt = "{} {} {} {}".format(self.tokenizer.sep_token, trigger[3],
                                                    self.tokenizer.sep_token, role_map[candidate])

                    elif self.config.model_type == "feat+sep+CRF":
                        role_map = role_tags[self.dataset][self.config.role_type]
                        prompt = "{} {}".format(self.tokenizer.sep_token, role_map[candidate])
                    elif self.config.model_type == "feat+sep+CRF+rolefeat":
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
                if self.config.model_type == "feat+CRF+Triprompt+Typeprompt":
                    evetype_map = evetype_tags[self.config.dataset][self.config.trigger_type]
                    prompt = "{} {} {} {}".format(self.tokenizer.sep_token, evetype_map[trigger[2]], 
                                                self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                
                elif self.config.model_type == "feat+CRF+Triprompt+Typeprompt+Rolehint":
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

                elif self.config.model_type == "feat+CRF+Triprompt":
                    prompt = "{} {}".format(self.tokenizer.sep_token, trigger[3])
                    prompt_id = self.tokenizer.encode(prompt, add_special_tokens=False)
                    piece_id = self.tokenizer.convert_tokens_to_ids(piece)
                    enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + \
                                piece_id + \
                                prompt_id + \
                                [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                elif self.config.model_type == "feat+CRF":
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
    
    def encode(self, piece_idxs, attention_masks, token_lens, trigger_types, role_types):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks, 
                                    trigger_type_ids=trigger_types,
                                    role_type_ids=role_types)
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
        bert_outputs = self.encode(enc_idxs, enc_attn, token_lens, trigger_types, role_types)
        if self.config.use_trigger_feature:
            # get trigger embedding
            trigger_vec = get_trigger_embedding(bert_outputs, triggers)
            extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
            bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
        span_id_loss, _ = self.span_id(bert_outputs, token_nums, role_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch, use_ner_filter=False):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, trigger_types, role_types, token_lens, token_nums, triggers = self.process_data(batch)
            # encoding
            bert_outputs = self.encode(enc_idxs, enc_attn, token_lens, trigger_types, role_types)
            if self.config.use_trigger_feature:
                # get trigger embedding
                trigger_vec = get_trigger_embedding(bert_outputs, triggers)
                extend_tri_vec = trigger_vec.unsqueeze(1).repeat(1, bert_outputs.size(1), 1)
                bert_outputs = torch.cat((bert_outputs, extend_tri_vec), dim=-1)
            _, arguments = self.span_id(bert_outputs, token_nums, predict=True)
            input_texts = [self.tokenizer.decode(enc_idx, skip_special_tokens=True) for enc_idx in enc_idxs]
            if self.config.model_type.startswith("feat+sep"):
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