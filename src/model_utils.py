import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import ipdb

def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask

def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets

def token_lens_to_offsets_masked(token_lens, token_masks):
    offsets = []
    for seq_token_lens, seq_token_masks in zip(token_lens, token_masks):
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        assert len(seq_offsets) == len(seq_token_masks)
        seq_offsets = [so for so, m in zip(seq_offsets, seq_token_masks) if m == 1]
        offsets.append(seq_offsets)
    max_token_num = max([len(x) for x in offsets])
    offsets = [x+[-1]*(max_token_num - len(x)) for x in offsets]
    return offsets

def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

def token_lens_to_idxs_masked(token_lens, token_masks):
    max_token_len = max([max(x) for x in token_lens])
    max_token_num = max([sum(x) for x in token_masks])
    idxs, ratio_masks = [], []
    for seq_token_lens, seq_token_masks in zip(token_lens, token_masks):
        seq_idxs, seq_ratio_masks = [], []
        offset = 0
        assert len(seq_token_lens) == len(seq_token_masks)
        for token_len, m in zip(seq_token_lens, seq_token_masks):
            if m == 1:
                seq_idxs.extend([i + offset for i in range(token_len)]
                                + [-1] * (max_token_len - token_len))
                seq_ratio_masks.extend([1.0 / token_len] * token_len
                                + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - sum(seq_token_masks)))
        seq_ratio_masks.extend([0.0] * max_token_len * (max_token_num - sum(seq_token_masks)))
        idxs.append(seq_idxs)
        ratio_masks.append(seq_ratio_masks)    
    return idxs, ratio_masks, max_token_num, max_token_len

def tag_paths_to_spans(paths, token_nums, vocab, type_vocab):
    """
    Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)
            #tag = type_vocab[tag]
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
        batch_mentions.append(mentions)
    return batch_mentions

def get_role_seqlabels(roles, token_num, specify_role=None, use_unified_label=True):
    labels = ['O'] * token_num
    count = 0
    for role in roles:
        start, end = role[1][0], role[1][1]
        if end > token_num:
            continue
        role_type = role[1][2]

        if specify_role is not None:
            if role_type != specify_role:
                continue
        
        if any([labels[i] != 'O' for i in range(start, end)]):
            count += 1
            continue
        
        if (specify_role is not None) and use_unified_label:
            labels[start] = 'B-{}'.format("Pred")
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format("Pred")
        else:
            labels[start] = 'B-{}'.format(role_type)
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format(role_type)
    # if count:
    #     print('cannot cover {} entities due to span overlapping'.format(count))
    #     ipdb.set_trace()
    return labels

def get_relation_seqlabels(relations, token_num, specify_relation=None):
    labels = ['O'] * token_num
    count = 0
    for relation in relations:
        start, end = relation[1][0], relation[1][1]
        if end > token_num:
            continue
        relation_type = relation[2]

        if specify_relation is not None:
            if relation_type != specify_relation:
                continue
        
        if any([labels[i] != 'O' for i in range(start, end)]):
            count += 1
            continue
        
        if specify_relation is not None:
            labels[start] = 'B-{}'.format("Pred")
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format("Pred")
        else:
            labels[start] = 'B-{}'.format(relation_type)
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format(relation_type)
    # if count:
    #     print('cannot cover {} entities due to span overlapping'.format(count))
    #     ipdb.set_trace()
    return labels    

def get_relation_seqlabels_stage1(subject, relation_map, token_num):
    labels = ['O'] * token_num
    count = 0
    key = (subject['start'], subject['end'])
    if key in relation_map.keys():
        rels = relation_map[key]
        for rel in rels:
            start, end = rel[2], rel[3]
            if end > token_num:
                continue
            rel_type = rel[-1]
           
            if any([labels[i] != 'O' for i in range(start, end)]):
                count += 1
                continue
            
            labels[start] = 'B-{}'.format(rel_type)
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format(rel_type)
    # if count:
    #     print('cannot cover {} entities due to span overlapping'.format(count))
    #     ipdb.set_trace()
    return labels

def get_trigger_seqlabels(triggers, token_num, specify_role=None):
    labels = ['O'] * token_num
    count = 0
    for trigger in triggers:
        start, end = trigger[0], trigger[1]
        if end > token_num:
            continue
        trigger_type = trigger[2]

        if specify_role is not None:
            if trigger_type != specify_role:
                continue
        
        if any([labels[i] != 'O' for i in range(start, end)]):
            count += 1
            continue
        
        if specify_role is not None:
            labels[start] = 'B-{}'.format("Pred")
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format("Pred")
        else:
            labels[start] = 'B-{}'.format(trigger_type)
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format(trigger_type)
    # if count:
    #     print('cannot cover {} entities due to span overlapping'.format(count))
    return labels

def get_slot_seqlabels(slots, token_num, specify_slot=None):
    return get_trigger_seqlabels(slots, token_num, specify_role=specify_slot)

def get_entity_seqlabels(entities, token_num, specify_slot=None):
    return get_trigger_seqlabels(entities, token_num, specify_role=specify_slot)

def get_trigger_embedding(bert_outputs, triggers):
    masks = []
    max_tokens = bert_outputs.size(1)
    for trigger in triggers:
        seq_masks = [0] * max_tokens
        for element in range(trigger[0], trigger[1]):
            seq_masks[element] = 1
        masks.append(seq_masks)
    masks = bert_outputs.new(masks)
    average = ((bert_outputs*masks.unsqueeze(-1))/((masks.sum(dim=1,keepdim=True)).unsqueeze(-1))).sum(1)
    
    return average # batch x bert_dim

def get_entity_embedding(bert_outputs, entities):
    return get_trigger_embedding(bert_outputs, entities)

def random_sample_roles(valid_roles, positive_type, ns_ratio, output_num):
    unnormal_prob = []
    for role in valid_roles:
        if role in positive_type:
            unnormal_prob.append(1.0)
        elif role == 'O':
            unnormal_prob.append(0.0)
        else:
            unnormal_prob.append(ns_ratio)
    # normlize prob
    sum_prob = sum(unnormal_prob)
    normal_prob = [p/sum_prob for p in unnormal_prob]
    return np.random.choice(valid_roles, output_num, replace=False, p=normal_prob)

def filter_use_ner(argument_predictions,  entities):
    final_arguments = []
    for argument, entity in zip(argument_predictions, entities):
        final_argument = []
        entity_spans = [(ent['start'], ent['end']) for ent in entity]
        for arg in argument:
            if (arg[0], arg[1]) in entity_spans:
                final_argument.append(arg)
        final_arguments.append(final_argument)
    return final_arguments

def get_type_embedding_from_bert(bert_model, tokenizer, type_names):
    embs = []
    for tyn in type_names:
        if tyn != 'O':
            prompt = tokenizer(tyn, return_tensors='pt')
        else:
            prompt = tokenizer(tokenizer.unk_token, return_tensors='pt')
        emb = bert_model(**prompt)[0]
        # take average of the subtokens
        emb = torch.mean(emb[0, 1:-1, :], dim=0)
        embs.append(emb)
    return torch.stack(embs, dim=0)

class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        # self.same_type = self.map_same_types()
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens):
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores

class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
            outputs.append(inputs)
        return outputs[-1]