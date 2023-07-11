from tensorboardX import SummaryWriter
import random
import ipdb

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_eae_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    # entity_type_set = set()
    trigger_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        # entity_type_set.update(dataset.entity_type_set)
        trigger_type_set.update(dataset.event_type_set)
        role_type_set.update(dataset.role_type_set)

    prefix = ['B', 'I']
    # entity_label_stoi = {'O': 0}
    # for t in sorted(entity_type_set):
    #     for p in prefix:
    #         entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)

    # entity_type_stoi = {k: i for i, k in enumerate(sorted(entity_type_set), 1)}
    # entity_type_stoi['O'] = 0

    trigger_label_stoi = {'O': 0}
    for t in sorted(trigger_type_set):
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    trigger_type_stoi = {k: i for i, k in enumerate(sorted(trigger_type_set), 1)}
    trigger_type_stoi['O'] = 0

    role_label_stoi = {'O': 0}
    for t in sorted(role_type_set):
        for p in prefix:
            role_label_stoi['{}-{}'.format(p, t)] = len(role_label_stoi)

    role_type_stoi = {k: i for i, k in enumerate(sorted(role_type_set), 1)}
    role_type_stoi['O'] = 0

    unified_label_stoi = {'O': 0, 'B-Pred': 1, 'I-Pred':2}
    unified_type_stoi = {'O':0, 'Pred': 1}

    return {
        # 'entity_type': entity_type_stoi,
        # 'entity_label': entity_label_stoi,
        'trigger_type': trigger_type_stoi,
        'trigger_label': trigger_label_stoi,
        'role_type': role_type_stoi,
        'role_label': role_label_stoi,
        'unified_type': unified_type_stoi,
        'unified_label': unified_label_stoi
    }

def generate_ed_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    trigger_type_set = set()
    for dataset in datasets:
        trigger_type_set.update(dataset.event_type_set)

    prefix = ['B', 'I']

    trigger_label_stoi = {'O': 0}
    for t in sorted(trigger_type_set):
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    trigger_type_stoi = {k: i for i, k in enumerate(sorted(trigger_type_set), 1)}
    trigger_type_stoi['O'] = 0

    entity_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)

    prefix = ['B', 'I']

    entity_label_stoi = {'O': 0}
    for t in sorted(entity_type_set):
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(sorted(entity_type_set), 1)}
    entity_type_stoi['O'] = 0

    unified_label_stoi = {'O': 0, 'B-Pred': 1, 'I-Pred':2}
    unified_type_stoi = {'O':0, 'Pred': 1}

    return {
        'entity_type': entity_type_stoi,
        'entity_label': entity_label_stoi,
        'trigger_type': trigger_type_stoi,
        'trigger_label': trigger_label_stoi,
        'unified_type': unified_type_stoi,
        'unified_label': unified_label_stoi
    }

def generate_ner_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)

    prefix = ['B', 'I']

    entity_label_stoi = {'O': 0}
    for t in sorted(entity_type_set):
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(sorted(entity_type_set), 1)}
    entity_type_stoi['O'] = 0

    unified_label_stoi = {'O': 0, 'B-Pred': 1, 'I-Pred':2}
    unified_type_stoi = {'O':0, 'Pred': 1}

    return {
        'entity_type': entity_type_stoi,
        'entity_label': entity_label_stoi,
        'unified_type': unified_type_stoi,
        'unified_label': unified_label_stoi
    }

def generate_re_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    relation_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)
        relation_type_set.update(dataset.relation_type_set)

    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    for t in sorted(entity_type_set):
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(sorted(entity_type_set), 1)}
    entity_type_stoi['O'] = 0

    relation_label_stoi = {'O': 0}
    for t in sorted(relation_type_set):
        for p in prefix:
            relation_label_stoi['{}-{}'.format(p, t)] = len(relation_label_stoi)

    relation_type_stoi = {k: i for i, k in enumerate(sorted(relation_type_set), 1)}
    relation_type_stoi['O'] = 0

    unified_label_stoi = {'O': 0, 'B-Pred': 1, 'I-Pred':2}
    unified_type_stoi = {'O':0, 'Pred': 1}

    return {
        'entity_type': entity_type_stoi,
        'entity_label': entity_label_stoi,
        'relation_type': relation_type_stoi,
        'relation_label': relation_label_stoi,
        'unified_type': unified_type_stoi,
        'unified_label': unified_label_stoi
    }

def generate_sp_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    intent_type_set = set()
    slot_type_set = set()
    for dataset in datasets:
        intent_type_set.update(dataset.intent_type_set)
        slot_type_set.update(dataset.slot_type_set)
        

    prefix = ['B', 'I']
    intent_label_stoi = {'O': 0}
    for t in sorted(intent_type_set):
        for p in prefix:
            intent_label_stoi['{}-{}'.format(p, t)] = len(intent_label_stoi)

    intent_type_stoi = {k: i for i, k in enumerate(sorted(intent_type_set), 1)}
    intent_type_stoi['O'] = 0
    
    slot_label_stoi = {'O': 0}
    for t in sorted(slot_type_set):
        for p in prefix:
            slot_label_stoi['{}-{}'.format(p, t)] = len(slot_label_stoi)

    slot_type_stoi = {k: i for i, k in enumerate(sorted(slot_type_set), 1)}
    slot_type_stoi['O'] = 0

    unified_label_stoi = {'O': 0, 'B-Pred': 1, 'I-Pred':2}
    unified_type_stoi = {'O':0, 'Pred': 1}

    return {
        'intent_type': intent_type_stoi,
        'intent_label': intent_label_stoi,
        'slot_type': slot_type_stoi,
        'slot_label': slot_label_stoi,
        'unified_type': unified_type_stoi,
        'unified_label': unified_label_stoi
    }