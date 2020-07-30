import re


def first_position_of_b_on_a(list_a, list_b):
    ''' We will remove from list_a the sequence that appears on list_b  '''
    seqa = list_a.copy()
    first_element = list_b[0]

    i = 0
    while i < len(seqa):
        item = seqa[i]  # takes item from list_a
        # if the item is the same as the first item of b check to see if the rest matches
        j = i + 1
        if item == first_element:
            k = 1  # go over the list_b
            while j < len(seqa) and seqa[j] == list_b[k]:
                k += 1
                j += 1
                if k == len(list_b):
                    return i
                    break

        i = j

    return None


def entities_tags_from_target_ids(model, target_ids):
    words2tag = {v: k for k, v in model.labels2words.items()}
    ents2ids = {
        k: model.tokenizer.encode(k) for k in model.entities2tokens.keys()
    }

    target_string = model.tokenizer.decode(target_ids)

    ent_pos = []
    for key in model.entities2tokens:
        k = f'\[{key[1:-1]}\]'
        ent_pos += [(key, m.start()) for m in re.finditer(k, target_string)]
    ent_pos = sorted(ent_pos, key=lambda x: x[1])

    start = 0
    ent_tags = []
    for ent, _ in ent_pos:
        ent_ids = ents2ids[ent]

        position = first_position_of_b_on_a(target_ids[start:], ent_ids)

        position += start

        n = position - start

        tag = words2tag[ent]

        if tag != 'O':
            btag = f'B-{tag}'
            itag = f'I-{tag}'
        else:
            btag = itag = tag

        ent_tags += [btag] + (n - 1) * [itag]

        start = position + len(ent_ids)

    return ent_tags


def clean_ids(tensor_ids):
    t = tensor_ids.cpu().numpy().tolist()
    try:
        idx = t.index(1) + 1
    except ValueError:
        pass
    else:
        t = t[:idx]
    return t


def truncate_or_pad(_list, n, fill_token='O'):
    _len = len(_list)
    if _len == n:
        return _list
    elif _len < n:
        m = n - _len
        _list = _list + m * [fill_token]
    elif _len > n:
        _list = _list[:n]
    return _list
