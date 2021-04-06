
import copy
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable


def load_vocab(file_path):
    vocab = OrderedDict()

    special_symbols = ['<pad>', '<bos>', '<eos>', '<unk>', '<mask>']
    for i, symb in enumerate(special_symbols):
        vocab[symb] = i

    idx = len(vocab)
    with open(file_path, encoding='utf8') as f:
        for line in f:
            w = line.strip()
            if w in vocab.keys():
                continue
            vocab[w] = idx
            idx += 1
    return vocab


def get_reverse_dict(dictionary):
    reverse_dict = {dictionary[k] : k for k in dictionary.keys()}
    return reverse_dict


def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    return padded_seqs


def padded_align_pos_sequence(align_pos_batch):
    # if a tgt has no src token aligned to it, add a mask_id
    bos_id, eos_id, mask_id, pad = 0, 1, 2, -1

    max_pos_num = 1
    for align_pos_sample in align_pos_batch:
        cur_max_pos_num = max([len(align_pos) for align_pos in align_pos_sample])
        max_pos_num = max(max_pos_num, cur_max_pos_num)

    local_padded_align_pos_batch = []
    local_padded_align_pos_label_batch = []
    for align_pos_sample in align_pos_batch:
        align_pos_sample = [align_pos if len(align_pos) > 0 else [mask_id] for align_pos in align_pos_sample]

        align_pos_input = [[bos_id]] + align_pos_sample
        align_pos_label = align_pos_sample + [[eos_id]]

        padded_align_pos = [align_pos + [pad] * (max_pos_num - len(align_pos)) for align_pos in align_pos_input]
        padded_align_pos_label = [align_pos + [pad] * (max_pos_num - len(align_pos)) for align_pos in align_pos_label]

        local_padded_align_pos_batch.append(padded_align_pos)
        local_padded_align_pos_label_batch.append(padded_align_pos_label)

    padded_align_pos_batch = padded_sequence(local_padded_align_pos_batch, [pad] * max_pos_num)
    padded_align_pos_label_batch = padded_sequence(local_padded_align_pos_label_batch, [pad] * max_pos_num)

    return padded_align_pos_batch, padded_align_pos_label_batch


def convert_to_tensor(batch, src_vocab, tgt_vocab=None, for_reordering=False, device=None, is_training=False):
    src_pad = src_vocab['<pad>']

    src_seq = [sample['src_tokens'] for sample in batch]
    src_lens = [len(seq) for seq in src_seq]
    src_lens = torch.LongTensor(src_lens)
    padded_src_seq = padded_sequence(src_seq, src_pad)

    if for_reordering:
        pseduo_vocab = [sample['pseduo_vocab'] for sample in batch]
        padded_pseduo_vocab = padded_sequence(pseduo_vocab, src_pad)
        pseduo_vocab = Variable(torch.LongTensor(padded_pseduo_vocab), requires_grad=False)
        if device:
            pseduo_vocab = pseduo_vocab.to(device)

    if is_training:
        tgt_pad = tgt_vocab['<pad>']
        tgt_bos = tgt_vocab['<bos>']
        tgt_eos = tgt_vocab['<eos>']

        tgt_seq = [sample['tgt_tokens'] for sample in batch]
        tgt_lens = [len(seq) for seq in tgt_seq]
        tgt_lens = torch.LongTensor(tgt_lens)

        label = [seq + [tgt_eos] for seq in tgt_seq]
        input_tgt = [[tgt_bos] + seq for seq in tgt_seq]

        padded_tgt_seq = padded_sequence(input_tgt, tgt_pad)
        padded_label = padded_sequence(label, tgt_pad)

        padded_batch = [padded_src_seq, padded_tgt_seq, padded_label]
        tensor_batch = [Variable(torch.LongTensor(item), requires_grad=False) for item in padded_batch]
        if device:
            tensor_batch = [item.to(device) for item in tensor_batch]
        src_seq, tgt_seq, label = tensor_batch

        if for_reordering:
            align_pos_batch = [sample['align_info'] for sample in batch]
            padded_align_pos_batch, padded_align_pos_label_batch = padded_align_pos_sequence(align_pos_batch)
            align_pos_tensor = Variable(torch.LongTensor(padded_align_pos_batch), requires_grad=False)
            align_pos_label_tensor = Variable(torch.LongTensor(padded_align_pos_label_batch), requires_grad=False)
            if device:
                align_pos_tensor = align_pos_tensor.to(device)
                align_pos_label_tensor = align_pos_label_tensor.to(device)
            return src_seq, src_lens, tgt_seq, tgt_lens, label, align_pos_tensor, align_pos_label_tensor, pseduo_vocab
        else:
            return src_seq, src_lens, tgt_seq, tgt_lens, label
    else:
        src_seq = Variable(torch.LongTensor(padded_src_seq))
        if device:
            src_seq = src_seq.to(device)

        if for_reordering:
            return src_seq, src_lens, pseduo_vocab
        else:
            return src_seq, src_lens


def convert_word_to_id(sent, vocab):
    unk = vocab['<unk>']
    w_list = [w for w in sent.strip().split() if w]
    tokens = [vocab.get(w, unk) for w in w_list]
    return tokens


def process_alignment(align_file):
    max_align = 1
    all_alignment = []
    with open(align_file, encoding='utf8') as f:
        for line in f:
            line_split = line.strip().split()
            tgt_pos_list = [int(piece.split('-')[1]) for piece in line_split]
            if len(tgt_pos_list) < 1:
                all_alignment.append([[]])
                continue

            max_tgt_pos = max(tgt_pos_list)

            align = [[] for i in range(max_tgt_pos + 1)]
            for piece in line_split:
                src_pos, tgt_pos = [int(pos) for pos in piece.split('-')]
                align[tgt_pos].append(src_pos)

            cur_max= max([len(pos_list) for pos_list in align])
            max_align = max(max_align, cur_max)

            all_alignment.append(align)

    return all_alignment, max_align


def load_data(src_file, src_vocab, tgt_file=None, tgt_vocab=None, align_file=None):
    with open(src_file, encoding='utf8') as f:
        src_sent_list = f.readlines()

    if tgt_file:
        with open(tgt_file, encoding='utf8') as f:
            tgt_sent_list = f.readlines()
        assert len(src_sent_list) == len(tgt_sent_list)

    if align_file:
        alignments, _ = process_alignment(align_file)
        assert len(src_sent_list) == len(alignments), "%d v.s. %d" % (len(src_sent_list), len(alignments))

    for i, src_sent in enumerate(src_sent_list):
        sample = {
            'src_tokens': None,
            'pseduo_vocab': None,
            'tgt_tokens': None,
            'align_info': None
        }

        src_tokens = convert_word_to_id(src_sent, src_vocab)
        unique_src = sorted(np.unique(src_tokens).tolist())
        pseduo_vocab = [src_vocab['<bos>'], src_vocab['<eos>'], src_vocab['<mask>']] + unique_src
        sample['src_tokens'] = src_tokens
        sample['pseduo_vocab'] = pseduo_vocab

        if tgt_file:
            tgt_sent = tgt_sent_list[i]
            tgt_tokens = convert_word_to_id(tgt_sent, tgt_vocab)
            sample['tgt_tokens'] = tgt_tokens

            if align_file:
                tgt_len = len(tgt_tokens)
                cur_align = alignments[i]
                cur_align = cur_align + [[] for k in range(tgt_len - len(cur_align))]

                token_pos_dict = {token: pos for pos, token in enumerate(pseduo_vocab)}

                new_align = []
                for token_align in cur_align:
                    new_token_align = [token_pos_dict[src_tokens[pos]] for pos in token_align]
                    new_align.append(new_token_align)
                sample['align_info'] = new_align

        yield sample


def parallel_data_len(sample):
    src_len = len(sample['src_tokens']) if sample['src_tokens'] else 0
    tgt_len = len(sample['tgt_tokens']) if sample['tgt_tokens'] else 0
    return max(src_len, tgt_len)


def cluster_fn(data, bucket_size, len_fn):
    def cluster(id):
        return len_fn(data[id]) // bucket_size
    return cluster


def token_number_batcher(data, max_token_num, len_fn, bucket_size=3):
    sample_ids = list(range(len(data)))
    np.random.shuffle(sample_ids)
    sample_ids = sorted(sample_ids, key=cluster_fn(data, bucket_size, len_fn))

    total_len = 0
    sample_lens = []
    batch, batch_list = [], []
    for sample_id in sample_ids:
        batch.append(sample_id)
        length = len_fn(data[sample_id])
        sample_lens.append(length)
        total_len += length

        if total_len >= max_token_num:
            batch_list.append(batch)
            total_len = 0
            sample_lens = []
            batch = []

    if batch:
        batch_list.append(batch)

    np.random.shuffle(batch_list)
    for batch in batch_list:
        yield [data[id] for id in batch]
