
import time
import itertools
import logging

import torch
from torch.autograd import Variable

from beamsearch import BeamSearch
from greedysearch import GreedySearch
from util import sequence_mask
from data import convert_to_tensor, get_reverse_dict
from nat import NATBase, ReorderNAT


logger = logging.getLogger(__name__)

MAX_LENGTH_DIF = 50

class Translator():

    def __init__(self, model, src_vocab, tgt_vocab, batch_size, beam_size=1, \
                 use_lpd=False, delta_len=0, rescore_model=None, device=None):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.reverse_tgt_vocab = get_reverse_dict(tgt_vocab)
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.use_lpd = use_lpd
        self.delta_len = abs(delta_len)
        self.rescore_model = rescore_model
        if self.use_lpd:
            assert self.rescore_model is not None
        self.device = device

    def translate(self, data):
        result = {'prediction': [], 'length': []}

        all_preds = []
        start_time = time.time()
        for pos in range(0, len(data), self.batch_size):
            batch = list(itertools.islice(data, pos, pos + self.batch_size))

            if isinstance(self.model, ReorderNAT):
                predictions = self.reorder_nat_translate_batch(batch)
            elif isinstance(self.model, NATBase):
                predictions = self.nonautoregressive_translate_batch(batch)
            else:
                if self.beam_size == 1:
                    predictions = self.greedy_translate_batch(batch)
                else:
                    predictions = self.translate_batch(batch)
            all_preds.append(predictions)

        end_time = time.time()
        logger.info('Total decoding time: %f' % (end_time - start_time))

        for pred in all_preds:
            pred_tgt_seq, length = pred['prediction'], pred['length']
            if length is not None:
                length = length.tolist()
                prediction = [pred_seq[:pred_len] for (pred_len, pred_seq) \
                                    in zip(length, pred_tgt_seq.tolist())]
            else:
                prediction = [n_best[0][1].tolist() for n_best in pred_tgt_seq]

            result['prediction'].extend(prediction)

        def _to_sentence(seq):
            raw_sentence = [self.reverse_tgt_vocab[id] for id in seq if id != self.tgt_vocab['<eos>']]
            sentence = " ".join(raw_sentence)
            return sentence

        result['prediction'] = [_to_sentence(pred) for pred in result['prediction']]
        return result

    def nonautoregressive_translate_batch(self, batch):
        self.model.eval()

        batch_tensor = convert_to_tensor(batch, self.src_vocab, device=self.device)
        src_seq, src_lens = batch_tensor[:2]

        src_enc, src_mask = self.model.encode(src_seq, src_lens)
        tgt_len_logit, pred_tgt_lens = self.model.predict_length(src_enc, src_lens, is_training=False)

        copied_src, tgt_mask = self.model.get_copied_src(src_seq, src_lens, pred_tgt_lens)
        tgt_mask = tgt_mask.unsqueeze(1)

        _, w_logit = self.model.decode(copied_src, src_enc, src_mask, tgt_mask)
        _, pred_tgt_seq = w_logit.max(dim=-1)

        results = {'prediction': pred_tgt_seq, 'length': pred_tgt_lens}
        return results

    def reorder_nat_translate_batch(self, batch):
        self.model.eval()

        use_ar_reorder = getattr(self.model, 'use_ar_reorder', True)
        batch_tensor = convert_to_tensor(batch, self.src_vocab, for_reordering=True, device=self.device)
        src_seq, src_lens = batch_tensor[:2]
        pseudo_vocab = batch_tensor[-1]
        batch_size = src_seq.size(0)

        src_enc, src_mask = self.model.encode(src_seq, src_lens)

        pseudo_vocab_mat, pseudo_vocab_mask = self.model.get_pseudo_vocab_matrix(pseudo_vocab)
        pseudo_vocab_mat_T = torch.transpose(pseudo_vocab_mat, 1, 2)

        if not use_ar_reorder:
            tgt_len_logit, pred_tgt_lens = self.model.predict_length(src_enc, src_lens, is_training=False)

            if self.use_lpd:
                range_len = 2 * self.delta_len + 1
                len_range = torch.arange(0, range_len) - self.delta_len

                pred_tgt_lens = pred_tgt_lens.unsqueeze(-1) + len_range.unsqueeze(0).to(pred_tgt_lens.device)
                pred_tgt_lens = torch.clamp(pred_tgt_lens, 1).view(-1)

                bsz, src_max_len, hidden_size = src_enc.size()
                src_lens = src_lens.unsqueeze(1).expand(bsz, range_len).contiguous().view(-1)
                src_seq = src_seq.unsqueeze(1).expand(bsz, range_len, -1).contiguous().view(bsz * range_len, src_max_len)
                src_enc = src_enc.unsqueeze(1).expand(bsz, range_len, -1, hidden_size).contiguous().view(bsz * range_len, src_max_len, hidden_size)
                src_mask = src_mask.unsqueeze(1).expand(bsz, range_len, 1, src_max_len).contiguous().view(bsz * range_len, 1, src_max_len)

                pseudo_vocab = pseudo_vocab.unsqueeze(1).expand(bsz, range_len, -1).contiguous().view(bsz * range_len, -1)
                pseudo_vocab_mask = pseudo_vocab_mask.unsqueeze(1).expand(bsz, range_len, 1, -1).contiguous().view(bsz * range_len, 1, -1)
                pseudo_vocab_mat = pseudo_vocab_mat.unsqueeze(1).expand(bsz, range_len, -1, hidden_size).contiguous().view(bsz * range_len, -1, hidden_size)
                pseudo_vocab_mat_T = pseudo_vocab_mat_T.unsqueeze(1).expand(bsz, range_len, hidden_size, -1).contiguous().view(bsz * range_len, hidden_size, -1)

            copied_src_for_reorder, tgt_mask = self.model.get_copied_src(src_seq, src_lens, pred_tgt_lens)
            pos_logit = self.model.nar_reorder(pseudo_vocab_mat_T, copied_src_for_reorder, src_enc, pseudo_vocab_mask=pseudo_vocab_mask, \
                                               src_mask=src_mask, tgt_mask=tgt_mask.unsqueeze(1))

            if self.model.use_ndgd:
                weight = (pos_logit / self.model.ndgd_temp).softmax(dim=-1)
                pseudo_trans_emb, _ = self.model.get_pseudo_trans_emb(memory=pseudo_vocab_mat, align_weight=weight)
            else:
                _, align= pos_logit.max(dim=-1)
                align = align.masked_fill(tgt_mask, -1).unsqueeze(-1)
                pseudo_trans_emb, _ = self.model.get_pseudo_trans_emb(pseudo_vocab=pseudo_vocab, align_pos=align)

            pseudo_trans_emb = pseudo_trans_emb * self.model.hidden_size ** 0.5
        else:
            max_len_list = [MAX_LENGTH_DIF + length for length in src_lens.tolist()]
            searcher = GreedySearch(
                bos_id=0,
                eos_id=1,
                batch_size=batch_size,
                max_len_list=max_len_list,
                device=self.device)

            memory_bank, memory_mask = src_enc.clone(), src_mask.clone()

            scaled_pseudo_vocab_mat = pseudo_vocab_mat * self.model.hidden_size ** 0.5
            scaled_pseudo_vocab_mat_cache = scaled_pseudo_vocab_mat.clone()

            unfinished = torch.arange(batch_size, dtype=torch.long, device=self.device)
            input_list = [[] for i in range(batch_size)]
            logit_list = [[] for i in range(batch_size)]

            prev_token = searcher.alive_seq[:, -1]
            pseudo_token_emb = torch.gather(scaled_pseudo_vocab_mat_cache,
                                       1,
                                       prev_token.view(-1, 1, 1).expand(-1, 1, scaled_pseudo_vocab_mat_cache.size(-1)))
            pseudo_token_mask = torch.zeros(pseudo_token_emb.size()[:2], dtype=torch.uint8, device=self.device)

            max_len = max(max_len_list)
            for step in range(max_len):
                logit = self.model.ar_reorder(pseudo_token_emb, pseudo_vocab_mat_T, memory_bank, \
                                              pseudo_trans_mask=pseudo_token_mask, pseudo_vocab_mask=pseudo_vocab_mask, \
                                              src_mask=memory_mask, step=step)
                searcher.search_one_step(logit.squeeze(1))

                prev_token = searcher.alive_seq[:, -1].view(-1, 1, 1)
                pseudo_token_emb = torch.gather(scaled_pseudo_vocab_mat_cache,
                                            1,
                                            prev_token.view(-1, 1, 1).expand(-1, 1, scaled_pseudo_vocab_mat_cache.size(-1)))

                for i in range(unfinished.size(0)):
                    ind = unfinished[i]
                    input_list[ind].append(pseudo_token_emb[i, :, :])
                    logit_list[ind].append(logit[i, :, :])

                if searcher.is_finished.any():
                    searcher.update_finished()
                    if searcher.done:
                        break

                    select_indices = searcher.selected_indices
                    unfinished = unfinished.index_select(0, select_indices)
                    memory_bank = memory_bank.index_select(0, select_indices)
                    memory_mask = memory_mask.index_select(0, select_indices)

                    pseudo_vocab_mask = pseudo_vocab_mask.index_select(0, select_indices)
                    pseudo_token_emb = pseudo_token_emb.index_select(0, select_indices)
                    pseudo_token_mask = pseudo_token_mask.index_select(0, select_indices)
                    pseudo_vocab_mat_T = pseudo_vocab_mat_T.index_select(0, select_indices)
                    scaled_pseudo_vocab_mat_cache = scaled_pseudo_vocab_mat_cache.index_select(0, select_indices)

                    self.model.ar_reorder_module.map_state(
                        lambda state, dim: state.index_select(dim, select_indices))

            pos_prediction = searcher.get_final_results()
            pos_prediction = [pair[0][1] for pair in pos_prediction]

            def clip_eos(predictions, vec_list=None):
                pos_eos = 1
                cleaned_vec_list = None
                if vec_list is not None:
                    for pred, vec in zip(predictions, vec_list):
                        assert len(pred) == len(vec)
                    cleaned_vec_list = [vec_list[i] if predictions[i][-1] != pos_eos else vec_list[i][:-1] for i in range(len(vec_list))]
                cleaned_preds = [pred if pred[-1] != pos_eos else pred[:-1] for pred in predictions]
                return cleaned_preds, cleaned_vec_list

            if self.model.use_ndgd:
                _, logit_list = clip_eos(pos_prediction, logit_list)

                pred_tgt_lens = [len(logit_vecs) for logit_vecs in logit_list]
                max_tgt_len = max(pred_tgt_lens)

                logit_list = [torch.cat(logit_vecs) if logit_vecs else None for logit_vecs in logit_list]
                logit_mat = torch.zeros((batch_size, max_tgt_len, pseudo_vocab_mat.size(1)), dtype=torch.float, device=self.device)
                for i, logit in enumerate(logit_list):
                    if logit is not None:
                        logit_mat[i, 0:logit.size(0), :] = logit

                weight = (logit_mat / self.model.ndgd_temp).softmax(dim=-1)
                pseudo_trans_emb = torch.matmul(weight, scaled_pseudo_vocab_mat)
            else:
                _, input_list = clip_eos(pos_prediction, input_list)

                pred_tgt_lens = [len(inp) for inp in input_list]
                max_tgt_len = max(pred_tgt_lens)

                input_list = [torch.cat(inp, dim=0) if inp else None for inp in input_list]
                pseudo_trans_emb = torch.zeros((batch_size, max_tgt_len, pseudo_vocab_mat.size(-1)), dtype=torch.float, device=self.device)
                for i, inp in enumerate(input_list):
                    if inp is not None:
                        pseudo_trans_emb[i, 0:inp.size(0) ,:] = inp

            pred_tgt_lens = torch.LongTensor(pred_tgt_lens)
            tgt_mask = ~sequence_mask(pred_tgt_lens).to(self.device)

        pseudo_trans_emb = self.model.dropout(self.model.add_pos_emb(pseudo_trans_emb))
        pseudo_trans_emb = pseudo_trans_emb.masked_fill(tgt_mask.unsqueeze(-1), 0.)

        _, w_logit = self.model.decode(pseudo_trans_emb, src_enc, src_mask, tgt_mask.unsqueeze(1))
        _, pred_tgt_seq = w_logit.max(dim=-1)

        if not use_ar_reorder and self.use_lpd:
            def preprocess(tgt_seq, mask, bos, eos, pad):
                bsz, length = tgt_seq.size()
                bos_tensor = torch.tensor([bos], dtype=torch.long, device=tgt_seq.device)
                eos_tensor = torch.tensor([eos], dtype=torch.long, device=tgt_seq.device)

                tgt_seq_in = torch.cat((bos_tensor.view(1, 1).expand(bsz, 1), tgt_seq), 1)

                zero_mask = torch.zeros((bsz, 1), dtype=torch.uint8, device=mask.device)
                new_mask = torch.cat((mask, zero_mask), dim=1)

                one_mask = torch.ones((bsz, 1), dtype=torch.uint8, device=mask.device)
                valid_mask = torch.cat((one_mask, mask), dim=1)

                tgt_seq_label = torch.cat((tgt_seq, eos_tensor.view(1, 1).expand(bsz, 1)), 1)
                tgt_seq_label = tgt_seq_label.masked_fill(1 - new_mask, 0) + (1 - new_mask.long()) * eos
                tgt_seq_label = tgt_seq_label.masked_fill(1 - valid_mask, pad)

                return tgt_seq_in, tgt_seq_label

            tgt_seq_in, tgt_seq_label = preprocess(pred_tgt_seq, mask=1 - tgt_mask, bos=self.tgt_vocab['<bos>'], \
                                                   eos=self.tgt_vocab['<eos>'], pad=self.tgt_vocab['<pad>'])

            self.rescore_model.eval()
            nll = self.rescore_model(src_seq, tgt_seq_in, src_lens, tgt_seq_label, scoring=True)
            nll = torch.sum(nll, dim=-1)

            _, best_seq_idx = nll.view(bsz, range_len).min(dim=-1)
            pred_tgt_lens = torch.gather(pred_tgt_lens.view(bsz, range_len), 1, best_seq_idx.view(bsz, 1))
            pred_tgt_lens = pred_tgt_lens.view(-1)

            pred_tgt_seq = torch.gather(
                                pred_tgt_seq.view(bsz, range_len, -1),
                                1,
                                best_seq_idx.view(bsz, 1, 1).expand(bsz, 1, pred_tgt_seq.size(-1)))
            pred_tgt_seq = pred_tgt_seq.squeeze(1)

        results = {'prediction': pred_tgt_seq, 'length': pred_tgt_lens}
        return results

    def greedy_translate_batch(self, batch):
        self.model.eval()

        batch_tensor = convert_to_tensor(batch, self.src_vocab, device=self.device)
        src_seq, src_lens = batch_tensor[:2]
        batch_size = src_seq.size(0)

        memory_bank, memory_mask = self.model.encode(src_seq, src_lens)

        max_len_list = [MAX_LENGTH_DIF + length for length in src_lens.tolist()]
        searcher = GreedySearch(
            bos_id=self.tgt_vocab['<bos>'],
            eos_id=self.tgt_vocab['<eos>'],
            batch_size=batch_size,
            max_len_list=max_len_list,
            device=self.device)

        max_len = max(max_len_list)
        for step in range(max_len):
            dec_input = searcher.alive_seq[:, -1].view(-1, 1)
            _, logit = self.model.decode(dec_input, memory_bank, memory_mask, step=step)
            searcher.search_one_step(logit.squeeze(1))

            if searcher.is_finished.any():
                searcher.update_finished()
                if searcher.done:
                    break

                select_indices = searcher.selected_indices
                memory_bank = memory_bank.index_select(0, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)

                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        predictions = searcher.get_final_results()
        return {'prediction': predictions, 'length': None}

    def translate_batch(self, batch):
        self.model.eval()

        batch_tensor = convert_to_tensor(batch, self.src_vocab, device=self.device)
        src_seq, src_lens = batch_tensor[:2]
        batch_size = src_seq.size(0)

        memory_bank, memory_mask = self.model.encode(src_seq, src_lens)
        memory_bank = memory_bank.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2), memory_bank.size(3))
        memory_mask = memory_mask.unsqueeze(1).expand(-1, self.beam_size, -1, -1)
        memory_mask = memory_mask.contiguous().view(-1, memory_mask.size(2), memory_mask.size(3))

        max_len_list = [MAX_LENGTH_DIF + length for length in src_lens.tolist()]
        searcher = BeamSearch(
            bos_id=self.tgt_vocab['<bos>'],
            eos_id=self.tgt_vocab['<eos>'],
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_len_list=max_len_list,
            device=self.device)

        max_len = max(max_len_list)
        for step in range(max_len):
            dec_input = searcher.alive_seq[:, -1].view(-1, 1)
            _, logit = self.model.decode(dec_input, memory_bank, memory_mask, step=step)
            log_prob = logit.squeeze(1).log_softmax(dim=-1)

            searcher.search_one_step(log_prob)
            any_beam_is_finished = searcher.is_finished.any()
            if any_beam_is_finished:
                searcher.update_finished()
                if searcher.done:
                    break

            select_indices = searcher.selected_indices

            if any_beam_is_finished:
                memory_bank = memory_bank.index_select(0, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        predictions = searcher.get_final_results()
        return {'prediction': predictions, 'length': None}
