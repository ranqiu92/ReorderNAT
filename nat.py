
import torch
import torch.nn as nn

from transformer import *
from util import sequence_mask, mean, init_weights


class NATBase(nn.Module):

    def __init__(self, hidden_size, src_emb_conf, tgt_emb_conf=None, dropout=0.1):
        super(NATBase, self).__init__()
        if tgt_emb_conf is None:
            self.embedding = self._init_embedding(src_emb_conf)
            self.src_embedding = self.embedding
            self.tgt_embedding = self.embedding
        else:
            self.src_embedding = self._init_embedding(src_emb_conf)
            self.tgt_embedding = self._init_embedding(tgt_emb_conf)
        self.pos_embedding = PositionEmbedding(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def _init_embedding(self, emb_conf):
        vocab_size = emb_conf['vocab_size']
        emb_size = emb_conf['emb_size']
        padding_idx = emb_conf.get('padding_idx', None)
        return nn.Embedding(vocab_size, emb_size, padding_idx)

    def add_pos_emb(self, emb, step=None):
        bsz, length, _ = emb.size()
        pos_emb = self.pos_embedding(length=length, step=step).to(emb.device)
        pos_emb = pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        emb = emb + pos_emb
        return emb

    def embed(self, input, embedding, step=None):
        bsz, length = input.size()
        emb = embedding(input) * self.hidden_size ** 0.5
        pos_emb = self.pos_embedding(length, step=step).to(input.device)
        pos_emb = pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        emb = self.dropout(emb + pos_emb)
        return emb

    def encode(self, src, src_lens):
        src_emb = self.embed(src, self.src_embedding)
        src_enc, src_mask = self.encoder(src_emb, src_lens)
        return src_enc, src_mask

    # uniform copy
    def get_copied_src(self, src_seq, src_lens, tgt_lens):
        bsz = src_seq.size(0)
        tgt_mask = ~sequence_mask(tgt_lens).to(src_seq.device)

        tgt_max_len, _ = torch.max(tgt_lens.view(-1), dim=0)
        pos = torch.arange(0., tgt_max_len).unsqueeze(0).expand(bsz, -1)
        pos = pos.to(src_seq.device)

        src_lens = src_lens.float().to(src_seq.device) - 1
        tgt_lens = tgt_lens.float().to(src_seq.device) - 1
        tgt_lens = torch.clamp(tgt_lens, min=1)
        pos = pos / tgt_lens.unsqueeze(-1) * src_lens.unsqueeze(-1)
        pos = pos.round().long().masked_fill(tgt_mask, 0)

        copied_src_seq = torch.gather(src_seq, 1, pos)
        copied_src_seq = copied_src_seq.masked_fill(tgt_mask, self.src_embedding.padding_idx)
        copied_src = self.embed(copied_src_seq, self.src_embedding)

        return copied_src, tgt_mask

    def predict_length(self, src_enc, src_lens, is_training=True):
        src_mean = mean(src_enc, src_lens.to(src_enc.device))
        logit = self.len_pred_fc(src_mean)

        if is_training:
            return logit

        max_len_dif = (logit.size(-1) - 1) // 2
        len_dif_mask = sequence_mask(
                torch.clamp(max_len_dif - src_lens, min=1),
                logit.size(-1))
        len_dif_mask = len_dif_mask.to(src_enc.device)

        masked_len_logit = logit.masked_fill(len_dif_mask, float('-inf'))
        _, pred_tgt_lens = masked_len_logit.max(dim=1)
        pred_tgt_lens = pred_tgt_lens - max_len_dif + src_lens.to(src_enc.device)

        return logit, pred_tgt_lens


class NAT(NATBase):

    def __init__(self, enc_layers, dec_layers, hidden_size, head_num, ffn_size, \
                 src_emb_conf, tgt_emb_conf=None, max_len_dif=50, dropout=0.1, \
                 use_label_smoothing=True, smooth_rate=0.15):
        super(NAT, self).__init__(hidden_size, src_emb_conf, tgt_emb_conf, dropout)
        self.encoder = TransformerEncoder(enc_layers, hidden_size, head_num, ffn_size, dropout)
        self.decoder = TransformerDecoder(dec_layers, hidden_size, head_num, ffn_size, dropout, causal=False)
        self.len_pred_fc = nn.Linear(hidden_size, 2 * max_len_dif + 1)
        self.use_label_smoothing = use_label_smoothing
        self.smooth_rate = smooth_rate
        self.apply(init_weights)

    def forward(self, src_seq, src_lens, tgt_lens, label=None, len_label=None):
        src_max_len = max(int(src_lens.max()), 1)
        src_seq = src_seq[:, :src_max_len]
        tgt_max_len = max(int(tgt_lens.max()), 1)
        label = label[:, :tgt_max_len]

        src_enc, src_mask = self.encode(src_seq, src_lens)

        tgt_len_logit = self.predict_length(src_enc.detach(), src_lens)
        len_log_prob = F.log_softmax(tgt_len_logit, dim=-1)

        copied_src, tgt_mask = self.get_copied_src(src_seq, src_lens, tgt_lens)
        tgt_dec, w_logit = self.decode(copied_src, src_enc, src_mask, tgt_mask.unsqueeze(1))
        w_log_prob = F.log_softmax(w_logit, dim=-1).transpose(1, 2)

        w_loss, len_loss = self.compute_loss(w_log_prob, label, len_log_prob, len_label)

        return w_loss, len_loss

    def compute_loss(self, w_log_prob, label, len_log_prob, len_label):
        if self.use_label_smoothing:
            w_loss = smooth_loss(w_log_prob, label, pad=self.tgt_embedding.padding_idx, eps=self.smooth_rate)
        else:
            loss_func = nn.NLLLoss(ignore_index=self.tgt_embedding.padding_idx, reduction='sum')
            w_loss = loss_func(w_log_prob, label)

        len_loss_func = nn.NLLLoss(reduction='sum')
        len_loss = len_loss_func(len_log_prob, len_label)

        return w_loss, len_loss

    def decode(self, copied_src, src_enc, src_mask=None, tgt_mask=None):
        tgt_dec = self.decoder(copied_src, src_enc, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask, mask_self=True)
        logit = F.linear(tgt_dec, self.tgt_embedding.weight)
        return tgt_dec, logit


class ReorderNAT(NATBase):

    def __init__(self, enc_layers, dec_layers, hidden_size, head_num, ffn_size, src_emb_conf, tgt_emb_conf=None,
                 reorder_layers=1, use_ar_reorder=True, ndgd_temp=1., use_ndgd=False, max_len_dif=50, dropout=0.1, \
                 use_label_smoothing=True, smooth_rate=0.15):
        super(ReorderNAT, self).__init__(hidden_size, src_emb_conf, tgt_emb_conf, dropout)
        self.encoder = TransformerEncoder(enc_layers, hidden_size, head_num, ffn_size, dropout)
        if use_ar_reorder:
            reorder_layers = 1
            self.ar_reorder_module = RNNDecoder(hidden_size, hidden_size, head_num, dropout)
        else:
            self.nar_reorder_module = TransformerDecoder(reorder_layers, hidden_size, head_num, ffn_size, dropout, causal=False)
            self.len_pred_fc = nn.Linear(hidden_size, 2 * max_len_dif + 1)

        nat_dec_layers = dec_layers - reorder_layers
        self.decoder = TransformerDecoder(nat_dec_layers, hidden_size, head_num, ffn_size, dropout, causal=False)

        self.use_ar_reorder = use_ar_reorder
        self.use_label_smoothing = use_label_smoothing
        self.smooth_rate = smooth_rate

        assert ndgd_temp > 0
        self.ndgd_temp = ndgd_temp
        self.use_ndgd = use_ndgd
        self.apply(init_weights)

    def forward(self, src_seq, src_lens, tgt_lens, align_pos, pseudo_vocab, label, align_pos_label, len_label=None):
        src_max_len = max(int(src_lens.max()), 1)
        tgt_max_len = max(int(tgt_lens.max()), 1)

        src_seq = src_seq[:, :src_max_len]
        label = label[:, :tgt_max_len]
        align_pos = align_pos[:, :tgt_max_len + 1, :]
        if self.use_ar_reorder:
            align_pos_label = align_pos_label[:, :tgt_max_len + 1, :]
        else:
            align_pos_label = align_pos_label[:, :tgt_max_len, :]

        src_enc, src_mask = self.encode(src_seq, src_lens)

        pseudo_vocab_mat, pseudo_vocab_mask = self.get_pseudo_vocab_matrix(pseudo_vocab)
        pseudo_vocab_mat_T = torch.transpose(pseudo_vocab_mat, 1, 2)
        
        pseudo_trans_emb, pseudo_trans_mask = self.get_pseudo_trans_emb(pseudo_vocab=pseudo_vocab, align_pos=align_pos)
        pseudo_trans_emb = pseudo_trans_emb * self.hidden_size ** 0.5

        if self.use_ar_reorder:
            pos_logit = self.ar_reorder(pseudo_trans_emb, pseudo_vocab_mat_T, src_enc, \
                    pseudo_trans_mask=pseudo_trans_mask, pseudo_vocab_mask=pseudo_vocab_mask, src_mask=src_mask)
        else:   
            copied_src_for_reorder, tgt_mask = self.get_copied_src(src_seq, src_lens, tgt_lens)
            pos_logit = self.nar_reorder(pseudo_vocab_mat_T, copied_src_for_reorder, src_enc, \
                    pseudo_vocab_mask=pseudo_vocab_mask, src_mask=src_mask, tgt_mask=tgt_mask.unsqueeze(1))
        pos_log_prob = F.log_softmax(pos_logit, dim=-1).transpose(1, 2)

        if self.use_ndgd:
            weight = F.softmax(pos_logit / self.ndgd_temp, dim=-1)
            pseudo_trans_emb = torch.matmul(weight, pseudo_vocab_mat * self.hidden_size ** 0.5)
            if self.use_ar_reorder:
                pseudo_trans_emb = pseudo_trans_emb[:, :-1, :]
        else:
            pseudo_trans_emb = pseudo_trans_emb[:, 1:, :]

        tgt_mask = ~sequence_mask(tgt_lens).to(src_seq.device)
        pseudo_trans_emb = self.dropout(self.add_pos_emb(pseudo_trans_emb))
        pseudo_trans_emb = pseudo_trans_emb.masked_fill(tgt_mask.unsqueeze(-1), 0)
        tgt_dec, w_logit = self.decode(pseudo_trans_emb, src_enc, src_mask, tgt_mask.unsqueeze(1))
        w_log_prob = F.log_softmax(w_logit, dim=-1).transpose(1, 2)

        len_log_prob = None
        if not self.use_ar_reorder:
            tgt_len_logit = self.predict_length(src_enc.detach(), src_lens)
            len_log_prob = F.log_softmax(tgt_len_logit, dim=-1)

        w_loss, pos_loss, len_loss = self.compute_loss(w_log_prob, label, pos_log_prob, align_pos_label, \
                        pos_cls_mask=pseudo_vocab_mask.squeeze(1), len_log_prob=len_log_prob, len_label=len_label)

        return w_loss, pos_loss, len_loss

    def compute_loss(self, w_log_prob, label, pos_log_prob, align_pos_label, pos_cls_mask, len_log_prob=None, len_label=None):
        if self.use_label_smoothing:
            w_loss = smooth_loss(w_log_prob, label, pad=self.tgt_embedding.padding_idx, eps=self.smooth_rate)
            pos_loss = self.smooth_pos_loss(pos_log_prob, pos_cls_mask, align_pos_label.squeeze(-1), eps=self.smooth_rate)
        else:
            loss_func = nn.NLLLoss(ignore_index=self.tgt_embedding.padding_idx, reduction='sum')
            w_loss = loss_func(w_log_prob, label)

            align_pos_label = align_pos_label.transpose(1, 2)
            align_pos_mask = align_pos_label >= 0
            align_pos_num = torch.sum(align_pos_mask, dim=1).float()

            align_pos_label = torch.clamp(align_pos_label, min=0)
            selected_pos_log_prob = torch.gather(
                pos_log_prob,
                1,
                align_pos_label)

            pos_loss = -torch.sum(selected_pos_log_prob.masked_fill(1 - align_pos_mask, 0.), dim=1)
            pos_loss = torch.sum(pos_loss / torch.clamp(align_pos_num, min=1.))

        len_loss = None
        if len_log_prob is not None:
            len_loss_func = nn.NLLLoss(reduction='sum')
            len_loss = len_loss_func(len_log_prob, len_label)

        return w_loss, pos_loss, len_loss

    def smooth_pos_loss(self, log_prob, cls_mask, label, eps=0.15):
        bsz, max_nclass, _ = log_prob.size()

        mask = label < 0
        label = torch.clamp(label, min=0)

        ncls = torch.sum(1 - cls_mask, dim=-1).float().view(bsz, 1, 1).expand_as(log_prob)
        smoothed_label = torch.zeros_like(log_prob)
        smoothed_label = smoothed_label + eps / torch.clamp((ncls - 1), 1.)
        smoothed_label = smoothed_label.masked_fill(cls_mask.unsqueeze(-1), 0)

        class_range = torch.arange(0, max_nclass).to(log_prob.device)
        selected_mask = class_range.view(1, max_nclass, 1).expand_as(log_prob) == label.unsqueeze(1).expand_as(log_prob)
        smoothed_label = smoothed_label.masked_fill(selected_mask, 1 - eps)
        smoothed_label = smoothed_label.masked_fill(mask.unsqueeze(1), 0.)

        loss = torch.sum(-log_prob * smoothed_label)

        return loss

    def get_pseudo_vocab_matrix(self, pseudo_vocab):
        pseudo_vocab_mask = pseudo_vocab.data.eq(self.src_embedding.padding_idx).unsqueeze(1)
        pseudo_vocab_mat = self.src_embedding(pseudo_vocab)
        return pseudo_vocab_mat, pseudo_vocab_mask

    def get_pseudo_trans_emb(self, pseudo_vocab=None, align_pos=None, memory=None, align_weight=None):
        if align_pos is not None:
            align_pos_mask = align_pos >= 0
            align_pos_num = torch.sum(align_pos_mask, dim=-1).float()
            pseudo_trans_mask = align_pos_num <= 0

            bsz, length, num = align_pos.size()
            align_pos = torch.clamp(align_pos, min=0)
            pseudo_trans_seq = torch.gather(
                pseudo_vocab,
                1,
                align_pos.view(bsz, -1)).view(bsz, length, num)
            pseudo_trans_emb = self.src_embedding(pseudo_trans_seq)

            pseudo_trans_emb = pseudo_trans_emb.masked_fill(1 - align_pos_mask.unsqueeze(-1), 0.)
            pseudo_trans_emb = torch.sum(pseudo_trans_emb, dim=2) / torch.clamp(align_pos_num, min=1.).unsqueeze(-1)
        elif align_weight is not None:
            pseudo_trans_emb = torch.matmul(align_weight, memory)
            pseudo_trans_mask = torch.zeros(align_weight.size()[:2], dtype=torch.uint8, device=memory.device)
        else:
            assert False

        return pseudo_trans_emb, pseudo_trans_mask

    def decode(self, pseudo_trans_emb, src_enc, src_mask=None, tgt_mask=None):
        tgt_dec = self.decoder(pseudo_trans_emb, src_enc, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask, mask_self=True)
        logit = F.linear(tgt_dec, self.tgt_embedding.weight)
        return tgt_dec, logit

    def nar_reorder(self, pseudo_vocab_mat_T, copied_src, src_enc, pseudo_vocab_mask=None, src_mask=None, tgt_mask=None):
        dec = self.nar_reorder_module(copied_src, src_enc, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask, mask_self=True)
        logit = torch.matmul(dec, pseudo_vocab_mat_T)

        if pseudo_vocab_mask is not None:
            logit = logit.masked_fill(pseudo_vocab_mask, -1e18)

        return logit

    def ar_reorder(self, pseudo_trans_emb, pseudo_vocab_mat_T, src_enc, pseudo_trans_mask=None, pseudo_vocab_mask=None, src_mask=None, step=None):
        pseudo_trans_emb = self.dropout(pseudo_trans_emb)

        if pseudo_trans_mask is not None:
            pseudo_trans_emb = pseudo_trans_emb.masked_fill(pseudo_trans_mask.unsqueeze(-1), 0.)
            pseudo_trans_mask = pseudo_trans_mask.unsqueeze(1)

        dec = self.ar_reorder_module(pseudo_trans_emb, src_enc, src_pad_mask=src_mask, tgt_pad_mask=pseudo_trans_mask, step=step)
        logit = torch.matmul(dec, pseudo_vocab_mat_T)

        if pseudo_vocab_mask is not None:
            logit = logit.masked_fill(pseudo_vocab_mask, -1e18)

        return logit
