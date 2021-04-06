
import torch


class GreedySearch():

    def __init__(self, bos_id, eos_id, batch_size, max_len_list=None, device=None):
        self.hypotheses = [[] for _ in range(batch_size)]
        self.alive_seq = torch.full([batch_size, 1], bos_id, dtype=torch.long, device=device)
        self.is_finished = torch.zeros([batch_size], dtype=torch.uint8, device=device)
        self._batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)

        assert len(max_len_list) == batch_size
        self.max_len_th = torch.tensor(max_len_list, dtype=torch.long, device=device)

        self.eos_id = eos_id
        self.batch_size = batch_size
        self.device = device

        self.selected_indices = None
        self.done = False

    def search_one_step(self, token_score):
        topk_scores, topk_ids = token_score.max(dim=-1)
        self.alive_seq = torch.cat(
            [self.alive_seq,
            topk_ids.view(-1, 1)],
            dim=1)

        len_exceed = self.max_len_th <= (self.alive_seq.size(-1) - 1)
        self.is_finished = topk_ids.eq(self.eos_id) | len_exceed

    def update_finished(self):
        finished = self.is_finished.nonzero().view(-1)
        for i in finished:
            b = self._batch_offset[i]
            self.hypotheses[b].append((None, self.alive_seq[i, 1:]))

        self.done = self.is_finished.all()
        if self.done:
            return

        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        self.selected_indices = is_alive.nonzero().view(-1)
        self._batch_offset = self._batch_offset[is_alive]
        self.max_len_th = self.max_len_th[is_alive]

    def get_final_results(self):
        return self.hypotheses
