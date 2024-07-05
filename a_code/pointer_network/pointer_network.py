# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var


class PointerNetwork(nn.Module):
    def __init__(self, input_size, emb_size, weight_size, answer_seq_len, hidden_size=512, is_GRU=True):
        super(PointerNetwork, self).__init__()

        self.input_size = input_size            # 4
        self.answer_seq_len = answer_seq_len    # 4
        self.weight_size = weight_size          # 256
        self.emb_size = emb_size                # 32
        self.hidden_size = hidden_size          # 512
        self.is_GRU = is_GRU

        # input_size: 4, emb_size: 32
        self.emb = nn.Embedding(input_size, emb_size)  # embed inputs
        if is_GRU:
            # GRUCell's input is always batch first
            self.enc = nn.GRU(input_size=emb_size, hidden_size=hidden_size, batch_first=True)   # num_layers=1
            self.dec = nn.GRUCell(input_size=emb_size, hidden_size=hidden_size)
        else:
            # LSTMCell's input is always batch first
            self.enc = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True)  # num_layers=1
            self.dec = nn.LSTMCell(input_size=emb_size, hidden_size=hidden_size)

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)       # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)       # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)      # scaling sum of enc and dec by v.T

    def forward(self, input):
        # input.size(): (250, 4)
        batch_size = input.size(0)
        input = self.emb(input) # (bs, L, embd_size)
        # input.size(): (250, 4, 32)

        # Encoding
        encoder_states, hc = self.enc(input)                            # encoder_state: (bs, L, H)
        # encoder_states.size(): (250, 4, 512)
        encoder_states = encoder_states.transpose(1, 0)                 # (L, bs, H) = (4, 250, 512)

        # Decoding states initialization
        decoder_input = to_var(torch.zeros(batch_size, self.emb_size))  # (bs, embd_size) = (250, 32)
        hidden = to_var(torch.zeros([batch_size, self.hidden_size]))    # (bs, h) = (250, 512)
        cell_state = encoder_states[-1]                                 # (bs, h) = (250, 512)

        probs = []

        # Decoding
        hidden = cell_state     # BUGGGGGG!!!!!! 이 라인이 필요함!!!!
        for i in range(self.answer_seq_len):                                        # range(4)
            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden)                            # (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state))  # (bs, h), (bs, h)
            # hidden.size(): (250, 512)

            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)            # (L, bs, W)
            blend2 = self.W2(hidden)                    # (bs, W)
            blend_sum = F.tanh(blend1 + blend2)         # (L, bs, W)
            # blend_sum.size(): (4, 250, 256)

            # self.vt(blend_sum).size(): (4, 250, 1)
            # self.vt(blend_sum).squeeze().size(): (4, 250)
            out = self.vt(blend_sum).squeeze()          # (L, bs)

            out = F.log_softmax(out.transpose(0, 1).contiguous(), dim=-1) # (bs, L) = (250, 4)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           # (bs, M, L) = (250, 4, 4)

        return probs
