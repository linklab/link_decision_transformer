# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from g_utils import to_var


class PointerNetwork(nn.Module):
    def __init__(
            self, embed_input_size, embed_size, weight_size, answer_seq_len,
            hidden_size=512, is_single_value_data=True, is_GRU=True, decoder_input_always_zero=True
    ):
        super(PointerNetwork, self).__init__()

        self.answer_seq_len = answer_seq_len    # 4
        self.weight_size = weight_size          # 256
        self.embed_size = embed_size            # 32
        self.hidden_size = hidden_size          # 512
        self.is_GRU = is_GRU
        self.decoder_input_always_zero = decoder_input_always_zero

        if is_single_value_data:
            # embed_input_size: 4, embed_size: 32
            self.embed = nn.Embedding(embed_input_size, embed_size)     # embed inputs
        else:
            # TSP --> embed_input_size: 2, embed_size: 32
            self.embed = nn.Linear(embed_input_size, embed_size)        # embed inputs

        if is_GRU:
            # GRUCell's input is always batch first
            self.enc = nn.GRU(input_size=embed_size, hidden_size=hidden_size, batch_first=True)   # num_layers=1
            self.dec = nn.GRUCell(input_size=embed_size, hidden_size=hidden_size)
        else:
            # LSTMCell's input is always batch first
            self.enc = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)  # num_layers=1
            self.dec = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)

        self.W1 = nn.Linear(in_features=hidden_size, out_features=weight_size, bias=False)       # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)       # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False)      # scaling sum of enc and dec by v.T

    def forward(self, input):
        # input.size(): (250, 4)
        batch_size = input.size(0)
        input_seq_len = input.size(1)

        input = self.embed(input)  # (bs, L, embd_size) = (250, 4, 32)
        # input.size(): (250, 4, 32)

        # Encoding
        encoder_states, hc = self.enc(input)                            # encoder_state: (bs, L, H)
        # encoder_states.size(): (250, 4, 512)
        encoder_states = encoder_states.transpose(1, 0)                 # (L, bs, H) = (4, 250, 512)
        #print(encoder_states.size(), hc.size(), "!!!!!!!!!")
        assert torch.equal(encoder_states[-1], hc.squeeze()), (encoder_states.size(), hc.size())

        # Decoding states initialization
        hidden = encoder_states[-1]                                     # (bs, h) = (250, 512)
        cell_state = to_var(torch.zeros([batch_size, self.hidden_size]))  # (bs, h) = (250, 512) for LSTM

        probs = []

        if self.decoder_input_always_zero:
            decoder_input = to_var(torch.zeros(batch_size, self.embed_size))  # (bs, embd_size) = (250, 32)
        else:
            decoder_input = to_var(input[:, 0, :])                            # (bs, embd_size) = (250, 32)

        mask = torch.ones([batch_size, input_seq_len]).detach()

        # Decoding
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
            out = out.transpose(0, 1).contiguous()      # (bs, L) = (250, 4)
            out = out * mask
            out = F.log_softmax(out, dim=-1)  # (bs, L) = (250, 4)
            probs.append(out)

            if self.decoder_input_always_zero:
                decoder_input = to_var(torch.zeros(batch_size, self.embed_size))  # (bs, embd_size) = (250, 32)
            else:
                _, indices = torch.max(out, dim=-1)  # len(indices) = bs

                # new_mask = mask.clone()
                # for i in range(new_mask.shape[0]):
                #     new_mask[i, indices[i]] = -10_000
                #
                # mask = new_mask

                mask = mask.scatter(dim=-1, index=indices.unsqueeze(-1), value=float('-inf'))

                # input: (bs, L, embed_size) --> transposed_tensor: (L, bs, embed_size)
                sliced_tensor = input[:, indices, :].clone()  # (bs, bs, embed_size)
                decoder_input = sliced_tensor.view(-1, batch_size, self.embed_size)[:, 0, :]  # (bs, embed_size)

        probs = torch.stack(probs, dim=1)           # (bs, M, L) = (250, 4, 4)

        return probs
