import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import a_generate_data
from e_utils import to_var
from b_pointer_network import PointerNetwork


# from pointer_network import PointerNetwork
def train(model, X, Y, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
    N = X.size(0)   # N = bs = 9000
    L = X.size(1)   # L = 30

    for epoch in range(n_epochs):
        # for i in range(len(train_batches))
        for i in range(0, N - batch_size, batch_size):
            x = X[i: i + batch_size]  # (bs, L) = (250, 30)
            y = Y[i: i + batch_size]  # (bs, M) = (250, 2)

            probs = model(x)             # (bs, M, L) = (250, 2, 30)
            outputs = probs.view(-1, L)  # (bs * M, L)
            y = y.flatten()             # (bs * M)
            loss = F.nll_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            # for _ in range(2): # random showing results
            #     pick = np.random.randint(0, batch_size)
            #     probs = probs.contiguous().view(batch_size, M, L).transpose(2, 1) # (bs, L, M)
            #     y = y.view(batch_size, M)
            #     print("predict: ", probs.max(1)[1][pick][0], probs.max(1)[1][pick][1],
            #           "target  : ", y[pick][0], y[pick][1])
            test(model, X, Y)


def test(model, X, Y):
    probs = model(X) # (bs, M, L)
    _v, indices = torch.max(probs, 2) # (bs, M)
    # show test examples
    # for i in range(len(indices)):
    #     print('-----')
    #     print('test', [v for v in X[i]])
    #     print('label', [v for v in Y[i]])
    #     print('pred', [v for v in indices[i]])
    #     if torch.equal(Y[i], indices[i]):
    #         print('eq')
    #     if i>20: break
    correct_count = sum([1 if torch.equal(ind, y) else 0 for ind, y in zip(indices, Y)])
    print('Acc: {:.2f}% ({}/{})'.format(correct_count/len(X)*100, correct_count, len(X)))


def main():
    total_size = 10000
    weight_size = 256
    emb_size = 32
    batch_size = 250
    answer_seq_len = 2
    n_epochs = 5

    dataset, starts, ends = a_generate_data.generate_set_seq(total_size)
    targets = np.vstack((starts, ends)).T   # [total_size, 2]
    dataset = np.array(dataset)             # [total_size, 30]

    input_seq_len = dataset.shape[1]        # 30
    inp_size = 11  # 0 to 10

    # Convert to torch tensors
    input = to_var(torch.LongTensor(dataset))       # (total_size, 30)
    targets = to_var(torch.LongTensor(targets))     # (total_size, 2)

    data_split = (int)(total_size * 0.9)
    train_X = input[:data_split]        # (9000, 30)
    train_Y = targets[:data_split]      # (9000, 2)
    test_X = input[data_split:]         # (1000, 30)
    test_Y = targets[data_split:]       # (1000, 2)

    model = PointerNetwork(inp_size, emb_size, weight_size, answer_seq_len)
    if torch.cuda.is_available():
        model.cuda()
    train(model, train_X, train_Y, batch_size, n_epochs)
    print('----Test result---')
    test(model, test_X, test_Y)


if __name__ == '__main__':
    main()
