import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import a_generate_data
from g_utils import to_var
from c_pointer_network import PointerNetwork


# from pointer_network import PointerNetwork
def train(model, X, Y, batch_size, n_epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
    N = X.size(0)  # N = bs = 9000
    L = X.size(1)  # L = 4

    for epoch in range(n_epochs):
        for i in range(0, N - batch_size, batch_size):
            train_batch = X[i: i + batch_size]  # (bs, L) = (bs, 4)
            target_batch = Y[i: i + batch_size]  # (bs, M) = (bs, 4)

            probs = model(train_batch)             # (bs, M, L)
            output_batch = probs.view(-1, L)  # (bs * M, L)
            target_batch = target_batch.flatten()              # (bs * M)
            loss = F.nll_loss(output_batch, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            for _ in range(2):  # random showing results
                pick = np.random.randint(0, batch_size)
                _v, indices = torch.max(probs, dim=2)
                target_batch = target_batch.view(batch_size, L)
                print('Train pred: ', [v.item() for v in indices[pick]])
                print('Train label:', [v.item() for v in target_batch[pick]])
            validate(model, X, Y, is_train=True)


def validate(model, X, Y, is_train=False):
    probs = model(X) # probs.size(): (9000, 4, 4)

    # max values, max indices
    _v, indices = torch.max(probs, dim=2)   # (bs, M)

    ####################################################
    # show validate examples
    if not is_train:
        for i in range(len(indices)):
            print('-----')
            print('Validate Data', [v.item() for v in X[i]])
            print('Validate Pred', [v.item() for v in indices[i]])
            print('Validate Label', [v.item() for v in Y[i]])
            if torch.equal(Y[i], indices[i]):
                print('SAME')
            if i > 20: break
    #############################################################

    correct_count = sum([1 if torch.equal(ind, y) else 0 for ind, y in zip(indices, Y)])
    print('Validate Accuracy: {:.2f}% ({}/{})'.format(correct_count / len(X) * 100, correct_count, len(X)))
    print()


def main():
    total_size = 10000
    weight_size = 256
    embed_size = 32
    batch_size = 250
    n_epochs = 5
    input_seq_len = 20

    input, targets = a_generate_data.make_seq_data(total_size, input_seq_len)
    embed_inp_size = input_seq_len

    # Convert to torch tensors
    input = to_var(torch.LongTensor(input))  # (N, L)
    targets = to_var(torch.LongTensor(targets))  # (N, L)

    data_split = (int)(total_size * 0.9)
    train_X = input[:data_split]    # 9000, 4
    train_Y = targets[:data_split]  # 9000, 4

    test_X = input[data_split:]     # 1000, 4
    test_Y = targets[data_split:]   # 1000, 4

    # inp_size: 4, embed_size: 32, weight_size: 256, input_seq_len: 4
    model = PointerNetwork(
        embed_input_size=embed_inp_size,
        embed_size=embed_size,
        weight_size=weight_size,
        answer_seq_len=input_seq_len
    )

    if torch.cuda.is_available():
        model.cuda()
    train(model, train_X, train_Y, batch_size, n_epochs)
    print('----Validate Result---')
    validate(model, test_X, test_Y)


if __name__ == '__main__':
    main()
