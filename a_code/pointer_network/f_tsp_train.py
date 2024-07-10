import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import a_generate_data
from a_code.pointer_network.b_generate_tsp_data import TSPDataset
from g_utils import to_var
from c_pointer_network import PointerNetwork


# from pointer_network import PointerNetwork
def train(model, dataloader, test_dataloader, batch_size, n_epochs, number_of_cities):
    model.train()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            train_batch = to_var(sample_batched['Points'])
            target_batch = to_var(sample_batched['Solution'])

            probs = model(train_batch)   # (bs, M, L) = (250, 10, 10)
            output_batch = probs.view(-1, number_of_cities)  # (bs * M, L)

            target_batch = target_batch.flatten()              # (bs * M)
            loss = F.cross_entropy(output_batch, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
            # for _ in range(2):  # random showing results
            #     pick = np.random.randint(0, batch_size)
            #     probs = probs.contiguous().view(batch_size, number_of_cities, number_of_cities).transpose(2, 1) # (bs, L, M)
            #     y = target_batch.view(batch_size, number_of_cities)
            #     print('train pred', [v.item() for v in probs.max(1)[1][pick]])
            #     print('train label', [v.item() for v in y[pick]])

            for _ in range(2):  # random showing results
                pick = np.random.randint(0, batch_size)
                _v, indices = torch.max(probs, dim=2)  # (bs, number_of_cities)
                target_batch = target_batch.view(batch_size, number_of_cities)
                print('Train pred: ', [v.item() for v in indices[pick]])
                print('Train label:', [v.item() for v in target_batch[pick]])

            validate(model, test_dataloader, is_train=True)


def validate(model, dataloader, is_train=False):
    sample_batched = next(iter(dataloader))
    test_batch = to_var(sample_batched['Points'])
    target_batch = to_var(sample_batched['Solution'])

    probs = model(test_batch) # (bs, M, L)

    # max values, max indices
    _v, indices = torch.max(probs, dim=2) # (bs, M)

    ####################################################
    #show validate examples
    if not is_train:
        for i in range(len(indices)):
            print('-----')
            print('Validate Data: ', [v.tolist() for v in test_batch[i]])
            print('Validate Pred: ', [v.item() for v in indices[i]])
            print('Validate Label:', [v.item() for v in target_batch[i]])
            if torch.equal(target_batch[i], indices[i]):
                print('SAME')
            if i > 20: break
    ############################################################

    correct_count = sum([1 if torch.equal(ind, y) else 0 for ind, y in zip(indices, target_batch)])
    print('Validate Accuracy: {:.2f}% ({}/{})'.format(correct_count/len(test_batch)*100, correct_count, len(test_batch)))
    print()


def main():
    data_size = 4
    test_data_size = 100

    weight_size = 256
    embed_size = 128
    batch_size = 2
    n_epochs = 100

    number_of_cities = 5

    dataset = TSPDataset(data_size=data_size, seq_len=number_of_cities)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TSPDataset(data_size=test_data_size, seq_len=number_of_cities)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = PointerNetwork(
        embed_input_size=2,
        embed_size=embed_size,
        weight_size=weight_size,
        answer_seq_len=number_of_cities,
        is_single_value_data=False,
        decoder_input_always_zero=False
    )
    if torch.cuda.is_available():
        model.cuda()

    train(model, dataloader, test_dataloader, batch_size, n_epochs, number_of_cities)

    print('----Validate Result---')
    validate(model, test_dataloader)


if __name__ == '__main__':
    main()
