import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from a_code.pointer_network.b_generate_tsp_data import TSPDataset
from g_utils import to_var
from c_pointer_network import PointerNetwork


# from pointer_network import PointerNetwork
def train(model, dataloader, test_dataloader, batch_size, n_epochs, number_of_cities):
    model.train()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        train_epoch_loss = 0.0
        train_epoch_data_size = 0
        train_epoch_correct_count = 0
        for i_batch, sample_batched in enumerate(dataloader):
            train_data_batch = to_var(sample_batched['Points'])
            train_target_batch = to_var(sample_batched['Solution'])
            train_epoch_data_size += len(train_data_batch)

            probs = model(train_data_batch)   # (bs, M, L) = (250, 10, 10)
            train_output_batch = probs.view(-1, number_of_cities)  # (bs * M, L)

            train_target_batch_flatten = train_target_batch.flatten()              # (bs * M)
            loss = F.cross_entropy(train_output_batch, train_target_batch_flatten)
            train_epoch_loss += loss.item()

            _v, indices = torch.max(probs, dim=2)
            train_epoch_correct_count += sum(
                [1 if torch.equal(ind, y) else 0 for ind, y in zip(indices, train_target_batch)]
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if epoch % 1 == 0:
            print('Epoch: {}'.format(epoch))
            print('Train Loss: {:.5f}, Train Accuracy: {:.2f}% ({}/{})'.format(
                train_epoch_loss / train_epoch_data_size,
                train_epoch_correct_count / train_epoch_data_size * 100,
                train_epoch_correct_count,
                train_epoch_data_size
            ))

            for _ in range(2):  # random showing results
                pick = np.random.randint(0, len(train_target_batch))
                _v, indices = torch.max(probs, dim=2)  # (bs, number_of_cities)
                print('Train pred: ', [v.item() for v in indices[pick]])
                print('Train label:', [v.item() for v in train_target_batch[pick]])

            validate(model, test_dataloader, number_of_cities, is_train=True)


def validate(model, dataloader, number_of_cities, is_train=False):
    sample_batched = next(iter(dataloader))
    valid_data = to_var(sample_batched['Points'])
    valid_target = to_var(sample_batched['Solution'])

    probs = model(valid_data)                        # (bs, M, L) = (1000, 10, 10)
    valid_output = probs.view(-1, number_of_cities)  # (bs * M, L)

    valid_target_flatten = valid_target.flatten()              # (bs * M)
    valid_loss = F.cross_entropy(valid_output, valid_target_flatten)

    # max values, max indices
    _v, indices = torch.max(probs, dim=2) # (bs, M)

    ####################################################
    #show validate examples
    if not is_train:
        for i in range(len(indices)):
            print('-----')
            print('Validate Data: ', [v.tolist() for v in valid_data[i]])
            print('Validate Pred: ', [v.item() for v in indices[i]])
            print('Validate Label:', [v.item() for v in valid_target[i]])
            if torch.equal(valid_target[i], indices[i]):
                print('SAME')
            if i > 20: break
    ############################################################

    correct_count = sum([1 if torch.equal(ind, y) else 0 for ind, y in zip(indices, valid_target)])
    print('Validate Loss: {:.5f}, Validate Accuracy: {:.2f}% ({}/{})'.format(
        valid_loss,
        correct_count / len(valid_data) * 100,
        correct_count,
        len(valid_data)
    ))
    print()


def main():
    data_size = 20240
    test_data_size = 100

    weight_size = 256
    embed_size = 128
    batch_size = 256
    n_epochs = 300

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
    validate(model, test_dataloader, number_of_cities)


if __name__ == '__main__':
    main()
