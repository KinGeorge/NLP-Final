from src.utils.utils import make_cuda
from src.apis.train import train, evaluate
from src.models.LSTM.model import Poetry_Model_lstm
import argparse
import torch
import os
from src.datasets.dataloader import Poetry_Dataset, train_vec, get_poetry, split_text
from torch.utils.data import DataLoader


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--model', type=str, default='lstm',
                        help="lstm/GRU/Seq2Seq/Transformer/GPT-2")
    parser.add_argument('--Word2Vec', default=True)
    parser.add_argument('--Augmented_dataset', default=False, help="augmented dataset")
    parser.add_argument('--strict_dataset', default=False, help="strict dataset")

    parser.add_argument('--batch_size', type=int, default=64,
                        help="Specify batch size")
    parser.add_argument('--num_epochs', type=int, default=50,
                        help="Specify the number of epochs for competitive search")
    parser.add_argument('--log_step', type=int, default=100,
                        help="Specify log step size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument('--data', type=str, default='data/poetry.txt',
                        help="Path to the dataset")
    parser.add_argument('--Augmented_data', type=str, default='data/poetry_7.txt',
                        help="Path to the Augmented_dataset")
    parser.add_argument('--n_hidden', type=int, default=128)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--save_path', type=str, default='save_models/')

    return parser.parse_args()


def main():
    args = parse_arguments()
    # if you want to change the data(org data or argument data), please delete file: 'split_poetry.txt' and 'org_poetry.txt'
    if os.path.exists("data/split_poetry.txt") and os.path.exists("data/org_poetry.txt"):
        print("pre_file exit!")
    else:
        split_text(get_poetry(args))

    all_data, (w1, word_2_index, index_2_word) = train_vec()
    args.word_size, args.embedding_num = w1.shape

    dataset = Poetry_Dataset(w1, word_2_index, all_data, args.Word2Vec)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=int(args.batch_size/4), shuffle=True)

    if args.model == 'lstm':
        best_model = Poetry_Model_lstm(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec)
    elif args.model == 'GRU':
        best_model = Poetry_Model_lstm(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec)
    elif args.model == 'Seq2Seq':
        best_model = Poetry_Model_lstm(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec)
    elif args.model == 'Transformer':
        best_model = Poetry_Model_lstm(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec)
    elif args.model == 'GPT-2':
        best_model = Poetry_Model_lstm(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec)
    else:
        print("Please choose a model!\n")

    best_model = make_cuda(best_model)
    best_model = train(args, best_model, train_data_loader)

    torch.save(best_model.state_dict(), args.save_path + args.model + '_' + str(args.num_epochs)+'.pth')

    print('test evaluation:')
    evaluate(args, best_model, valid_data_loader)


if __name__ == '__main__':
    main()
