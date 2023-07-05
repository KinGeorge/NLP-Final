import torch
import argparse
import numpy as np
from src.models.model import Poetry_Model
from src.datasets.dataloader import train_vec
from src.utils.utils import make_cuda


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--model', type=str, default='gru',
                        help="lstm/gru/seq2seq/transformer/gpt-2")
    parser.add_argument('--Word2Vec', default=True)
    parser.add_argument('--strict_dataset', default=False, help="strict dataset")
    parser.add_argument('--n_hidden', type=int, default=128)

    parser.add_argument('--save_path', type=str, default='save_models/gru_25.pth')

    return parser.parse_args()


def generate_poetry(model, model_name, head_string, w1, word_2_index, index_2_word):
    print("藏头诗生成中...., {}".format(head_string))
    poem = ""
    # 以句子的每一个字为开头生成诗句
    for head in head_string:
        if head not in word_2_index:
            print("抱歉，不能生成以{}开头的诗".format(head))
            return

        sentence = head
        max_sent_len = 20

        h_0 = torch.tensor(np.zeros((2, 1, args.n_hidden), dtype=np.float32))
        c_0 = torch.tensor(np.zeros((2, 1, args.n_hidden), dtype=np.float32))

        input_eval = word_2_index[head]
        for i in range(max_sent_len):
            if args.Word2Vec:
                word_embedding = torch.tensor(w1[input_eval][None][None])
            else:
                word_embedding = torch.tensor([input_eval]).unsqueeze(dim=0)
            if model_name == 'lstm':
                pre, (h_0, c_0) = model(word_embedding, h_0, c_0)
            elif model_name == 'gru':
                pre, h_0 = model(word_embedding, h_0)
            elif model_name == 'seq2seq':
                pass
            elif model_name == 'transformer':
                pre, _ = model(word_embedding) 
            elif model_name == 'gpt-2':
                pass
            char_generated = index_2_word[int(torch.argmax(pre))]

            if char_generated == '。':
                break
            # 以新生成的字为输入继续向下生成
            input_eval = word_2_index[char_generated]
            sentence += char_generated

        poem += '\n' + sentence

    return poem


if __name__ == '__main__':
    args = parse_arguments()
    all_data, (w1, word_2_index, index_2_word) = train_vec()
    args.word_size, args.embedding_num = w1.shape
    # string = input("诗头:")
    string = '中山大学'
    
    model = Poetry_Model(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec, args.model)

    model.load_state_dict(torch.load(args.save_path))
    model = make_cuda(model)
    poem = generate_poetry(model, args.model, string, w1, word_2_index, index_2_word)
    print(poem)
