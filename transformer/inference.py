import torch
import argparse
import numpy as np
from src.models.model import Poetry_Model
from src.datasets.dataloader import train_vec
from src.utils.utils import make_cuda


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--model', type=str, default='transformer',
                        help="lstm/GRU/Seq2Seq/Transformer/GPT-2")
    parser.add_argument('--Word2Vec', default=True)
    parser.add_argument('--strict_dataset', default=False, help="strict dataset")
    parser.add_argument('--n_hidden', type=int, default=128)

    parser.add_argument('--save_path', type=str, default='save_models/transformer_200.pth')

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
        max_sent_len = 6

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
                # pre, _ = model(sentence_ids_embedding)
            elif model_name == 'gpt-2':
                pass
            elif model_name == 'bert':
                pre, _ = model(input_eval)
            elif model_name == 'bart':
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
                from transformers import BertTokenizer
                import random
                # tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")
                tokenizer=BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
                sentence_ids = tokenizer(sentence,return_tensors="pt")["input_ids"][0]
                # 去掉结束标记
                sentence_ids = sentence_ids[:-1].unsqueeze(0)
                pre, _ = model(sentence_ids)
                # 预训练本身也没有问题
                # 解码编码过程本身没有问题
                # 用不同的函数加载预训练模型也都一样
                val, idx=torch.topk(pre[-1],k=1,dim=-1)
                char_generated = tokenizer.decode(idx[random.randint(0,len(idx)-1)].item()) # 用模型的tokenizer解码
            if model_name != 'bart':
                import random
                val, idx=torch.topk(pre[-1],k=1,dim=-1)
                char_generated = index_2_word[idx[random.randint(0,len(idx)-1)].item()]
                # 以新生成的字为输入继续向下生成
                input_eval = word_2_index[char_generated]
            if char_generated == '。':
                break
            sentence += char_generated

        poem += '\n' + sentence

    return poem


def generate_poetry_style(style):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline
    from transformers import BertTokenizer, BartForConditionalGeneration
    tokenizer=BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    # tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")
    style_ids = tokenizer(style,return_tensors="pt")["input_ids"][0]
    # 去掉结束标记
    style_ids = style_ids[:-1]
    _poem=style
    poem=""
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    model = Text2TextGenerationPipeline(model,tokenizer)
    for i in range(20):
        next_word=model(_poem+"[MASK]")[0]['generated_text'][-1]
        poem=poem+next_word
        _poem=_poem+next_word
    return poem


if __name__ == '__main__':
    args = parse_arguments()
    all_data, (w1, word_2_index, index_2_word) = train_vec()
    args.word_size, args.embedding_num = w1.shape
    # string = input("诗头:")
    string = '自然语言'
    
    model = Poetry_Model(args.n_hidden, args.word_size, args.embedding_num, args.Word2Vec, args.model)

    model.load_state_dict(torch.load(args.save_path))
    model = make_cuda(model)
    # 打印参数
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    #         break
    # print(model(torch.tensor([[0,0]])))    
    # print(model(torch.tensor([[100,20000]])))    
    # print(model(torch.tensor([[1027,726]])))    
    poem = generate_poetry(model, args.model, string, w1, word_2_index, index_2_word)
    print(poem)
    # from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
    # tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    # model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    # text2text_generator = Text2TextGenerationPipeline(model, tokenizer)  
    # pre=text2text_generator("乱花渐欲迷[MASK]眼", max_length=50, do_sample=False)
    # print(pre[0]['generated_text'])
    # print(generate_poetry_style(style="会当凌绝"))
