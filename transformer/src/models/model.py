import torch
import numpy as np
import torch.nn as nn
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertModel, BertForTokenClassification

class Poetry_Model(nn.Module):
    def __init__(self, hidden_num, word_size, embedding_num, Word2Vec, model='lstm',seq_len=78):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_num = hidden_num
        self.Word2Vec = Word2Vec
        self.model_name = model

        self.embedding = nn.Embedding(word_size, embedding_num)
        if model == 'lstm':
            self.model = nn.LSTM(input_size=embedding_num, hidden_size=hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)
        elif model == 'gru':
            self.model = nn.GRU(input_size=embedding_num, hidden_size=hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)
        elif model == 'seq2seq':
            pass
        elif model == 'transformer':
            self.model = nn.Transformer(d_model=embedding_num, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
                                    dim_feedforward=256, dropout=0.3, activation='relu',batch_first=True)
        elif model == 'gpt-2':
            pass
        elif model == 'bert':
            self.model = PoetryBertModel(input_length=seq_len)
        elif model == 'bart':
            from transformers import BartForConditionalGeneration
            self.model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
        else:
            raise ValueError("Please choose a model!\n")
        
        self.pos_encoder= PositionalEncoding(d_model=embedding_num,dropout=0.5)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0, 1)
        if model == 'transformer':
            self.linear = nn.Linear(embedding_num, word_size)

        else:
            self.linear = nn.Linear(hidden_num, word_size)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    def forward_lstm(self, xs_embedding, h_0=None, c_0=None):
        # xs_embedding: [batch_size, max_seq_len, n_feature] n_feature=256
        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            c_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        xs_embedding = xs_embedding.to(self.device)
        if not self.Word2Vec:
            xs_embedding = self.embedding(xs_embedding)
        hidden, (h_0, c_0) = self.model(xs_embedding, (h_0, c_0))
        hidden_drop = self.dropout(hidden)
        hidden_flatten = self.flatten(hidden_drop)
        pre = self.linear(hidden_flatten)
        # pre：[batch_size*max_seq_len, vocab_size]
        return pre, (h_0, c_0)

    def forward_gru(self, xs_embedding, h_0=None):
        if h_0 == None:
            h_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)
        xs_embedding = xs_embedding.to(self.device)
        if not self.Word2Vec:
            xs_embedding = self.embedding(xs_embedding)
        output, h_0 = self.model(xs_embedding, h_0)
        output = self.dropout(output)
        output = self.flatten(output)
        pre = self.linear(output)
        return pre, h_0
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)  # [sz, sz] 作用是生成上三角矩阵
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))  # [sz, sz] 作用是将上三角矩阵中的0替换为-inf，1替换为0
        return mask

    def forward_transformer(self, xs_embedding, targets, src_mask=None, tgt_mask=None):
        xs_embedding = xs_embedding.to(self.device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets).unsqueeze(0).unsqueeze(0)
        targets = targets.to(self.device)
        # targets = self.embedding(targets)
        if not self.Word2Vec:
            xs_embedding = self.embedding(xs_embedding)
        xs_embedding = self.pos_encoder(xs_embedding) # [batch_size, max_seq_len, n_feature]
        targets = self.pos_encoder(targets) # [batch_size, max_seq_len, n_feature]
        tgt_mask = self.generate_square_subsequent_mask(targets.shape[1]).to(self.device)
        src_mask = self.generate_square_subsequent_mask(xs_embedding.shape[1]).to(self.device)
        output = self.model(xs_embedding, targets, src_mask, tgt_mask)
        output = self.flatten(output)
        pre = self.linear(output)
        return pre, None
    
    def forward_bert(self, token, input_length=None):
        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token).unsqueeze(0).unsqueeze(0).to(self.device) #inference阶段
            input_length=1
        input_mask = (token != 0).float()
        return self.model(token, input_mask,input_length), None

    def forward_bart(self, token):
        token=token.to(self.device)
        tgt_mask = self.generate_square_subsequent_mask(token.shape[1]).to(self.device)
        src_mask = self.generate_square_subsequent_mask(token.shape[1]).to(self.device)
        output_logits = self.model(input_ids=token,attention_mask=src_mask[0:token.shape[0]],decoder_input_ids=token,decoder_attention_mask=tgt_mask[0:token.shape[0]]).logits
        pre = self.flatten(output_logits)
        return pre, None
    
    def forward(self, xs_embedding, h_0=None, c_0=None):
        if self.model_name == 'lstm':
            return self.forward_lstm(xs_embedding, h_0, c_0)
        elif self.model_name == 'gru':
            return self.forward_gru(xs_embedding, h_0)
        elif self.model_name == 'transformer':
            return self.forward_transformer(xs_embedding, xs_embedding)
        elif self.model_name == 'bert':
            return self.forward_bert(xs_embedding)
        elif self.model_name == 'bart':
            return self.forward_bart(xs_embedding)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p = dropout) 

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term) # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class PoetryBertModel(nn.Module):
    """
    基于BERT预训练模型的诗歌生成模型
    """
    def __init__(self, input_length: int):
        super(PoetryBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.bert_for_class = BertForTokenClassification(self.bert_model.config)
        self.flatten = nn.Flatten(0, 1)
        # 生成下三角矩阵，用来mask句子后边的信息
        self.sequence_length = input_length
        self.lower_triangle_mask = torch.tril(torch.ones((self.sequence_length, self.sequence_length), dtype=torch.long))

    def forward(self, token, input_mask, input_length=None): # token.shape=[64,78]
        # 计算 attention_mask 
        attention_mask = torch.bmm(input_mask.unsqueeze(-1), input_mask.unsqueeze(-2)).squeeze()

        # 注意力机制计算中有效的位置
        if input_length is not None:
            lower_triangle_mask = torch.tril(torch.ones((input_length, input_length), dtype=torch.long))
        else:
            lower_triangle_mask = self.lower_triangle_mask
        attention_mask = attention_mask * lower_triangle_mask.to(attention_mask.device)

        output_logits = self.bert_for_class(token, attention_mask=attention_mask)
        output_logits = self.flatten(output_logits[0])

        return output_logits