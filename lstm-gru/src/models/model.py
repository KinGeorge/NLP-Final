import torch
import numpy as np
import torch.nn as nn

# from seq2seq import Encoder, Decoder, Seq2Seq


class Poetry_Model(nn.Module):
    def __init__(self, hidden_num, word_size, embedding_num, Word2Vec, model='lstm'):
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
            # self.model = Seq2Seq(Encoder(word_size, hidden_num, embedding_num),
            #                      Decoder(embedding_num, hidden_num, embedding_num))
            pass
        elif model == 'transformer':
            self.model = nn.Transformer(d_model=embedding_num, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
                                        dim_feedforward=256, dropout=0.3, activation='relu')
        elif model == 'gpt-2':
            pass
        else:
            raise ValueError("Please choose a model!\n")

        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0, 1)
        if model == 'transformer':
            self.linear = nn.Linear(embedding_num, word_size)
        else:
            self.linear = nn.Linear(hidden_num, word_size)
        self.cross_entropy = nn.CrossEntropyLoss()

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

    def forward_transformer(self, xs_embedding, src_mask=None, tgt_mask=None):
        xs_embedding = xs_embedding.to(self.device)
        if not self.Word2Vec:
            xs_embedding = self.embedding(xs_embedding)
        output = self.model(xs_embedding, xs_embedding, src_mask, tgt_mask)
        output = self.dropout(output)
        output = self.flatten(output)
        pre = self.linear(output)
        return pre, None

    def forward(self, xs_embedding, h_0=None, c_0=None):
        if self.model_name == 'lstm':
            return self.forward_lstm(xs_embedding, h_0, c_0)
        elif self.model_name == 'gru':
            return self.forward_gru(xs_embedding, h_0)
        elif self.model_name == 'transformer':
            return self.forward_transformer(xs_embedding)
