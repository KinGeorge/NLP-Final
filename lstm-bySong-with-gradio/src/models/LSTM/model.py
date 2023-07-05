import torch
import numpy as np
import torch.nn as nn


class Poetry_Model_lstm(nn.Module):
    def __init__(self, hidden_num, word_size, embedding_num, Word2Vec):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_num = hidden_num
        self.Word2Vec = Word2Vec

        self.embedding = nn.Embedding(word_size, embedding_num)
        self.lstm = nn.LSTM(input_size=embedding_num, hidden_size=hidden_num, batch_first=True, num_layers=2,
                            bidirectional=False)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten(0, 1)
        self.linear = nn.Linear(hidden_num, word_size)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, xs_embedding, h_0=None, c_0=None):
        # xs_embedding: [batch_size, max_seq_len, n_feature] n_feature=128
        if h_0 == None or c_0 == None:
            h_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
            c_0 = torch.tensor(np.zeros((2, xs_embedding.shape[0], self.hidden_num), dtype=np.float32))
        h_0 = h_0.to(self.device)
        c_0 = c_0.to(self.device)
        xs_embedding = xs_embedding.to(self.device)
        if not self.Word2Vec:
            xs_embedding = self.embedding(xs_embedding)
        hidden, (h_0, c_0) = self.lstm(xs_embedding, (h_0, c_0))
        hidden_drop = self.dropout(hidden)
        hidden_flatten = self.flatten(hidden_drop)
        pre = self.linear(hidden_flatten)
        # pre：[batch_size*max_seq_len, vocab_size]
        return pre, (h_0, c_0)
