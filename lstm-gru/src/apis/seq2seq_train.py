import torch
from src.utils.utils import make_cuda
import torch.optim as optim
import torch.nn as nn
from src.models.seq2seq import Encoder, Decoder, Seq2Seq


def seq_train(args, data_loader):
    model = Seq2Seq(Encoder(args.word_size, args.n_hidden, args.embedding_num),
                                 Decoder(args.embedding_num, args.n_hidden, args.embedding_num)).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    num_epochs = args.num_epochs
    loss_function = nn.MSELoss()

    for epoch in range(num_epochs):
        loss = 0
        for step, (features, targets) in enumerate(data_loader):
            features = make_cuda(features)
            targets = make_cuda(targets)

            optimizer.zero_grad()

            pre = model(features, targets)
            mse_loss = loss_function(pre, targets)
            loss += mse_loss.item()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.3d/%.3d] Step [%.3d/%.3d]: CROSS_loss=%.4f"
                      % (epoch + 1,
                         num_epochs,
                         step + 1,
                         len(data_loader),
                         loss / args.log_step))
                loss = 0
