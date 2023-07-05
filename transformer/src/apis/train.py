import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import make_cuda
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(args, model, data_loader):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    num_epochs = args.num_epochs

    for epoch in range(num_epochs):
        loss = 0
        for step, (features, targets) in enumerate(data_loader):
            features = make_cuda(features)
            targets = make_cuda(targets)

            optimizer.zero_grad()
            pre, _ = model(features)
            crs_loss = model.cross_entropy(pre, targets.reshape(-1)) #已经忽视了0
            loss += crs_loss.item()
            crs_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.3d/%.3d] Step [%.3d/%.3d]: CROSS_loss=%.4f, RCROSS_loss=%.4f"
                      % (epoch + 1,
                         num_epochs,
                         step + 1,
                         len(data_loader),
                         loss / args.log_step,
                         math.sqrt(loss / args.log_step)))
                loss = 0

        # Loss = []
        # for step, (features, targets) in enumerate(valid_data_loader):
        #     features = make_cuda(features)
        #     targets = make_cuda(targets)
        #     model.eval()
        #     preds = model(features)
        #     valid_loss = CrossLoss(preds, targets)
        #     Loss.append(valid_loss)
        # print("Valid loss: %.3d\n" % (np.mean(Loss)))

    return model


def evaluate(args, model, data_loader):
    model.eval()
    loss = []
    for step, (features, targets) in enumerate(data_loader):
        features = make_cuda(features)
        targets = make_cuda(targets)
        pre, _ = model(features)
        crs_loss = model.cross_entropy(pre, targets.reshape(-1))
        loss.append(crs_loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    print("loss=%.4f" % (np.mean(loss)))
