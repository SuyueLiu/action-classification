import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ActNet
from utils import ImgLabelLoader
from logger import Logger


def train(lr, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()  # softmax + loss

    train_set = ImgLabelLoader('./data/hands_actions/test', './data/hands_actions/test.csv')
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    val_set = ImgLabelLoader('./data/hands_actions/val', './data/hands_actions/val.csv')
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    # training
    accuracy_his = []
    loss_his = []
    best_val_accuray = -1
    best_model = None

    tf_logger = Logger(os.path.join(sys.path[0] + '/log/'))

    for epoch in range(epochs):
        total_train = 0
        train_correct = 0
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")):
            train_x = x.to(device)
            train_y = y.to(device)

            scores = model(train_x)
            loss = loss_func(scores, train_y)

            _, train_preds = torch.max(scores.data, 1)
            total_train += train_y.size(0)
            train_correct += (train_preds == train_y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_accuracy = float(train_correct/total_train)


        # validate every epoch
        num_correct = 0
        num_samples = 0
        for val_x, val_y in tqdm(val_loader, desc='validation'):
            model.eval()

            val_x = val_x.to(device)
            val_y = val_y.to(device)

            val_scores = model(val_x)
            # if torch.cuda.is_available():
            #     pred_y = torch.max(val_out, 1)[1].cuda().data
            # else:
            #     pred_y = torch.max(val_out, 1)[1].data.numpy()
            _, preds = torch.max(val_scores.data, 1)
            num_correct += (preds == val_y).sum().item()
            num_samples += preds.size(0)

            # val_accuracy = float((pred_y == val_y.data.numpy()).astype(int).sum()) / float(val_y.size(0))
        val_accuracy = float(num_correct / num_samples)
        print('Epoch:', epoch, '| training loss: %.4f' % loss.data.cpu().numpy(), '| val accuracy: %.2f' % val_accuracy)
        # accuracy_his.append(val_accuracy)
        # loss_his.append(loss.data.numpy())
        tf_logger.scalar_summary('train accuracy', train_accuracy, epoch)
        tf_logger.scalar_summary('loss', loss.data.numpy(), epoch)

        if val_accuracy > best_val_accuray:
            best_val_accuray = val_accuracy
            best_model = model
            torch.save(best_model.state_dict(), './weights/best.pt')  # save the best weights


if __name__ == '__main__':
    train(lr=0.0001, epochs=10)