import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from utils import ImgLabelLoader
from model import ActNet


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(img_path, label_path, weights, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActNet().to(device)
    model.eval()

    test_set = ImgLabelLoader(img_path, label_path)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    # load weights
    model.load_state_dict(torch.load(weights, map_location=device))

    with torch.no_grad():  # without gradient
        correct = 0
        total = 0
        preds = []
        y_true = []
        for batch_i, (test_x, test_y) in enumerate(tqdm(test_loader, desc='Testing ')):
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            scores = model(test_x)
            _, pred = torch.max(scores.data, 1)
            correct += (pred == test_y).sum().item()
            total += test_y.size(0)
            y_true.extend(list(test_y.data.cpu().numpy()))
            preds.extend(list(pred.data.cpu().numpy()))

        test_accu = float(correct / total)
        # print('Accuracy of the network on test set: {} %.2f'.format(100 * test_accu))
        print('Accuracy of the network on test set: %.2f' % (100 * test_accu))

        with open('results/confusion_matrix.txt', 'w') as f:
            f.write(str(y_true) + '\n' + str(preds))

        cm = confusion_matrix(y_true, preds)
        print('Confusion Matrix: \n', cm)
        plot_confusion_matrix(cm, ['take', 'cut', 'wash'], 'The confusion matrix of action classification')
        plt.savefig('./confusion_matrix.png')
        plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--img_path', type=str, default='./data/hands_actions/test', help='the test image directory')
    parse.add_argument('--label_path', type=str, default='./data/hands_actions/test.csv', help='the path of label')
    parse.add_argument('--weights', type=str, default='weights/best-0.0001-99epoch.pt', help='weights path')
    parse.add_argument('--batch_size', type=int, default=2, help='size of each image batch')
    opt = parse.parse_args()

    print(opt)

    test(opt.img_path, opt.label_path, opt.weights, opt.batch_size)
