import os
import argparse

import numpy as np
import torch
import shutil
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from PIL import Image

from model import ActNet


def get_names(names_path):
    with open(names_path, 'r') as f:
        names = f.readlines()
    if '.DS_Store' in names:
        names.remove('.DS_Store')

    return names


def detect(weights, source, names_path, output):
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActNet().to(device)
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device))

    model.eval()

    imgs = []
    predictions = []

    for img_name in os.listdir(source):
        if img_name != '.DS_Store':
            img_path = os.path.join(source, img_name)
            input_img = transforms.ToTensor()(Image.open(img_path))
            input_img = input_img.to(device)
            input_img = Variable(torch.unsqueeze(input_img, dim=0).float(), requires_grad=False)

            with torch.no_grad():
                score = model(input_img)
                _, pred = torch.max(score.data, 1)

            imgs.append(img_path)
            predictions.append(pred)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 3)]

    names = get_names(names_path)

    for img_i, (path, img_pred) in enumerate(zip(imgs, predictions)):
        # create plot
        img = np.array(Image.open(path))
        # plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        plt.text(
            0,
            0,
            s=names[img_pred],
            color=colors[img_pred],
            verticalalignment='top',
            fontdict={'size': 25}
        )

        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split('/')[-1].split('.')[0]
        plt.savefig('output/{}.jpg'.format(filename), bbox_inches='tight', pad_inches=0.0)
        print('Saving image: ', path.split('/')[-1], '---> prediction: ', names[img_pred])
        plt.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weights', type=str, default='weights/best-0.0001-99epoch.pt', help='weights path')
    parse.add_argument('--source', type=str, default='data/samples', help='image source')
    parse.add_argument('--names', type=str, default='data/action.names', help='names path')
    parse.add_argument('--out', type=str, default='output', help='output')

    opt = parse.parse_args()
    print(opt)

    detect(opt.weights, opt.source, opt.names, opt.out)
