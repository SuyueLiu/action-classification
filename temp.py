# import torch
# import cv2
# import numpy as np
# from torch.autograd import Variable
#
# from model import ActNet
#
# model = ActNet()
#
#
# def get_test_input(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (300, 300))
#     img_ = img[:,:,::-1].transpose((2, 0, 1))
#     img_ = img_[np.newaxis, :,:,:]/255.0
#     img_ = torch.from_numpy(img_).float()
#     img_ = Variable(img_)
#     return img_
#
#
# img = get_test_input('./data/samples/cut_P01_P01_01_0000006751.jpg')
#
#
# scores = model(img)
#
# print(scores)


with open('/Users/aaron/Downloads/loss.svg', 'r') as f:
    content = f.read()
print(content)
