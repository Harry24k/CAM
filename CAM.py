import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
    
r"""
Implementation of CAM

Arguments:
    model (nn.Module): a model with one GAP(Global Average Pooling) and one FC(Fully-Connected).
    images (torch.tensor): input images of (batch_size, n_channel, height, width).
    last_conv_name (str) : the name of the last convolutional layer of the model.
    fc_name (str) : the name of the last fully-connected layer of the model.
    label (list): According to the label, activated area will be changed.
      * Default : None (It will be automatically determined by predicted label)
      * Warning : It has to be same size as the batch_size of the input images.
    normalize (Bool) : Normalized output will be returned if it is True.
      * Default : True (The output have a value between 0 and 255)
    resize (Bool) : Resized output will be returned.
      * Default : True (The output will be resized as same as the input images)

.. note:: it is modified from "https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py"

"""

def CAM(model, images, last_conv_name, fc_name, label=None, normalize=True, resize=True) :

    device = next(model.parameters()).device

    size = images.shape[-2:]
    
    # 가장 마지막 Conv Layer의 Output 가져오기
    last_conv_features = []

    def hook_feature(module, input, output):
        last_conv_features.append(output.data)

    # inception5b가 Output을 출력할 때마다 hook_feature을 호출
    model._modules.get(last_conv_name).register_forward_hook(hook_feature)
    
    # FC Layer의 weight을 가져오기
    params = dict(getattr(model, fc_name).named_parameters())
    weight_softmax = params['weight'].data
    
    # eval 모드에서 forward 진행
    model.eval()
    feature = model(images.to(device))

    # 예측값 가져오기
    _, pre = feature.max(dim=1)    
    conv_feature = last_conv_features[0]
    b, nc, h, w = conv_feature.shape

    if label is None :
        label = pre
    
    cam = torch.bmm(weight_softmax[label].reshape(b, 1, nc), conv_feature.reshape((b, nc, h*w)))
    cam = cam.reshape(b, 1, h, w)
    
    # Min-Max Normalization
    if normalize :
        cam = (cam - cam.min()) / (cam.max()- cam.min())
        cam = (255 * cam).int().float()

    # Resize
    if resize :
        cam = nn.UpsamplingBilinear2d(size=size)(cam)
    
    return cam, pre