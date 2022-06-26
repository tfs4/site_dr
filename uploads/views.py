from django.shortcuts import render
from .forms import UploadForm
from django.http import JsonResponse
from django.shortcuts import HttpResponse
import torch
from torch import nn
import torchvision.models as models
import cv2, glob, numpy
import os
from PIL import Image
from torchvision.transforms import transforms
from django.contrib import messages

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

def home_view(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    classe = 'DR_0'
    if is_ajax(request=request):
       if form.is_valid():
            document = form.save(commit=False)
            document.save()
            result =  classifier(document.image)
            result = result.numpy()
            result = str(result[0])
            if result == '1':
                mensagem = "Diabetic Retinopathy Detected"
            else:
                mensagem = "Diabetic retinopathy NOT detected"

       return JsonResponse({'message': mensagem})
    context = {
        'form': form,
    }
    return render(request, 'uploads/main.html', context)


def get_densenet121_2_classes():

    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(256, 2),
    )

    return densenet121



def scaleRadius(img,scale):
    k = img.shape[0]/2
    x = img[int(k), :, :].sum(1)
    r=(x>x.mean()/10).sum()/2
    if r == 0:
        r = 1
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def preprocessing(img):
        scale = 500
        cont = 0
        a = cv2.imread('media/'+img)
        a = scaleRadius(a, 500)
        b = numpy.zeros(a.shape)
        x = a.shape[1] / 2
        y = a.shape[0] / 2
        center_coordinates = (int(x), int(y))
        cv2.circle(b, center_coordinates, int(scale * 0.9), (1, 1, 1), -1, 8, 0)
        aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128) * b + 128 * (1 - b)

        dir = img.split('/')
        cv2.imwrite('media/processed/'+dir[1], aa)






def classifier(file):
    preprocessing(str(file))

    dir = str(file).split('/')

    classificador = get_densenet121_2_classes()
    path_loader = torch.load('model_A.pt')
    classificador.load_state_dict(path_loader)
    classificador.eval()

    test_transform = transforms.Compose([transforms.Resize([1024, 1024]),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ])

    a = Image.open('media/processed/'+dir[1])
    image = test_transform(a).float()
    image = image.unsqueeze(0)
    output = classificador(image)
    predictions = output.argmax(dim=1).detach()
    return predictions