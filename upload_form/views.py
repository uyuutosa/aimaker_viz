from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.template.context_processors import csrf
from django.conf import settings
from upload_form.models import FileNameModel
import sys, os
import clothing_size_estimator.clothing_size_estimator as c
import PIL.Image as I
from numpy import *

UPLOADE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/static/posts/'
intermed_file = 'intermed.txt'

def form(request):
    if request.method != 'POST':
        return render(request, 'upload_form/form.html')

    height = request.POST['height']
    weight = request.POST['weight']
    resize = int(request.POST['resize'])

    if 'Tight' in request.POST['cloth']:
        feel = 'tight'
    elif 'Loose' in request.POST['cloth']:
        feel = 'loose'
    else:
        feel = 'Nomal'

    o = open(os.path.join(UPLOADE_DIR, intermed_file), "w")
    o.write(str(height) + "\n")
    o.write(str(weight) + "\n")
    o.write(feel + "\n")

        

    if 'process' in request.POST:
        return redirect('upload_form:complete')
    elif 'frontal' in request.POST:
        name = 'frontal'
    else:
        name = 'side'

    file = request.FILES['form']

    #path = os.path.join(UPLOADE_DIR, name + ".png")
    path = os.path.join(UPLOADE_DIR, file.name)
    destination = open(path, 'wb')

    for chunk in file.chunks():
        destination.write(chunk)

    img = I.open(path)
    width, height = img.size
    ratio = height / width
    img.resize((resize, int(resize*ratio))).save(os.path.join(UPLOADE_DIR, name + ".png"))

    insert_data = FileNameModel(file_name = file.name)
    insert_data.save()

    return redirect('upload_form:form')

def complete(request):

    o = open(os.path.join(UPLOADE_DIR, 'height_weight.txt'))
    height = int(o.readline())
    weight = int(o.readline())
    feel   = o.readline()

    a = c.clothingSizeEstimator(
            os.path.join(UPLOADE_DIR, 'frontal.png'),
            os.path.join(UPLOADE_DIR, 'side.png'),
            height_cm=height,
            weight_kg=weight,
            feel=feel
            )

    a.getExtractBackgroundImages(transform="",gpu_id=0, divide_size=(1,1),pad=40,thresh=0)
    a.getPoseImages(gpu_id=1)
    param = a.getImage()
    I.fromarray(a.frontal_outlined_arr.astype(uint8)).save(os.path.join(UPLOADE_DIR, 'frontal_outline.png'))
    I.fromarray(a.side_outlined_arr.astype(uint8)).save(os.path.join(UPLOADE_DIR, 'side_outline.png'))
    I.fromarray(a.frontal_labeled_arr.astype(uint8)[...,::-1]).save(os.path.join(UPLOADE_DIR, 'frontal_estimate.png'))
    I.fromarray(a.side_labeled_arr.astype(uint8)[...,::-1]).save(os.path.join(UPLOADE_DIR, 'side_estimate.png'))
    I.fromarray(a.frontal_binary.astype(uint8)).save(os.path.join(UPLOADE_DIR, 'frontal_binary.png'))
    I.fromarray(a.side_binary.astype(uint8)).save(os.path.join(UPLOADE_DIR, 'side_binary.png'))
    I.fromarray(a.frontal_pose_labeled_image.astype(uint8)).save(os.path.join(UPLOADE_DIR, 'frontal_pose.png'))
    I.fromarray(a.side_pose_labeled_image.astype(uint8)).save(os.path.join(UPLOADE_DIR, 'side_pose.png'))

    return render(request, 'upload_form/complete.html', param)
