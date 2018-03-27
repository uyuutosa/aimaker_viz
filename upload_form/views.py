from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.template.context_processors import csrf
from django.conf import settings
from upload_form.models import FileNameModel
import sys, os
import clothing_size_estimator.clothing_size_estimator as c

UPLOADE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/static/posts/'

def form(request):
    if request.method != 'POST':
        return render(request, 'upload_form/form.html')

    if 'process' in request.POST:
        return redirect('upload_form:complete')
    elif 'frontal' in request.POST:
        name = 'frontal'
    else:
        name = 'side'

    o = open(os.path.join(UPLOADE_DIR, 'height_width.txt'), "w")
    height
    width  = int(o.readline())

    path = os.path.join(UPLOADE_DIR, name + ".png")
    #path = os.path.join(UPLOADE_DIR, file.name)
    destination = open(path, 'wb')

    for chunk in file.chunks():
        destination.write(chunk)

    insert_data = FileNameModel(file_name = file.name)
    insert_data.save()

    return redirect('upload_form:form')

def complete(request):

    o = open(os.path.join(UPLOADE_DIR, 'height_width.txt'))
    height = int(o.readline())
    width  = int(o.readline())

    a = c.clothingSizeEstimator(
            os.path.join(UPLOADE_DIR, 'frontal.png'),
            os.path.join(UPLOADE_DIR, 'side.png'),
            height_cm=height,
            width_cm=width,
            )

    a.getExtractBackgroundImages(transform="",gpu_id=0, divide_size=(1,1),pad=40,thresh=0)
    a.getPoseImages(gpu_id=1)
    param = a.getImage()
    I.fromarray(a.frontal_outlined_arr.astype(uint8)).save('frontal_outline.png')
    I.fromarray(a.slide_outlined_arr.astype(uint8)).save('side_outline.png')
    I.fromarray(a.frontal_labeled_arr.astype(uint8)).save('frontal_labeled.png')
    I.fromarray(a.side_labeled_arr.astype(uint8)).save('side_labeled.png')
    I.fromarray(a.frontal_binary.astype(uint8)).save('frontal_binary.png')
    I.fromarray(a.slide_binary.astype(uint8)).save('side_binary.png')
    I.fromarray(a.frontal_pose_labeled_image.astype(uint8)).save('frontal_pose.png')
    I.fromarray(a.side_pose_labeled_image.astype(uint8)).save('side_pose.png')

    return render(request, 'upload_form/complete.html', param)
