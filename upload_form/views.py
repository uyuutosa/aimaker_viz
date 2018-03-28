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

convert_image = {
    1: lambda img: img,
    2: lambda img: img.transpose(I.FLIP_LEFT_RIGHT),                              # 左右反転
    3: lambda img: img.transpose(I.ROTATE_180),                                   # 180度回転
    4: lambda img: img.transpose(I.FLIP_TOP_BOTTOM),                              # 上下反転
    5: lambda img: img.transpose(I.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_90),  # 左右反転＆反時計回りに90度回転
    6: lambda img: img.transpose(I.ROTATE_270),                                   # 反時計回りに270度回転
    7: lambda img: img.transpose(I.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_270), # 左右反転＆反時計回りに270度回転
    8: lambda img: img.transpose(I.ROTATE_90),                                    # 反時計回りに90度回転
}



def form(request):
    if request.method != 'POST':
        return render(request, 'upload_form/form.html')

    if "ft'in" in request.POST['unit_height']:
        feet, inch = request.POST['height'].split("'")
        height = float(feet) * 30.48 + float(inch) * 2.54
    else:
        height = request.POST['height']

    if "lb" in request.POST['unit_weight']:
        lb = request.POST['weight']
        weight = float(lb) * 0.45359237  
    else:
        weight = request.POST['weight']

    if "gpu_ids" in request.POST['unit_weight']:
        gpu_ids = request.POST['gpu_ids']

    else:
        gpu_ids = "0,0"

        
        

    weight = request.POST['weight']
    resize = int(request.POST['resize'])

    if 'Tight' in request.POST['cloth']:
        feel = 'tight'
    elif 'Loose' in request.POST['cloth']:
        feel = 'loose'
    else:
        feel = 'normal'

    o = open(os.path.join(UPLOADE_DIR, intermed_file), "w")
    o.write(str(height) + "\n")
    o.write(str(weight) + "\n")
    o.write(feel + "\n")
    o.write(gpu_ids + "\n")

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
    exif = img._getexif()
    if exif is not None:
        orientation = exif.get(0x112, 1)
        img = convert_image[orientation](img)
    width, height = img.size
    ratio = height / width
    img.resize((resize, int(resize*ratio))).save(os.path.join(UPLOADE_DIR, name + ".png"))

    insert_data = FileNameModel(file_name = file.name)
    insert_data.save()

    return redirect('upload_form:form')

def complete(request):

    o = open(os.path.join(UPLOADE_DIR, intermed_file))
    height   = int(o.readline())
    weight   = int(o.readline())
    feel     = o.readline().strip()
    gpu_ids  = [int(x) for x in o.readline().strip().split(",")]

    a = c.clothingSizeEstimator(
            os.path.join(UPLOADE_DIR, 'frontal.png'),
            os.path.join(UPLOADE_DIR, 'side.png'),
            height_cm=height,
            weight_kg=weight,
            feel=feel
            )

    a.getExtractBackgroundImages(transform="", gpu_id=gpu_ids[0], divide_size=(1,1),pad=40,thresh=0)
    a.getPoseImages(gpu_id=gpu_ids[1])
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
