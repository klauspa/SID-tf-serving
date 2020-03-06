import base64
import json
from io import BytesIO

import numpy as np
import requests
import argparse
import rawpy
import time
import shutil
import zipfile
import os
import cv2
import glob
import scipy.misc
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (4095 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def unzip_file(zip_src, dst_dir):
    """
    解压zip文件
    :param zip_src: zip文件的全路径
    :param dst_dir: 要解压到的目的文件夹
    :return:
    """
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, "r")
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        return "请上传zip类型压缩文件"

#tensorflow serving地址
SERVER_URL = 'http://localhost:8501/v1/models/saved_model:predict'

# 简单测试
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return "hello world"

@app.route('/predict/image/', methods=['POST'])
def image():
    user_id = request.form["user_id"]
    print(type(user_id))
    print("current user: ".format(user_id))
    ratio = int(request.form["ratio"])
    print("the ratio: ".format(ratio))
    img_filename = request.form["filename"]
    print("the filename: ".format(img_filename))

    DNG_binary = base64.b64decode(request.form["b64"])
    upload_dir_image = "user_images"
    if not os.path.isdir(upload_dir_image):
        os.mkdir(upload_dir_image)
    user_dir = os.path.join(upload_dir_image, user_id)
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)
    #定义上传文件存储位置
    filepath = os.path.join(user_dir, img_filename)
    
    #写入文件
    try:
        with open(filepath, "wb") as f:
            f.write(DNG_binary)
    except:
        return "上传失败 请重试"
    print(filepath)
    raw = rawpy.imread(filepath)
    input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
    #im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    #scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
    
    input_full = np.minimum(input_full, 1.0)
    print("input shape {}".format(input_full.shape))

    start =time.time()
    tf_serving_request = '{{"signature_name": "output", "instances": {} }}'.format(input_full.tolist())
    
    resp = requests.post(SERVER_URL, data=tf_serving_request)
    #200 tensorflow serving成功返回结果
    print('response.status_code: {}'.format(resp.status_code))
    end = time.time()
    print('Running time: {} Seconds'.format(end-start))

    output = resp.json()["predictions"]
    output = np.array(output)
    print("output shape {}".format(output.shape))

    output = np.minimum(np.maximum(output, 0), 1)
    output = output[0, :, :, :]

    out_filename = img_filename.split(".")[0] + ".png"
    out_path = os.path.join(user_dir, out_filename)
    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(out_path)
    
    with open(out_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream

@app.route('/predict/video/', methods=['POST'])
def video():
    user_id = request.form["user_id"]
    print("current user: ".format(user_id))
    ratio = int(request.form["ratio"])
    fps = int(request.form["fps"])

    zip_video = request.files.get("file")
    print(zip_video.filename)

    ret_list = zip_video.filename.rsplit(".", maxsplit=1)
    if len(ret_list) != 2:
        return "请上传zip类型压缩文件"
    if ret_list[1] != "zip":
        return "请上传zip类型压缩文件"
    
    #先保存压缩文件到本地，再对其进行解压，然后删除压缩文件
    upload_dir_video = "user_videos"
    if not os.path.isdir(upload_dir_video):
        os.mkdir(upload_dir_video)
    user_dir = os.path.join(upload_dir_video, user_id)
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)
    #定义上传文件存储位置
    filepath = os.path.join(user_dir, zip_video.filename)
    
    #注意：每次上传解压文件至一个以当前系统时间命名的文件夹 目的是怕多次上传文件积压
    try:
        #存储
        zip_video.save(filepath)
        
        target_path = os.path.join(user_dir, "videos"+str(time.time())) # 解压后的文件保存到的路径
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        ret = unzip_file(filepath, target_path)
        #os.remove(filepath)  # 删除文件
    except:
        return "上传失败 请重试"

    if ret:
        print("保存成功")

    #一帧帧进行处理
    for idx, img in enumerate(os.listdir(target_path)):
        
        img_path = os.path.join(target_path, img)
        print(img_path)
        raw = rawpy.imread(img_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        #im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        #scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
    
        input_full = np.minimum(input_full, 1.0)
        print("input shape {}".format(input_full.shape))

        start =time.time()
        tf_serving_request = '{{"signature_name": "output", "instances": {} }}'.format(input_full.tolist())
        
        resp = requests.post(SERVER_URL, data=tf_serving_request)
        #200 请求tf serving成功
        print('response.status_code: {}'.format(resp.status_code))
        end = time.time()
        print('Running time: {} Seconds'.format(end-start))

        output = resp.json()["predictions"]
        output = np.array(output)
        print("output shape {}".format(output.shape))

        output = np.minimum(np.maximum(output, 0), 1)
        output = output[0, :, :, :]

        out_filename = img.split(".")[0] + ".png"
        out_path = os.path.join(target_path, out_filename)
        scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(out_path)

    image_folder = target_path
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = images[:len(os.listdir(target_path))]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video_name = user_dir + "视频.avi"
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    #shutil.rmtree(target_path) #删除解压目录 包括生成的推理图片

    return send_file(video_name)

if __name__ == "__main__":
    app.run(port=8080, debug=True)


