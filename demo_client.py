#coding=utf-8
import requests
import base64

image_or_video = ["image/", "video/"]
image_path = "AHMG1737[1].DNG"
zip_path = "视频.zip"
ratio = 200 #用户自定义
user_id = "0123456789" #用户ID用户名 

#测试网络
response1 = requests.post("http://localhost:8080/hello/")
print(response1.content)

#是图片还是压缩包 image_or_video[1]压缩包 image_or_video[0]图片
url = "http://localhost:8080/predict/" + image_or_video[0]

#单张图片
if url[-6:] == "image/":
    with open(image_path, "rb") as imageFile:
        b64_image = base64.b64encode(imageFile.read())
    
    data = {"user_id": user_id, "ratio": ratio, "filename": image_path, "b64": b64_image}

    resp_image = requests.post(url, data=data)

    outpng = resp_image.content
    outpng = base64.b64decode(outpng)

    with open("out.png", "wb") as f:
        f.write(outpng)

#视频处理
if url[-6:] == "video/":
    zip_file = {'file':open(zip_path,'rb')}
    data = {"user_id": user_id, "ratio": ratio}
    respvideo = requests.post(url, files=zip_file, data=data)
    
    if respvideo == "请上传zip类型压缩文件":
        print(respvideo)

    else:
        video_data = respvideo.content
        with open("video_recieved.avi", "wb") as vid:
            vid.write(video_data)




