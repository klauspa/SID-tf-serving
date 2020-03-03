# SID-tf-serving
tensorflow serving to serve a model called Learning-to-See-in-the-Dark from this repo https://github.com/cchen156/Learning-to-See-in-the-Dark

# 依赖安装
1.推荐anaconda3(python3.6.9) tensorflow-gpu==1.14
2.rawpy (pip install rawpy)
3.flask (pip install flask)
4.tensorflow-model-server (tensorflow-serving, 安装:
    (1) echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-         universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
        curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
    (2) apt-get update && apt-get install tensorflow-model-server  )
   
# 如何使用
1.运行python test_Sony.py以生成pb模型 供tensorflow-serving 使用
2.运行tensorflow_model_server --rest_api_port=8501 --model_name=saved_model -model_base_path=/.../saved_model/pb (根据绝对路径)
  此时tensorflow-serving已单独部署
3.运行python app.py部署flask 
  flask作用：与tensorflow-serving通信获取推理数据
             接收上传图片与图片序列压缩包（可返回生成视频）
4.运行demo_client.py
  简单python客户端测试 第十六行url = "http://localhost:8080/predict/" + image_or_video[0] 
  image_or_video[0] 为上传图片
  image_or_video[1] 为上传压缩包
  
