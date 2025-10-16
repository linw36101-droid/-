OpenCV 实验项目合集  

包含内容  
- 实验 1：颜色分割（Color Segmentation）  
- 实验 2：最小框选与多边形拟合（Bounding Box & Polygon Approximation）  
- 扩展实验：视频中的蓝色瓶子检测（Blue Bottle Detection）  

---

 
- Python ≥ 3.8  
- OpenCV ≥ 4.0  
- NumPy ≥ 1.20  

安装命令：
```bash
pip install opencv-python numpy
 实验一：颜色分割（Color Segmentation）
 实验目标
利用 HSV 空间提取特定颜色（粉色），在图像中实现自动分割与高亮显示。

🔧 实现步骤
颜色空间转换：BGR → HSV

颜色阈值设定：使用 cv2.inRange() 提取目标颜色范围

掩膜与结果融合：cv2.bitwise_and() 获取目标区域

输出结果：显示 mask 图像与分割图像

 相关文件
源文件：opencv.py

输入图片：image1.png

输出结果：outputs/result.png

 实验二：最小框选与多边形拟合（Bounding Box & Polygon Approximation）
 实验目标
通过轮廓检测，对光源或特定形状进行最小矩形框选与多边形拟合。

 实现步骤
图像预处理：灰度化 + 二值化

轮廓提取：cv2.findContours() 获取目标边缘

面积过滤：排除过小或过大的轮廓

多边形拟合：cv2.approxPolyDP() 平滑轮廓边缘

最小矩形框选：cv2.boundingRect() 绘制目标矩形

 相关文件
源文件：main.py

输入图片：image1.png

输出图片：outputs/final_result.png

 扩展实验：视频中蓝色瓶子检测（Blue Bottle Detection）
实验目标
检测视频中蓝色瓶子区域，实现实时框选显示。

🔧 实现原理
逐帧读取视频：cv2.VideoCapture()

颜色空间转换：BGR → HSV

颜色范围设定：蓝色 HSV 阈值

轮廓检测与过滤：基于面积选出瓶身与瓶盖

绘制结果：在原视频上框选目标

 相关文件
源文件：video.py

输入视频：video.mp4

输出视频：outputs/snaps/blue_bottle_split.mp4

使用方式
打开 PyCharm；

将本仓库下载或克隆至本地；

在对应目录下执行：

运行 opencv.py 查看颜色分割结果；

运行 main.py 查看最小框选与拟合结果；

运行 video.py 查看蓝色瓶子检测视频结果；

输出结果将自动保存在 outputs/ 文件夹中。
