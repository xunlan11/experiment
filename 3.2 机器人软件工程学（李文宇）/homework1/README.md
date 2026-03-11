# <center> 作业一
作业要求：
1. 下载并编译OpenCV 3.4.1源码。
2. 使用其编写一个显示图片的程序。
3. 使用CMake和make进行编译链接程序。
4. 提交到远程仓库。

实现步骤：
- 下载并编译源码
1. 创建homework1/目录
2. 从OpenCV官网`https://opencv.org/releases/`下载3.4.1版本的源码压缩包，并解压到适当位置
3. 在解压得到的opencv-3.4.1/中建立build/
4. 在opencv-3.4.1/中使用`cmake -S ./ -B build/ -D CMAKE_BUILD_TYPE=Release -D BUILD_opencv_python3=OFF`命令进行编译，添加`-D BUILD_opencv_python3=OFF`是因为没有匹配的python导致报错
5. 在build/中使用`make -j$(nproc)`命令进行快速编译
6. 在build/中使用`sudo make insall`命令进行安装
- 编写程序并编译
1. 在homework1/中建立display/
2. 下载图片到display/
3. 在display/中编写display.cpp程序，注意图片的索引地址
4. 在display/中编写CMakeLists.txt，注意关联的cpp文件(display.cpp)、生成的文件名(display)和查找的OpenCV库
5. 在display/中使用`cmake ./`命令进行编译（这里并未创建build/）
6. 在display/中使用`make`命令进行生成
7. 在display/中使用`./display`命令运行程序，获得图片
- 提交到远程仓库