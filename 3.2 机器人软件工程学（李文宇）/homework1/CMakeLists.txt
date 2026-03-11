cmake_minimum_required(VERSION 3.16.3)
project(display)
add_executable(display display.cpp) 

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 链接 OpenCV 库
target_link_libraries(display ${OpenCV_LIBS})