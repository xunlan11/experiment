# <center> 作业三
作业要求：
1. 编写消息发布器和消息订阅器，前者采用c++，后者采用Python。
2. 消息由浮点数和字符串两种数据类型组成。
3. 上传代码及说明文档。

实现步骤：
- 构建包并编译
1. 在课堂使用的工作空间catkin_ws下的src/中，使用`catkin_create_pkg homework3 roscpp rospy std_msgs message_generation message_runtime`命令创建包，并添加相应依赖。
2. 在homework3/下创建msg/文件夹，并添加messages.msg文件，其包含需求的信息，为浮点数和字符串两种。
3. 在homework3/下的package.xml文件中，确保其有message_generation和message_runtime。该步在上课中提到，但新包中已经存在。
4. 在homework3/下的CMakeLists.txt文件中修改内容，确保查找依赖的包（该步在上课中提到，但新包中已经存在），并取消add_message_files和generate_messages部分的注释。注意关联msg文件时要确保文件名正确。
5. 在catkin_ws/下使用`catkin_make`命令进行编译。
- c++消息发布器
1. 在homework3/下的src/中新建cpp文件，添加c++的消息发布器代码，注意关联msg文件时要确保文件名正确。
2. 在homework3/下的CMakeLists.txt文件中修改内容，注意生成的node的名称。
- Python消息订阅器
1. 在homework3/下的scripts/中新建py文件，添加Python的消息订阅器代码，注意关联msg文件时要确保文件名正确。
2. 在homework3/下使用`chmod +x scripts/subscriber.py`确保给予Python脚本可执行权限。
- 编译与运行
1. 在catkin_ws/下使用`catkin_make`命令进行编译。
2. 在第一个终端下启动roscore。
3. 在第二个终端下使用`source catkin_ws/devel/setup.bash`命令更新环境变量，之后使用`rosrun homework3 publisher`命令启动消息发布器。
4. 在第三个终端下使用`source catkin_ws/devel/setup.bash`命令更新环境变量，之后使用`rosrun homework3 subscriber.py`命令启动消息订阅器。
- 提交到远程仓库
注：仅提交了包的内容，需要在对应的工作空间下运行。