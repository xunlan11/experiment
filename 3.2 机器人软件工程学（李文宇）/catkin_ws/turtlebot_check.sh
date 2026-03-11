#!/bin/bash

echo "--- TurtleBot 自检 ---"
echo "确保source ROS环境并roscore"

# 颜色代码
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE_BOLD='\033[1;34m'
NC='\033[0m'

# 函数：检查ROS节点是否正在运行，check_ros_node "/node_name_pattern"
check_ros_node() {
    local node_pattern=$1
    local description=$2
    echo -n -e "[ ${BLUE_BOLD}${description}节点检查${NC} ] "
    if rosnode list 2>/dev/null | grep -q "$node_pattern"; then
        echo -e "${GREEN}运行中${NC}"
        return 0
    else
        echo -e "${RED}未运行${NC}"
        return 1
    fi
}

# 函数：检查ROS话题是否正在发布且有发布者，check_ros_topic "/topic_name"
check_ros_topic() {
    local topic_name=$1
    local description=$2
    echo -n -e "[ ${BLUE_BOLD}${description}话题检查${NC} ] "
    if rostopic info "$topic_name" &>/dev/null && [[ $(rostopic info "$topic_name" 2>/dev/null | grep "Publishers:" | grep -v "None") ]]; then
        echo -e "${GREEN}活跃${NC}"
        return 0
    else
        echo -e "${RED}不活跃 或 无发布者${NC}"
        return 1
    fi
}

echo "--- 硬件检查 ---"

# TurtleBot底盘和电池
echo -e "${BLUE_BOLD}[TurtleBot 底盘和电池]${NC}"
base_active=false
# 检查常见的底盘节点 (例如 Kobuki 的 mobile_base_nodelet_manager) 或 /odom 话题
if rosnode list 2>/dev/null | grep -q "/mobile_base_nodelet_manager"; then
    echo -e "[ TurtleBot 底盘 (mobile_base_nodelet_manager) ] ${GREEN}运行中${NC}"
    base_active=true
elif rosnode list 2>/dev/null | grep -q "/mobile_base"; then
     echo -e "[ TurtleBot 底盘 (mobile_base 节点) ] ${GREEN}运行中${NC}"
     base_active=true
elif rostopic info "/odom" &>/dev/null && [[ $(rostopic info "/odom" 2>/dev/null | grep "Publishers:" | grep -v "None") ]]; then
    echo -e "[ TurtleBot 底盘 (通过 /odom 话题) ] ${GREEN}活跃${NC}"
    base_active=true
else
    echo -e "[ TurtleBot 底盘 ] ${RED}未检测到 (已检查常用节点/话题)。请核实 'minimal.launch' 是否已正确启动。${NC}"
fi

if $base_active; then
    # 电池电量 (Kobuki 特定, 假设使用 /mobile_base/sensors/core)
    # kobuki_msgs/CoreSensors 中的 'battery' 字段是 0.1V 为单位的电压值
    battery_voltage_raw=$(timeout 1s rostopic echo -n 1 /mobile_base/sensors/core/battery 2>/dev/null | tail -n1 | tr -d '[:space:]')

    if [[ -n "$battery_voltage_raw" && "$battery_voltage_raw" =~ ^[0-9]+$ ]]; then
        battery_voltage=$(echo "scale=1; $battery_voltage_raw / 10.0" | bc)
        # 用于百分比计算的电压范围: MinVoltage = 13.2V (132), MaxOperationalVoltage = 16.0V (160)
        MIN_BATTERY_RAW=132 
        MAX_BATTERY_RAW=160 # 用于百分比显示的保守最大值
        
        percentage=0
        if (( $(echo "$battery_voltage_raw > $MIN_BATTERY_RAW" | bc -l) && $(echo "$MAX_BATTERY_RAW > $MIN_BATTERY_RAW" | bc -l) )); then
            clamped_voltage_raw=$battery_voltage_raw
            if (( $(echo "$clamped_voltage_raw < $MIN_BATTERY_RAW" | bc -l) )); then clamped_voltage_raw=$MIN_BATTERY_RAW; fi
            if (( $(echo "$clamped_voltage_raw > $MAX_BATTERY_RAW" | bc -l) )); then clamped_voltage_raw=$MAX_BATTERY_RAW; fi
            percentage=$(echo "scale=0; ($clamped_voltage_raw - $MIN_BATTERY_RAW) * 100 / ($MAX_BATTERY_RAW - $MIN_BATTERY_RAW)" | bc)
        elif (( $(echo "$battery_voltage_raw <= $MIN_BATTERY_RAW" | bc -l) )); then
            percentage=0
        fi
        if (( $percentage > 100 )); then percentage=100; fi # 限制最大为100

        echo -e "[ TurtleBot 电池 ] 电压: $battery_voltage V (原始值: $battery_voltage_raw)。预估电量: ${YELLOW}$percentage %${NC}"
        if (( $(echo "$battery_voltage_raw < 140" | bc -l) )); then # 大约 14.0V
             echo -e "                 ${YELLOW}警告: 电池电量较低。${NC}"
        fi
    else
        echo -e "[ TurtleBot 电池 ] ${YELLOW}无法从 /mobile_base/sensors/core/battery 获取电池电量。${NC}"
        echo -e "                 Kobuki 底盘是否正在运行并发布此话题?"
    fi
else
    echo -e "[ TurtleBot 电池 ] ${YELLOW}因底盘未激活，跳过电池检查。${NC}"
fi
echo ""

# 摄像头
echo -e "${BLUE_BOLD}[相机连接]${NC}"
echo -n -e "[ 相机设备 (/dev/video*) ] "
if ls /dev/video* 1> /dev/null 2>&1; then
    echo -e "${GREEN}检测到${NC}。设备: $(ls /dev/video* | tr '\n' ' ')"
else
    echo -e "${RED}在 /dev 中未检测到视频设备。${NC}"
fi
camera_topic_found=false
if check_ros_topic "/usb_cam/image_raw" "USB相机"; then camera_topic_found=true; fi
if ! $camera_topic_found && check_ros_topic "/camera/rgb/image_raw" "Freenect RGB相机"; then camera_topic_found=true; fi
if ! $camera_topic_found; then
    echo -e "[ 相机ROS话题 ] ${YELLOW}未发现活动的标准相机话题。请检查相机启动文件 (usb_cam 或 freenect)。${NC}"
fi
echo ""

# 机械臂连接
echo -e "${BLUE_BOLD}[机械臂连接]${NC}"
ARM_DEVICE="/dev/ttyUSB0"
echo -n -e "[ 机械臂设备 ($ARM_DEVICE) ] "
if [ -e "$ARM_DEVICE" ]; then
    echo -e "${GREEN}检测到${NC}。"
    echo -n -e "    权限检查: "
    if [ -r "$ARM_DEVICE" ] && [ -w "$ARM_DEVICE" ]; then
        echo -e "${GREEN}读写正常${NC}。"
    else
        echo -e "${YELLOW}读写权限可能存在问题。当前权限: $(stat -c %a "$ARM_DEVICE")。请考虑 'sudo chmod 666 $ARM_DEVICE' 或设置udev规则。${NC}"
    fi
else
    echo -e "${RED}未检测到。请检查连接和驱动程序。${NC}"
fi
arm_ros_ok=false
if check_ros_node "/tilt_controller_spawner" "机械臂控制器生成器"; then arm_ros_ok=true; fi
if ! $arm_ros_ok && check_ros_topic "/tilt_controller/command" "机械臂指令话题"; then arm_ros_ok=true; fi
if ! $arm_ros_ok && check_ros_topic "/tilt_controller/state" "机械臂状态话题"; then arm_ros_ok=true; fi
if ! $arm_ros_ok; then
    echo -e "[ 机械臂ROS接口 ] ${YELLOW}未找到机械臂ROS组件 (节点/话题)。请检查 'roslaunch my_dynamixel start_tilt_controller.launch'。${NC}"
fi
echo ""

echo "--- 软件检查 ---"

# 地图文件
echo -e "${BLUE_BOLD}[地图文件]${NC}"
MAP_FILE_PATH_FROM_README="$HOME/my_maps/my_map.yaml"
echo -n -e "[ 地图文件 ($MAP_FILE_PATH_FROM_README) ] "
if [ -f "$MAP_FILE_PATH_FROM_README" ]; then
    echo -e "${GREEN}找到${NC}。"
    echo -n -e "    TURTLEBOT_MAP_FILE 环境变量: "
    if [ -n "$TURTLEBOT_MAP_FILE" ]; then
        if [ "$TURTLEBOT_MAP_FILE" == "$MAP_FILE_PATH_FROM_README" ]; then
            echo -e "${GREEN}已设置且匹配 ($TURTLEBOT_MAP_FILE)${NC}。"
        else
            echo -e "${YELLOW}已设置但路径不匹配 ('$TURTLEBOT_MAP_FILE' vs '$MAP_FILE_PATH_FROM_README')。${NC}"
        fi
    else
        echo -e "${YELLOW}未设置。请考虑执行: echo 'export TURTLEBOT_MAP_FILE=$MAP_FILE_PATH_FROM_README' >> ~/.bashrc 并 source ~/.bashrc${NC}"
    fi
else
    echo -e "${RED}未找到。${NC}"
    echo -e "    请确保已保存地图: rosrun map_server map_saver -f ~/my_maps/my_map"
fi
echo ""