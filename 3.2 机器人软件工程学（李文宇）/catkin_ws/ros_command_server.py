import socket
import threading
import subprocess
import json
import os
import time
import signal
import sys

class ROSCommandServer:
    def __init__(self, host='0.0.0.0', start_port=9877):
        self.host = host
        self.port = start_port
        self.server_socket = None
        self.running_processes = {}
        
        # 命令映射表
        self.commands = {
            "启动底盘": [
                {"cmd": "roslaunch turtlebot_bringup minimal.launch", "run_mode": "keep_alive"}
            ],
            
            "创建地图": [
                {"cmd": "roslaunch turtlebot_navigation gmapping_demo.launch", "run_mode": "keep_alive"},
                {"cmd": "roslaunch turtlebot_rviz_launchers view_navigation.launch", "run_mode": "keep_alive"},
                {"cmd": "rosrun teleop_twist_keyboard teleop_twist_keyboard.py", "run_mode": "keep_alive"}
            ],
            
            "保存地图": [
                {"cmd": "rosrun map_server map_saver -f ~/catkin_ws/maps/my_map", "run_mode": "run_once"}
            ],
            
            "开始导航": [
                {"cmd": "rosrun hello_pkg nav1.py", "run_mode": "run_once"}  
            ],
            
            "控制机械臂": [
                {"cmd": "roslaunch my_dynamixel controller_manager.launch", "run_mode": "keep_alive"},
                {"cmd": "roslaunch my_dynamixel start_tilt_controller.launch", "run_mode": "keep_alive"}
            ],
            
            "拿给客人": [
                {"cmd": "rosrun hello_pkg nav2.py", "run_mode": "run_once"}
            ],

            "目标识别": [
                {"cmd": "rosrun my_dynamixel arm1.py", "run_mode": "run_once"},
                {"cmd": "rosrun for_version align_and_grasp.py", "run_mode": "run_once"},
                {"cmd": "rosrun my_dynamixel arm3.py", "run_mode": "run_once"},
                {"cmd": "rosrun my_dynamixel move_forward.py", "run_mode": "run_once"},
                {"cmd": "rosrun my_dynamixel arm2.py", "run_mode": "run_once"}
            ]
        }
        
        # 需要顺序执行的命令标记
        self.sequential_commands = ["目标识别"]
        
        # 获取当前工作空间路径
        self.catkin_ws_path = os.path.expanduser("~/catkin_ws")
        if not os.path.exists(self.catkin_ws_path):
            self.catkin_ws_path = None
            print("警告: 未找到 ~/catkin_ws 目录，将仅使用系统 ROS 环境")
        else:
            print(f"使用工作空间: {self.catkin_ws_path}")
        
    def start(self):
        port = self.port
        max_attempts = 10
        
        # 初始化并启动服务器
        for attempt in range(max_attempts):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind((self.host, port))
                self.server_socket.listen(5)
                self.port = port
                print(f"ROS命令服务器已启动，监听 {self.host}:{port}")
                break
            except OSError as e:
                if e.errno == 98:
                    print(f"端口 {port} 被占用，尝试端口 {port + 1}...")
                    port += 1
                    continue
                else:
                    raise
        else:
            raise RuntimeError(f"无法绑定任何端口，尝试范围 {self.port}-{port}")
        
        # 接受客户端连接
        while True:
            client, address = self.server_socket.accept()
            print(f"收到来自 {address} 的连接")
            client_thread = threading.Thread(target=self.handle_client, args=(client,))
            client_thread.daemon = True
            client_thread.start()
            
    def handle_client(self, client):
        try:
            # 处理客户端请求
            data = client.recv(1024).decode('utf-8')
            if not data:
                return
                
            print(f"收到命令: {data}")
            command_data = json.loads(data)
            task_name = command_data.get("command")
            
            # 已知命令响应
            if task_name in self.commands:
                result = self.execute_commands(task_name)
                response = {"status": "success", "message": result}
            else:
                response = {"status": "error", "message": f"未知命令: {task_name}"}
                
            print(f"发送响应: {response}")
            client.sendall(json.dumps(response).encode('utf-8'))
        except Exception as e:
            print(f"处理客户端请求时出错: {e}")
            try:
                client.sendall(json.dumps({"status": "error", "message": f"服务器错误: {str(e)}"}).encode('utf-8'))
            except:
                pass
        # 确保无论处理成功还是失败，最终都会关闭与该客户端的连接
        finally:
            client.close()
    
    # 执行命令主函数
    def execute_commands(self, task_name):
        if task_name in self.running_processes and self.running_processes[task_name]:
            return f"{task_name}已经在运行中"
            
        commands = self.commands[task_name]
        result_message = f"正在执行{task_name}..."
        is_sequential = task_name in self.sequential_commands
        
        # 是否顺序执行
        if is_sequential:
            command_thread = threading.Thread(target=self.run_sequential_commands, args=(task_name, commands))
        else:
            command_thread = threading.Thread(target=self.run_parallel_commands, args=(task_name, commands))
            
        command_thread.daemon = True
        command_thread.start()
        
        return result_message
    
    # 并行执行命令
    def run_parallel_commands(self, task_name, commands):
        processes = []
        
        # 创建环境设置命令
        env_cmd = ". /opt/ros/noetic/setup.bash"
        if self.catkin_ws_path:
            env_cmd += f" && . {self.catkin_ws_path}/devel/setup.bash"
        
        # 存储后台进程
        keep_alive_processes = []
        
        for cmd_spec in commands:
            cmd = cmd_spec['cmd']
            run_mode = cmd_spec['run_mode']
            
            try:
                print(f"执行命令: {cmd} (模式: {run_mode})")
                
                # 处理sudo命令
                if cmd.startswith("sudo "):
                    print("注意: 发现sudo命令，将尝试无密码执行")
                    
                # 使用bash明确执行命令，并设置环境
                full_cmd = f"{env_cmd} && {cmd}"
                
                if run_mode == "keep_alive":
                    # 持续运行的命令使用Popen
                    process = subprocess.Popen(
                        ["/bin/bash", "-c", full_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    keep_alive_processes.append(process)
                    print(f"后台进程ID: {process.pid}")
                    
                    # 实时输出命令执行结果
                    stdout_thread = threading.Thread(
                        target=self.log_output, 
                        args=(process.stdout, f"[{task_name}] 输出: ")
                    )
                    stderr_thread = threading.Thread(
                        target=self.log_output, 
                        args=(process.stderr, f"[{task_name}] 错误: ")
                    )
                    
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    stdout_thread.start()
                    stderr_thread.start()
                else:
                    # 运行后关闭的命令使用run
                    result = subprocess.run(
                        ["/bin/bash", "-c", full_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    print(f"运行一次命令完成: {cmd}, 退出码: {result.returncode}")
                    if result.stdout:
                        print(f"[{task_name}] 输出: {result.stdout}")
                    if result.stderr:
                        print(f"[{task_name}] 错误: {result.stderr}")
                time.sleep(0.5)
            except Exception as e:
                print(f"执行命令 {cmd} 失败: {e}")
        
        # 存储后台进程
        self.running_processes[task_name] = keep_alive_processes
        print(f"任务 '{task_name}' 中的持续运行命令已启动")
    
    # 顺序执行命令
    def run_sequential_commands(self, task_name, commands):
        processes = []
        
        # 创建环境设置命令
        env_cmd = ". /opt/ros/noetic/setup.bash"
        if self.catkin_ws_path:
            env_cmd += f" && . {self.catkin_ws_path}/devel/setup.bash"
        
        # 标记已运行任务，防止重复执行
        self.running_processes[task_name] = True
        
        for i, cmd_spec in enumerate(commands):
            cmd = cmd_spec['cmd']
            run_mode = cmd_spec['run_mode']
            
            try:
                print(f"[顺序执行 {i+1}/{len(commands)}] 命令: {cmd} (模式: {run_mode})")
                
                full_cmd = f"{env_cmd} && {cmd}"
                
                if run_mode == "run_once":
                    # 使用run等待命令完成
                    result = subprocess.run(
                        ["/bin/bash", "-c", full_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    
                    print(f"[顺序命令 {i+1}/{len(commands)}] 命令完成，状态码: {result.returncode}")
                    
                    if result.stdout:
                        print(f"[{task_name}] 输出: {result.stdout}")
                    if result.stderr:
                        print(f"[{task_name}] 错误: {result.stderr}")
                    
                    if result.returncode != 0:
                        print(f"⚠️ 命令 '{cmd}' 返回非零状态码: {result.returncode}")
                else:
                    # 支持持续运行的命令
                    process = subprocess.Popen(
                        ["/bin/bash", "-c", full_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # 实时输出
                    stdout_thread = threading.Thread(
                        target=self.log_output, 
                        args=(process.stdout, f"[{task_name}] 输出: ")
                    )
                    stderr_thread = threading.Thread(
                        target=self.log_output, 
                        args=(process.stderr, f"[{task_name}] 错误: ")
                    )
                    
                    stdout_thread.daemon = True
                    stderr_thread.daemon = True
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    print(f"[顺序执行] 后台进程ID: {process.pid}")
                    # 添加进程到列表
                    self.running_processes.setdefault(task_name, []).append(process)
                    
            except Exception as e:
                print(f"执行命令 {cmd} 失败: {e}")
        
        print(f"[{task_name}] 所有顺序命令已执行完毕")
        # 如果任务中没有持续运行的命令，标记为空
        if task_name in self.running_processes and self.running_processes[task_name] is True:
            self.running_processes[task_name] = []
    
    # 实时记录命令输出
    def log_output(self, pipe, prefix):
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    print(f"{prefix}{line.strip()}")
        except ValueError:
            pass # 忽略"read of closed file"错误

# Ctrl+C退出
def signal_handler(sig, frame):
    print("\n正在关闭服务器...")
    if hasattr(server, 'server_socket') and server.server_socket:
        server.server_socket.close()
    
    # 清理所有运行中的进程
    for task_name, processes in server.running_processes.items():
        if processes:
            print(f"关闭任务 '{task_name}' 的进程...")
            for p in processes:
                if p.poll() is None:  # 检查进程是否仍在运行
                    p.terminate()
                    print(f"已终止进程 {p.pid}")
    
    sys.exit(0)

if __name__ == "__main__":
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    server = ROSCommandServer(start_port=9877)
    try:
        print("ROS命令服务器启动中...")
        print("使用Ctrl+C退出")
        server.start()
    except KeyboardInterrupt:
        print("服务器关闭")
    except Exception as e:
        print(f"服务器错误: {e}")