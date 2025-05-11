import sys
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from PyQt5.QtGui import QPixmap, QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# n皇后问题
class n_Queens:
    def __init__(self,n):
        self.size = n       # 棋盘的尺寸
        self.solutions = [] # 保存解
    # 判断新皇后是否和已有皇后冲突
    def check(self, X, Y):
        return X[0] == Y[0] or X[1] == Y[1] or X[0] - X[1] == Y[0] - Y[1] or X[0] + X[1] == Y[0] + Y[1]
    def valid(self, queens, new_queen):
        for queen in queens:
            if self.check(queen, new_queen):
                return False
        return True
    # 加入与回溯
    def n_queens(self, queens=[]):
        row = len(queens)
        # 达到最后一行，保存解
        if row == self.size:
            self.solutions.append(queens.copy())
            return
        for col in range(self.size):
            new_queen = [row, col]
            if self.valid(queens, new_queen):
                # 新皇后不和已有皇后冲突，添加到列表并向下进行
                queens.append(new_queen)
                if self.n_queens(queens):
                    return True
                # 加入的皇后导致无法继续加入新皇后，回溯
                queens.pop()
        return False
    # 绘制棋盘与解
    def board_solution(self, ax, solution_index):
        # 检验是否有解
        if not self.solutions:
            return
        # 绘制棋盘
        ax.clear()                 # 清空上一个解
        ax.set_xlim(0, self.size)  # 设置x、y轴范围，并使之等比例，关闭坐标轴
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_axis_off()
        # 填色
        colors = ['w', 'k']
        for i in range(self.size):
            for j in range(self.size):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[(i + j) % 2]))
        # 标数
        for i in range(self.size):
            ax.text(i + 0.5, self.size + 0.5, str(i + 1), ha='center', va='center')
            ax.text(-0.5, self.size - i - 0.5, str(i + 1), ha='center', va='center', rotation=0)
        # 绘制解
        crown_img = mpimg.imread('crown.png')
        crown_imagebox = OffsetImage(crown_img, zoom=0.1)
        solution = self.solutions[solution_index]
        for queen in solution:
            ab = AnnotationBbox(crown_imagebox, (queen[1] + 0.5, self.size - queen[0] - 0.5), frameon=False)
            ax.add_artist(ab)
# 可视化
class Ui_MainWindow():
    def __init__(self):
        self.start = 0 # 开始标签
        self.current_solution_index = 0  # 运行序数标签
    # 主界面
    def setupUi(self, MainWindow):
        # 主窗口
        MainWindow.resize(1200, 900)
        # 中央部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        # 布局器
        layout = QtWidgets.QVBoxLayout(self.centralwidget)
        # 背景
        self.background = QtWidgets.QLabel(self.centralwidget)
        image = QPixmap("background.jpg")
        self.background.setPixmap(
            image.scaled(MainWindow.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.background.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self.background)
        # 输入框及其文字
        self.cin_data = QtWidgets.QLineEdit(self.centralwidget)
        self.cin_data.setGeometry(QtCore.QRect(950, 100, 150, 80))
        self.cin_data.setStyleSheet("QLineEdit { font-family: Arial; font-size: 18pt; background-color: rgba(128, 128, 128, 50%); border: 1px solid gray; }")
        self.cin_label = QtWidgets.QLabel(self.centralwidget)
        self.cin_label.setGeometry(QtCore.QRect(900, 20, 250, 100))
        # 开始按钮
        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(925, 200, 200, 100))
        self.start_button.setStyleSheet("QPushButton { background-color: rgba(128, 128, 128, 50%); }")
        self.start_button.clicked.connect(self.run)
        # 暂停继续按钮
        self.pause_resume_button = QtWidgets.QPushButton(self.centralwidget)
        self.pause_resume_button.setGeometry(QtCore.QRect(925, 300, 200, 100))
        self.pause_resume_button.setStyleSheet("QPushButton { background-color: rgba(128, 128, 128, 50%); }")
        self.pause_resume_button.clicked.connect(self.pause_resume)
        # 步进按钮
        self.go_button = QtWidgets.QPushButton(self.centralwidget)
        self.go_button.setGeometry(QtCore.QRect(925, 400, 200, 100))
        self.go_button.setStyleSheet("QPushButton { background-color: rgba(128, 128, 128, 50%); }")
        self.go_button.clicked.connect(self.step)
        # 结果显示及其文字
        self.answer_lcd = QtWidgets.QLCDNumber(self.centralwidget)
        self.answer_lcd.setGeometry(QtCore.QRect(900, 550, 250, 100))
        self.answer_label = QtWidgets.QLabel(self.centralwidget)
        self.answer_label.setGeometry(QtCore.QRect(900, 480, 250, 100))
        # 运行时间显示
        self.time_label1 = QtWidgets.QLabel(self.centralwidget)
        self.time_label1.setGeometry(QtCore.QRect(900, 650, 250, 100))
        self.time_label2 = QtWidgets.QLabel(self.centralwidget)
        self.time_label2.setGeometry(QtCore.QRect(900, 700, 250, 100))
        # 动图展示及动态下标
        self.figure = Figure(figsize=(5, 5), dpi=200)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setParent(self.centralwidget)
        self.canvas.setGeometry(QtCore.QRect(50, 50, 800, 800))
        self.dynamic_label = QtWidgets.QLabel(self.centralwidget)
        self.dynamic_label.setGeometry(QtCore.QRect(100, 750, 300, 100))
        # 校徽
        self.badge = QtWidgets.QLabel(self.centralwidget)
        self.badge.setGeometry(QtCore.QRect(1050, 750, 150, 150))
        image = QPixmap('校徽.png').scaled(self.badge.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.badge.setPixmap(image)
        # 提示
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(50, 830, 1000, 100))
        # 文字
        self.Chinese(MainWindow)
    # 文本
    def Chinese(self, MainWindow):
        self.font = QFont()
        self.font.setFamily('宋体')
        self.font.setPointSize(18)
        translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(translate("1", "n皇后问题"))
        self.cin_label.setText(translate("1", "请输入皇后数："))
        self.cin_label.setFont(self.font)
        self.start_button.setText(translate("1", "开始"))
        self.start_button.setFont(self.font)
        self.pause_resume_button.setText(translate("1", "暂停"))
        self.pause_resume_button.setFont(self.font)
        self.go_button.setText(translate("1", "步进"))
        self.go_button.setFont(self.font)
        self.answer_label.setText(translate("1", "解数为："))
        self.answer_label.setFont(self.font)
        self.time_label1.setText(translate("1", "求解耗时为为："))
        self.time_label1.setFont(self.font)
        self.time_label2.setText(translate("1", "0.0s"))
        self.time_label2.setFont(self.font)
        self.label.setText(translate("1", "请先输入皇后数，在按开始按钮后再点击暂停和步进按钮！"))
        self.label.setFont(self.font)
    # 更新动图部分
    def update_plot(self):
        if self.current_solution_index < len(self.solver.solutions):
            self.solver.board_solution(self.ax, self.current_solution_index)
            self.canvas.draw()
            self.current_solution_index += 1
            self.dynamic_label.setText(f"当前为第{self.current_solution_index}个解")
            self.dynamic_label.setFont(self.font)
        else:
            self.timer.stop()
    # 链接1：start
    def run(self):
        try:
            n = int(self.cin_data.text())
            if n <= 0:
                raise ValueError("请输入一个正数！")
        except ValueError as e:
            QtWidgets.QMessageBox.warning(None, "错误输入", f"不接受{e}作为参数")
            return
        # 清空（为多次运行服务）
        self.figure.clf()
        self.canvas.draw()
        # 实例化问题
        self.start = 1
        start_time = time.time()
        self.solver = n_Queens(n)
        self.solver.n_queens()
        end_time = time.time()
        # 结果、运行时间展示
        self.answer_lcd.display(len(self.solver.solutions))
        self.time_label2.setText(f"{round(end_time - start_time, 5)}s")
        self.time_label2.setFont(self.font)
        # 动图
        self.ax = self.figure.add_subplot(111)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.update_plot())
        self.timer.start(100)  # 每个解停留0.1s
    # 链接2：pause_resume
    def pause_resume(self):
        if self.start == 0:
            QtWidgets.QMessageBox.warning(None, "错误操作", "请不要在未运行时暂停！")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.pause_resume_button.setText("继续")
        else:
            self.timer.start(100)
            self.pause_resume_button.setText("暂停")
    # 链接3：go
    def step(self):
        if self.start == 0:
            QtWidgets.QMessageBox.warning(None, "错误操作", "请不要在未运行时步进！")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.pause_resume_button.setText("继续")
            QtWidgets.QMessageBox.warning(None, "错误操作", "请不要在运行非暂停时步进！")
        else:
            if self.current_solution_index == len(self.solver.solutions):
                QtWidgets.QMessageBox.warning(None, "错误操作", "已经是最后一种解了，无法继续步进！")
                return
            self.update_plot()

app = QApplication(sys.argv)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec())