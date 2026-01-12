from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QMessageBox
from PyQt5.QtCore import pyqtSignal  # 核心新增：导入信号类
from PyQt5.QtGui import QFont
from loguru import logger
import config as config
from voiceprint_gui.train_data_window import TrainDataWindow

class TrainCountSettingWindow(QWidget):
    # 核心新增1：定义自定义信号，用于向主界面传递训练样本数（int类型）
    train_data_finished = pyqtSignal(int)

    def __init__(self, username, algorithm):
        super().__init__()
        self.username = username
        self.algorithm = algorithm
        self.train_count = 3
        self.min_required = 3 if algorithm == "SHERPA" else 3
        self.init_ui()
        logger.info(f"用户「{username}」的训练音频数量设置窗口初始化完成，{algorithm}算法最低要求{self.min_required}条")

    def init_ui(self):
        self.setWindowTitle(f"设置训练音频数量 - {self.username}")
        self.setFixedSize(400, 250)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 30, 40, 30)

        # 标题提示
        title_label = QLabel(f"请设置训练音频采集数量（{self.algorithm}算法最低要求{self.min_required}条）")
        title_label.setFont(QFont("微软雅黑", 10))
        main_layout.addWidget(title_label)

        # 数量选择
        count_layout = QHBoxLayout()
        count_label = QLabel("采集数量：")
        count_label.setFont(QFont("微软雅黑", 10))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(3, len(config.TRAIN_TEXT_MATERIALS))
        self.count_spin.setValue(self.min_required)
        self.count_spin.setFont(QFont("微软雅黑", 10))
        count_layout.addWidget(count_label)
        count_layout.addWidget(self.count_spin)
        main_layout.addLayout(count_layout)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("确认")
        self.confirm_btn.setStyleSheet("padding: 8px; background-color: #4CAF50; color: white;")
        self.confirm_btn.clicked.connect(self.confirm_count)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setStyleSheet("padding: 8px; background-color: #f44336; color: white;")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def confirm_count(self):
        """确认训练数量（核心修改：绑定TrainDataWindow的closed信号）"""
        self.train_count = self.count_spin.value()
        if self.train_count < self.min_required:
            reply = QMessageBox.question(
                self,
                "提示",
                f"{self.algorithm}算法需要至少{self.min_required}条训练样本，当前设置为{self.train_count}条，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        logger.success(f"用户「{self.username}」确认训练音频数量：{self.train_count}条（{self.algorithm}算法最低要求{self.min_required}条）")
        self.train_data_window = TrainDataWindow(self.username, self.algorithm, self.train_count)
        # 核心新增2：绑定TrainDataWindow的closed信号到当前窗口的槽函数
        self.train_data_window.closed.connect(self.on_train_data_window_closed)
        self.train_data_window.show()
        self.close()

    # 核心新增3：槽函数，接收TrainDataWindow传递的样本数，并转发给主界面
    def on_train_data_window_closed(self, recorded_count):
        """
        接收训练数据采集窗口关闭时传递的样本数
        :param recorded_count: 已录制的训练样本数
        """
        logger.info(f"用户「{self.username}」训练数据采集窗口关闭，已录制{recorded_count}条样本，即将转发给主界面")
        # 发射自定义信号，将样本数传递给主界面
        self.train_data_finished.emit(recorded_count)