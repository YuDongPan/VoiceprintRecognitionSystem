from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton, QSpinBox, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal
from loguru import logger

class TestModeSelectWindow(QWidget):
    # 定义转发测试完成的信号
    test_window_closed = pyqtSignal()  # 新增信号

    def __init__(self, username):
        super().__init__()
        self.username = username
        self.test_mode = "single"
        self.test_count = 3
        self.text_type = "standard"
        self.init_ui()
        logger.info(f"用户「{username}」的测试方式选择窗口初始化完成")

    def init_ui(self):
        self.setWindowTitle(f"选择测试方式 - {self.username}")
        self.setFixedSize(450, 350)
        main_layout = QVBoxLayout()

        # 1. 测试方式选择组
        mode_group = QGroupBox("选择测试方式")
        mode_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        mode_layout = QVBoxLayout()
        self.single_radio = QRadioButton("单次测试（录制1条音频，测试所有已训练算法）")
        self.multi_radio = QRadioButton("多次测试（录制3-5条音频，引入非目标被试数据）")
        self.single_radio.setChecked(True)
        self.single_radio.clicked.connect(self.on_test_mode_changed)
        self.multi_radio.clicked.connect(self.on_test_mode_changed)
        mode_layout.addWidget(self.single_radio)
        mode_layout.addWidget(self.multi_radio)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # 2. 多次测试数量设置
        count_group = QGroupBox("多次测试数量（3-5次）")
        count_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        count_layout = QHBoxLayout()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(3, 5)
        self.count_spin.setValue(3)
        self.count_spin.setEnabled(False)
        count_layout.addWidget(QLabel("测试次数："))
        count_layout.addWidget(self.count_spin)
        count_group.setLayout(count_layout)
        main_layout.addWidget(count_group)

        # 3. 文本类型选择组
        text_group = QGroupBox("选择语音文本类型")
        text_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        text_layout = QVBoxLayout()
        self.standard_text_radio = QRadioButton("使用系统提供的测试素材（推荐）")
        self.custom_text_radio = QRadioButton("自由发言（保持4-5秒自然朗读）")
        self.standard_text_radio.setChecked(True)
        self.standard_text_radio.clicked.connect(self.on_text_type_changed)
        self.custom_text_radio.clicked.connect(self.on_text_type_changed)
        text_layout.addWidget(self.standard_text_radio)
        text_layout.addWidget(self.custom_text_radio)
        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)

        # 4. 确认按钮
        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("确认并进入测试")
        self.confirm_btn.setStyleSheet("padding: 10px; background-color: #2196F3; color: white;")
        self.confirm_btn.clicked.connect(self.confirm_test_mode)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setStyleSheet("padding: 10px; background-color: #f44336; color: white;")
        self.cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def on_test_mode_changed(self):
        """测试方式切换"""
        if self.single_radio.isChecked():
            self.test_mode = "single"
            self.count_spin.setEnabled(False)
        else:
            self.test_mode = "multi"
            self.count_spin.setEnabled(True)
        self.test_count = self.count_spin.value()
        logger.info(f"用户「{self.username}」选择测试方式：{self.test_mode}，测试次数：{self.test_count}")

    def on_text_type_changed(self):
        """文本类型切换"""
        self.text_type = "standard" if self.standard_text_radio.isChecked() else "custom"
        logger.info(f"用户「{self.username}」选择文本类型：{self.text_type}")

    def confirm_test_mode(self):
        """确认测试方式，打开测试窗口"""
        from voiceprint_gui.test_window import TestWindow
        self.test_count = self.count_spin.value() if self.test_mode == "multi" else 1
        self.test_window = TestWindow(
            username=self.username,
            test_mode=self.test_mode,
            test_count=self.test_count,
            text_type=self.text_type
        )
        # 连接测试窗口的关闭信号到当前窗口的转发信号
        self.test_window.test_finished.connect(self.test_window_closed.emit)  # 新增
        self.test_window.show()
        self.close()
        logger.success(
            f"用户「{self.username}」确认测试配置：方式={self.test_mode}，次数={self.test_count}，文本类型={self.text_type}")