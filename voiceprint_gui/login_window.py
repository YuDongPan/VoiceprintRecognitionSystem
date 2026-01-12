import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from loguru import logger
import config as config

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        logger.info("登录窗口已初始化完成")

    def init_ui(self):
        self.setWindowTitle("声纹识别系统 - 登录")
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(50, 30, 50, 30)

        # 标题
        title_label = QLabel("声纹识别系统")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 用户名输入
        user_layout = QHBoxLayout()
        user_label = QLabel("用户名:")
        user_label.setFont(QFont("微软雅黑", 10))
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("请输入您的用户名")
        user_layout.addWidget(user_label)
        user_layout.addWidget(self.user_edit)
        layout.addLayout(user_layout)

        # 登录按钮
        btn_layout = QHBoxLayout()
        self.login_btn = QPushButton("登录")
        self.login_btn.setFont(QFont("微软雅黑", 10))
        self.login_btn.setStyleSheet("padding: 8px;")
        self.login_btn.clicked.connect(self.login)
        btn_layout.addWidget(self.login_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def login(self):
        """登录逻辑：验证用户名+创建用户文件夹"""
        username = self.user_edit.text().strip()
        logger.info(f"用户尝试登录，输入的用户名为：{username if username else '空值'}")

        if not username:
            warning_msg = "登录失败：请输入用户名"
            QMessageBox.warning(self, "警告", warning_msg, QMessageBox.Ok)
            logger.warning(warning_msg)
            return

        # 创建用户文件夹
        user_dir = os.path.join(config.DATASET_ROOT, username)
        os.makedirs(user_dir, exist_ok=True)

        logger.success(f"用户「{username}」登录成功，用户文件夹路径：{user_dir}")

        from voiceprint_gui.main_window import MainWindow
        # 打开主窗口并关闭登录窗口
        self.main_window = MainWindow(username)
        self.main_window.show()
        self.close()