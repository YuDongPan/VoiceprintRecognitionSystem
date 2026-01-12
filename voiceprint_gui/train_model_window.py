import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QTextEdit, QPushButton, QMessageBox, QGroupBox
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from loguru import logger
from model_trainer import TrainThread

class TrainModelWindow(QWidget):
    closed = pyqtSignal()

    def __init__(self, username, algorithm):
        super().__init__()
        self.username = username
        self.algorithm = algorithm
        self.train_thread = None
        logger.info(f"用户「{username}」的 {algorithm} 模型训练窗口初始化完成")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"训练{self.algorithm}模型 - {self.username}")
        self.setFixedSize(500, 300)

        main_layout = QVBoxLayout()

        # 标题
        title_label = QLabel(f"正在训练{self.algorithm}声纹模型...")
        title_label.setFont(QFont("微软雅黑", 12, QFont.Bold))
        main_layout.addWidget(title_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # 状态标签
        self.status_label = QLabel("准备开始训练...")
        main_layout.addWidget(self.status_label)

        # 日志组
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # 取消按钮
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("取消训练")
        self.cancel_btn.clicked.connect(self.cancel_train)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
        self.start_train()

    def start_train(self):
        """启动训练线程"""
        self.train_thread = TrainThread(self.username, self.algorithm)
        self.train_thread.status_signal.connect(self.update_log)
        self.train_thread.progress_signal.connect(self.update_progress)
        self.train_thread.finish_signal.connect(self.on_train_finished)
        self.train_thread.start()
        logger.info(f"用户「{self.username}」启动了 {self.algorithm} 模型训练线程")

    def update_log(self, message):
        """更新日志"""
        self.status_label.setText(message)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_text.moveCursor(self.log_text.textCursor().End)
        logger.info(f"用户「{self.username}」{self.algorithm} 模型训练过程：{message}")

    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)
        logger.debug(f"用户「{self.username}」{self.algorithm} 模型训练进度：{value}%")

    def on_train_finished(self, success, message):
        """训练完成回调"""
        self.update_log(message)
        self.cancel_btn.setText("关闭")
        if success:
            QMessageBox.information(self, "训练完成", message, QMessageBox.Ok)
            logger.success(f"用户「{self.username}」{self.algorithm} 模型训练成功：{message}")
        else:
            QMessageBox.warning(self, "训练失败", message, QMessageBox.Ok)
            logger.error(f"用户「{self.username}」{self.algorithm} 模型训练失败：{message}")

    def cancel_train(self):
        """取消训练/关闭窗口"""
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.terminate()
            cancel_msg = "训练已取消"
            self.update_log(cancel_msg)
            self.cancel_btn.setText("关闭")
            logger.warning(f"用户「{self.username}」手动取消了 {self.algorithm} 模型训练")
        else:
            self.close()

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.train_thread and self.train_thread.isRunning():
            reply = QMessageBox.question(self, "确认", "训练正在进行中，确定要关闭吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.train_thread.terminate()
                self.closed.emit()
                logger.warning(f"用户「{self.username}」强制关闭训练窗口，终止了正在进行的 {self.algorithm} 模型训练")
                event.accept()
            else:
                logger.info(f"用户「{self.username}」取消了关闭训练窗口的操作，训练继续进行")
                event.ignore()
        else:
            self.closed.emit()
            logger.info(f"用户「{self.username}」关闭了 {self.algorithm} 模型训练窗口")
            event.accept()