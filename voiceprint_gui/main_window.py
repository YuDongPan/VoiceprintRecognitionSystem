import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QRadioButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from loguru import logger
import config as config


class MainWindow(QWidget):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.user_dir = os.path.join(config.DATASET_ROOT, username)
        self.algorithm = "SHERPA"
        logger.info(f"「{username}」的主窗口已初始化，默认算法：{self.algorithm}")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(f"声纹识别系统 - {self.username}")
        self.setMinimumSize(600, 400)

        # 主布局
        main_layout = QVBoxLayout()

        # 欢迎信息
        welcome_label = QLabel(f"欢迎回来，{self.username}！")
        welcome_label.setFont(QFont("微软雅黑", 14, QFont.Bold))
        main_layout.addWidget(welcome_label)

        # 算法选择组
        algo_group = QGroupBox("选择声纹识别算法")
        algo_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        algo_layout = QVBoxLayout()
        self.sherpa_radio = QRadioButton("SHERPA算法 (需要至少3个训练样本)")
        self.ecapa_radio = QRadioButton("ECAPA算法 (需要至少3个训练样本)")
        self.sherpa_radio.setChecked(True)
        self.sherpa_radio.clicked.connect(self.on_algo_changed)
        self.ecapa_radio.clicked.connect(self.on_algo_changed)
        algo_layout.addWidget(self.sherpa_radio)
        algo_layout.addWidget(self.ecapa_radio)
        algo_group.setLayout(algo_layout)
        main_layout.addWidget(algo_group)

        # 数据状态组
        status_group = QGroupBox("当前数据状态")
        status_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        status_layout = QVBoxLayout()
        self.data_status = QLabel(self.get_data_status_text())
        status_layout.addWidget(self.data_status)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # 模型状态组
        model_group = QGroupBox("模型状态")
        model_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        model_layout = QVBoxLayout()
        self.model_status = QLabel(self.get_model_status_text())
        model_layout.addWidget(self.model_status)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # 功能按钮组
        core_btn_layout = QHBoxLayout()
        self.train_data_btn = QPushButton("采集训练数据")
        self.train_data_btn.setStyleSheet("padding: 10px; background-color: #4CAF50; color: white;")
        self.train_data_btn.clicked.connect(self.open_train_data_window)
        self.train_model_btn = QPushButton("训练模型")
        self.train_model_btn.setStyleSheet("padding: 10px; background-color: #2196F3; color: white;")
        self.train_model_btn.clicked.connect(self.open_train_model_window)
        self.test_btn = QPushButton("测试模型")
        self.test_btn.setStyleSheet("padding: 10px; background-color: #ff9800; color: white;")
        self.test_btn.clicked.connect(self.open_test_mode_select_window)
        core_btn_layout.addWidget(self.train_data_btn)
        core_btn_layout.addWidget(self.train_model_btn)
        core_btn_layout.addWidget(self.test_btn)

        # 退出登录按钮布局
        logout_btn_layout = QHBoxLayout()
        logout_btn_layout.setAlignment(Qt.AlignRight)
        self.logout_btn = QPushButton("退出登录")
        self.logout_btn.setStyleSheet("""
            padding: 10px 20px; 
            background-color: #f44336; 
            color: white; 
            border: none;
            border-radius: 4px;
        """)
        self.logout_btn.clicked.connect(self.on_logout_clicked)
        logout_btn_layout.addWidget(self.logout_btn)

        # 添加按钮布局到主布局
        main_layout.addLayout(core_btn_layout)
        main_layout.addLayout(logout_btn_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_algo_changed(self):
        """算法切换逻辑"""
        old_algo = self.algorithm
        if self.sherpa_radio.isChecked():
            self.algorithm = "SHERPA"
        else:
            self.algorithm = "ECAPA"
        logger.info(f"用户「{self.username}」将识别算法从 {old_algo} 切换为 {self.algorithm}")
        self.data_status.setText(self.get_data_status_text())
        self.model_status.setText(self.get_model_status_text())

    def get_data_status_text(self):
        """获取数据状态文本"""
        wav_count = len([f for f in os.listdir(self.user_dir) if f.endswith(".wav")])
        required = 3 if self.algorithm == "SHERPA" else 3
        status = "足够" if wav_count >= required else "不足"
        logger.debug(f"用户「{self.username}」数据状态：当前样本数 {wav_count}，{self.algorithm} 算法需要 {required} 个，状态 {status}")
        return f"当前训练样本数: {wav_count} 个，{self.algorithm}算法需要至少 {required} 个样本，当前样本{status}"

    def get_model_status_text(self):
        """获取模型状态文本"""
        sherpa_model_path = os.path.join(self.user_dir, "SHERPA_model.pkl")
        ecapa_model_path = os.path.join(self.user_dir, "ECAPA_model.pkl")
        sherpa_status = "已训练" if os.path.exists(sherpa_model_path) else "未训练"
        ecapa_status = "已训练" if os.path.exists(ecapa_model_path) else "未训练"
        model_status_text = f"SHERPA模型：{sherpa_status} | ECAPA模型：{ecapa_status}"
        logger.debug(f"用户「{self.username}」模型状态：SHERPA={sherpa_status}，ECAPA={ecapa_status}")
        return model_status_text

    def open_train_data_window(self):
        """打开训练数量设置窗口"""
        from voiceprint_gui.train_count_setting import TrainCountSettingWindow
        self.train_count_window = TrainCountSettingWindow(self.username, self.algorithm)
        # 绑定TrainCountSettingWindow的自定义信号到主界面的refresh_status方法
        self.train_count_window.train_data_finished.connect(self.refresh_status)
        self.train_count_window.show()
        logger.info(f"用户「{self.username}」打开了训练音频数量设置窗口")

    def open_train_model_window(self):
        """打开模型训练窗口"""
        wav_count = len([f for f in os.listdir(self.user_dir) if f.endswith(".wav")])
        required = 3 if self.algorithm == "SHERPA" else 3
        if wav_count < required:
            warning_msg = f"{self.algorithm}算法需要至少{required}个训练样本，当前只有{wav_count}个，请先采集更多数据"
            QMessageBox.warning(self, "数据不足", warning_msg, QMessageBox.Ok)
            logger.warning(f"用户「{self.username}」打开训练窗口失败：{warning_msg}")
            return

        from voiceprint_gui.train_model_window import TrainModelWindow
        self.train_model_window = TrainModelWindow(self.username, self.algorithm)
        self.train_model_window.show()
        self.train_model_window.closed.connect(self.refresh_status)
        logger.info(f"用户「{self.username}」打开了 {self.algorithm} 模型训练窗口，当前样本数：{wav_count}")

    def open_test_mode_select_window(self):
        """打开测试方式选择窗口"""
        sherpa_model_path = os.path.join(self.user_dir, "SHERPA_model.pkl")
        ecapa_model_path = os.path.join(self.user_dir, "ECAPA_model.pkl")
        has_trained_model = os.path.exists(sherpa_model_path) or os.path.exists(ecapa_model_path)
        if not has_trained_model:
            warning_msg = "尚未训练任何模型（SHERPA/ECAPA），请先训练至少一个模型再进行测试"
            QMessageBox.warning(self, "模型不存在", warning_msg, QMessageBox.Ok)
            logger.warning(f"用户「{self.username}」打开测试方式选择窗口失败：{warning_msg}")
            return

        from voiceprint_gui.test_mode_select import TestModeSelectWindow
        self.test_mode_window = TestModeSelectWindow(self.username)
        # 连接测试模式窗口的转发信号到主窗口的刷新方法
        self.test_mode_window.test_window_closed.connect(self.refresh_status)  # 新增
        self.test_mode_window.show()
        logger.info(f"用户「{self.username}」打开了测试方式选择窗口")

    def refresh_status(self):
        """刷新数据状态和模型状态"""
        logger.debug(f"用户「{self.username}」刷新了数据状态和模型状态")
        self.data_status.setText(self.get_data_status_text())
        self.model_status.setText(self.get_model_status_text())

    def on_logout_clicked(self):
        """退出登录逻辑"""
        logger.info(f"用户「{self.username}」触发了退出登录操作")
        reply = QMessageBox.question(
            self,
            "确认退出",
            f"你确定要退出当前账号「{self.username}」吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 关闭所有子窗口
            try:
                if hasattr(self, 'train_count_window') and self.train_count_window.isVisible():
                    self.train_count_window.close()
                if hasattr(self, 'train_data_window') and self.train_data_window.isVisible():
                    self.train_data_window.close()
                if hasattr(self, 'train_model_window') and self.train_model_window.isVisible():
                    self.train_model_window.close()
                if hasattr(self, 'test_mode_window') and self.test_mode_window.isVisible():
                    self.test_mode_window.close()
                if hasattr(self, 'test_window') and self.test_window.isVisible():
                    self.test_window.close()
            except Exception as e:
                logger.error(f"用户「{self.username}」退出时，关闭子窗口异常：{str(e)}")

            # 关闭主窗口，打开登录窗口
            from voiceprint_gui.login_window import LoginWindow
            self.close()
            self.login_window = LoginWindow()
            self.login_window.show()
            logger.success(f"用户「{self.username}」退出登录成功，已返回登录界面")
        else:
            logger.info(f"用户「{self.username}」取消了退出登录操作")