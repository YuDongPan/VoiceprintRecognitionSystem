import os
import time
import random
import pyaudio
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QTextEdit, QListWidget, \
    QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from loguru import logger
import config as config
from audio_recorder import RecorderThread


class TrainDataWindow(QWidget):
    # 扩展信号：携带已录制训练样本数
    closed = pyqtSignal(int)

    def __init__(self, username, algorithm, train_count):
        super().__init__()
        self.username = username
        self.algorithm = algorithm
        self.user_dir = os.path.join(config.DATASET_ROOT, username)
        self.train_count = train_count
        self.current_text_idx = 0
        self.record_thread = None
        self.shuffled_train_texts = config.TRAIN_TEXT_MATERIALS.copy()
        random.shuffle(self.shuffled_train_texts)
        self.is_finished_prompted = False  # 弹窗标志位
        logger.info(f"用户「{username}」的训练数据采集窗口初始化完成，采集数量：{train_count}条，训练素材已随机打乱")
        self.init_ui()
        # 初始化时刷新一次数量（避免已有文件不判断）
        self.check_if_train_finished()

    def init_ui(self):
        self.setWindowTitle(f"采集训练数据 - {self.username}")
        self.setMinimumSize(600, 500)

        main_layout = QVBoxLayout()

        # 说明文本
        info_label = QLabel(
            f"请朗读以下文本进行训练数据采集（已设定采集{self.train_count}条，{self.algorithm}算法需要至少{3 if self.algorithm == 'SHERPA' else 3}条数据）")
        info_label.setFont(QFont("微软雅黑", 10))
        main_layout.addWidget(info_label)

        # 当前条数显示
        self.count_label = QLabel(f"当前采集：第 {self.current_text_idx + 1} 条 / 共 {self.train_count} 条")
        self.count_label.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.count_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.count_label)

        # 文本显示组
        text_group = QGroupBox("请朗读以下文本：")
        text_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        text_layout = QVBoxLayout()
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("微软雅黑", 12))
        self.text_display.setMinimumHeight(100)
        self.text_display.setText(self.shuffled_train_texts[self.current_text_idx])
        text_layout.addWidget(self.text_display)
        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)

        # 录制控制组
        control_group = QGroupBox("录制控制")
        control_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        control_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一条")
        self.prev_btn.clicked.connect(self.prev_text)
        self.prev_btn.setEnabled(False)
        self.start_btn = QPushButton("开始录制")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn = QPushButton("停止录制")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_recording)
        self.stop_btn.setEnabled(False)
        self.next_btn = QPushButton("下一条")
        self.next_btn.clicked.connect(self.next_text)
        self.next_btn.setEnabled(self.train_count > 1)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.next_btn)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # 状态标签
        self.status_label = QLabel("当前状态：准备就绪")
        main_layout.addWidget(self.status_label)

        # 已录制列表组
        list_group = QGroupBox("已录制音频")
        list_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        list_layout = QVBoxLayout()
        self.audio_list = QListWidget()
        list_layout.addWidget(self.audio_list)
        list_group.setLayout(list_layout)
        main_layout.addWidget(list_group)

        self.setLayout(main_layout)
        self.refresh_audio_list()

    def update_count_label(self):
        """更新当前条数显示"""
        self.count_label.setText(f"当前采集：第 {self.current_text_idx + 1} 条 / 共 {self.train_count} 条")
        logger.debug(
            f"用户「{self.username}」当前训练采集进度：第 {self.current_text_idx + 1} 条 / 共 {self.train_count} 条")

    def prev_text(self):
        """上一条文本"""
        if self.current_text_idx > 0:
            old_idx = self.current_text_idx
            self.current_text_idx -= 1
            self.text_display.setText(self.shuffled_train_texts[self.current_text_idx])
            self.next_btn.setEnabled(True)
            if self.current_text_idx == 0:
                self.prev_btn.setEnabled(False)
            self.update_count_label()
            logger.info(f"用户「{self.username}」将训练文本从第 {old_idx + 1} 条切换为第 {self.current_text_idx + 1} 条")
            # 切换文本后也检查是否完成
            self.check_if_train_finished()

    def next_text(self):
        """下一条文本"""
        if self.current_text_idx < self.train_count - 1:
            old_idx = self.current_text_idx
            self.current_text_idx += 1
            self.text_display.setText(self.shuffled_train_texts[self.current_text_idx])
            self.prev_btn.setEnabled(True)
            if self.current_text_idx == self.train_count - 1:
                self.next_btn.setEnabled(False)
            self.update_count_label()
            logger.info(f"用户「{self.username}」将训练文本从第 {old_idx + 1} 条切换为第 {self.current_text_idx + 1} 条")
            # 切换文本后也检查是否完成
            self.check_if_train_finished()

    def start_recording(self):
        """开始录制"""
        timestamp = int(time.time())
        file_name = f"train_{self.current_text_idx + 1}_{timestamp}.wav"
        save_path = os.path.join(self.user_dir, file_name)
        text = self.shuffled_train_texts[self.current_text_idx]
        duration = 4 if "约 4 秒" in text else 5
        logger.info(
            f"用户「{self.username}」开始录制训练音频（第 {self.current_text_idx + 1} 条），保存路径：{save_path}，录制时长：{duration} 秒")

        self.record_thread = RecorderThread(save_path, duration, config.SAMPLE_RATE, config.CHANNELS, pyaudio.paInt16)
        self.record_thread.status_signal.connect(self.update_status)
        self.record_thread.finish_signal.connect(self.on_recording_finished)
        self.record_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def stop_recording(self):
        """停止录制"""
        if self.record_thread and self.record_thread.is_recording:
            self.record_thread.stop_recording()
            logger.info(f"用户「{self.username}」手动停止了训练音频录制（第 {self.current_text_idx + 1} 条）")

    def update_status(self, status):
        """更新状态"""
        self.status_label.setText(f"当前状态：{status}")
        logger.debug(f"用户「{self.username}」训练音频录制状态（第 {self.current_text_idx + 1} 条）：{status}")

    def on_recording_finished(self, file_path):
        """录制完成回调"""
        self.refresh_audio_list()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.prev_btn.setEnabled(self.current_text_idx > 0)
        self.next_btn.setEnabled(self.current_text_idx < self.train_count - 1)

        if self.current_text_idx < self.train_count - 1:
            self.next_text()
        else:
            # 最后一条录制完成后，强制检查是否满足数量
            self.check_if_train_finished()

        if "录制完成" in self.status_label.text():
            logger.success(
                f"用户「{self.username}」训练音频录制成功（第 {self.current_text_idx + 1 if self.current_text_idx < self.train_count else self.train_count} 条），文件已保存：{file_path}")
        else:
            logger.warning(f"用户「{self.username}」训练音频录制被取消（第 {self.current_text_idx + 1} 条），未生成有效文件")

        # 录制完成后必检
        self.check_if_train_finished()

    def refresh_audio_list(self):
        """刷新音频列表"""
        self.audio_list.clear()
        wav_files = sorted([f for f in os.listdir(self.user_dir) if f.endswith(".wav") and f.startswith("train_")])
        for f in wav_files:
            self.audio_list.addItem(f)
        logger.debug(f"用户「{self.username}」刷新训练音频列表，当前已录制 {len(wav_files)} 个训练文件")

    def get_recorded_train_count(self):
        """获取已录制的训练样本数"""
        if not os.path.exists(self.user_dir):
            return 0
        wav_files = [f for f in os.listdir(self.user_dir) if f.endswith(".wav") and f.startswith("train_")]
        return len(wav_files)

    def check_if_train_finished(self):
        """核心新增：独立的完成判断方法，多处调用确保触发"""
        recorded_count = self.get_recorded_train_count()
        # 弹窗条件：数量达标 + 未提示过
        if recorded_count >= self.train_count and not self.is_finished_prompted:
            # 强制弹窗（绑定当前窗口，确保置顶）
            QMessageBox.information(
                self,
                "录制完成",
                f"恭喜！已成功录制{recorded_count}条训练语音（满足{self.train_count}条的采集要求）\n您可以关闭当前窗口返回主界面，进行模型训练啦！",
                QMessageBox.Ok
            )
            self.is_finished_prompted = True  # 标记已提示，避免重复弹窗
            logger.success(f"用户「{self.username}」训练数据采集完成（{recorded_count}条），已弹出提示框")

    def closeEvent(self, event):
        """窗口关闭事件：发射携带样本数的信号"""
        if self.record_thread and self.record_thread.is_recording:
            self.record_thread.stop_recording()
            logger.warning(f"用户「{self.username}」关闭采集窗口时，强制停止了正在进行的录制")

        recorded_train_count = self.get_recorded_train_count()
        self.closed.emit(recorded_train_count)
        logger.info(f"用户「{self.username}」关闭了训练数据采集窗口（已采集{recorded_train_count}条训练样本）")
        event.accept()