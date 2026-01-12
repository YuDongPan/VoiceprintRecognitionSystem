import os
import time
import random
import pyaudio
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton, QTextEdit, QListWidget, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from loguru import logger
import config as config
from audio_recorder import RecorderThread
from model_tester import TestThread

class TestWindow(QWidget):
    # 定义测试窗口关闭信号
    test_finished = pyqtSignal()  # 新增信号

    def __init__(self, username, test_mode, test_count, text_type):
        super().__init__()
        self.username = username
        self.test_mode = test_mode
        self.test_count = test_count
        self.text_type = text_type
        self.user_dir = os.path.join(config.DATASET_ROOT, username)
        self.target_test_files = []  # 目标用户文件（真值：匹配）
        self.non_target_test_files = []  # 其他被试文件（真值：不匹配）
        self.current_test_idx = 0
        self.record_thread = None
        self.test_thread_dict = {}
        self.trained_algos = self.get_trained_algorithms()
        self.shuffled_test_texts = config.TEST_TEXT_MATERIALS.copy()
        random.shuffle(self.shuffled_test_texts)
        self.init_ui()
        logger.info(
            f"用户「{username}」的测试窗口初始化完成\n"
            f"  测试方式：{self.test_mode}\n"
            f"  测试次数：{self.test_count}\n"
            f"  文本类型：{self.text_type}\n"
            f"  已训练算法：{self.trained_algos}\n"
            f"  测试素材已随机打乱"
        )

    def get_trained_algorithms(self):
        """获取已训练算法"""
        trained_algos = []
        sherpa_path = os.path.join(self.user_dir, "SHERPA_model.pkl")
        ecapa_path = os.path.join(self.user_dir, "ECAPA_model.pkl")
        if os.path.exists(sherpa_path):
            trained_algos.append("SHERPA")
        if os.path.exists(ecapa_path):
            trained_algos.append("ECAPA")
        return trained_algos

    def get_non_target_audio_files(self, required_count):
        """获取非目标被试音频（真值：不匹配）"""
        non_target_files = []
        if not os.path.exists(config.DATASET_ROOT):
            logger.error(f"数据集根目录不存在：{config.DATASET_ROOT}，无法获取非目标素材")
            return non_target_files

        # 遍历所有非当前用户的目录
        all_user_dirs = []
        for item in os.listdir(config.DATASET_ROOT):
            item_path = os.path.join(config.DATASET_ROOT, item)
            if os.path.isdir(item_path) and item != self.username:
                all_user_dirs.append(item_path)

        if not all_user_dirs:
            logger.warning(f"用户「{self.username}」测试时，未找到其他被试用户目录，无法获取非目标数据")
            return non_target_files

        # 收集非目标用户音频
        for other_user_dir in all_user_dirs:
            other_user_name = os.path.basename(other_user_dir)
            other_wav_files = []
            for f in os.listdir(other_user_dir):
                f_path = os.path.join(other_user_dir, f)
                if f.endswith(".wav") and os.path.isfile(f_path):
                    other_wav_files.append(f_path)
            if other_wav_files:
                non_target_files.extend(other_wav_files)
                logger.debug(f"从非目标用户「{other_user_name}」收集到 {len(other_wav_files)} 个wav文件")
            else:
                logger.debug(f"非目标用户「{other_user_name}」目录下无有效wav文件")

        # 随机选取指定数量
        if len(non_target_files) > 0:
            if len(non_target_files) >= required_count:
                non_target_files = random.sample(non_target_files, required_count)
                logger.success(f"用户「{self.username}」成功获取 {len(non_target_files)} 个非目标被试音频文件（需求{required_count}个）")
            else:
                logger.warning(f"用户「{self.username}」仅获取 {len(non_target_files)} 个非目标被试音频（需求{required_count}个，已取全部）")
        else:
            logger.warning(f"用户「{self.username}」未获取到任何非目标被试音频文件")
        return non_target_files

    def init_ui(self):
        self.setWindowTitle(f"声纹测试 - {self.username}（{self.test_mode}模式）")
        self.setMinimumSize(700, 600)
        main_layout = QVBoxLayout()

        # 1. 测试信息提示
        info_text = f"测试配置：{self.test_mode.capitalize()} Test | 文本类型：{self.text_type.capitalize()} | 已训练算法：{', '.join(self.trained_algos) if self.trained_algos else '无'}"
        info_label = QLabel(info_text)
        info_label.setFont(QFont("微软雅黑", 10, QFont.Bold))
        main_layout.addWidget(info_label)

        # 当前测试条数显示
        self.test_count_label = QLabel(f"当前测试：第 {self.current_test_idx + 1} 条 / 共 {self.test_count} 条")
        self.test_count_label.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.test_count_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.test_count_label)

        # 2. 测试文本显示组
        text_group = QGroupBox("请朗读以下文本：")
        text_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        text_layout = QVBoxLayout()
        self.test_text_display = QTextEdit()
        self.test_text_display.setReadOnly(self.text_type == "standard")
        self.test_text_display.setFont(QFont("微软雅黑", 12))
        self.test_text_display.setMinimumHeight(100)
        if self.text_type == "standard":
            self.test_text_display.setText(self.shuffled_test_texts[self.current_test_idx % len(self.shuffled_test_texts)])
        else:
            self.test_text_display.setPlaceholderText("请在此输入自由发言的文本（或直接朗读，保持4-5秒）")
        text_layout.addWidget(self.test_text_display)
        text_group.setLayout(text_layout)
        main_layout.addWidget(text_group)

        # 3. 录制控制组
        control_group = QGroupBox("录制控制")
        control_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        control_layout = QHBoxLayout()
        self.prev_test_btn = QPushButton("上一次")
        self.prev_test_btn.clicked.connect(self.prev_test)
        self.prev_test_btn.setEnabled(False)
        self.start_test_btn = QPushButton("开始录制测试语音")
        self.start_test_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.start_test_btn.clicked.connect(self.start_test_recording)
        self.stop_test_btn = QPushButton("停止录制")
        self.stop_test_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        self.stop_test_btn.clicked.connect(self.stop_test_recording)
        self.stop_test_btn.setEnabled(False)
        self.next_test_btn = QPushButton("下一次")
        self.next_test_btn.clicked.connect(self.next_test)
        self.next_test_btn.setEnabled(self.test_count > 1)
        control_layout.addWidget(self.prev_test_btn)
        control_layout.addWidget(self.start_test_btn)
        control_layout.addWidget(self.stop_test_btn)
        control_layout.addWidget(self.next_test_btn)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # 4. 进度和状态
        status_layout = QHBoxLayout()
        self.test_status_label = QLabel("当前状态：准备就绪")
        self.test_progress_label = QLabel(f"进度：0/{self.test_count}")
        status_layout.addWidget(self.test_status_label)
        status_layout.addWidget(self.test_progress_label)
        main_layout.addLayout(status_layout)

        # 5. 测试文件列表
        file_group = QGroupBox("测试文件列表")
        file_group.setFont(QFont("微软雅黑", 11, QFont.Bold))
        file_layout = QVBoxLayout()
        self.target_file_list = QListWidget()
        self.target_file_list.setMaximumHeight(100)
        self.non_target_file_list = QListWidget()
        self.non_target_file_list.setMaximumHeight(100)
        file_layout.addWidget(QLabel("目标用户文件（真值：匹配）："))
        file_layout.addWidget(self.target_file_list)
        file_layout.addWidget(QLabel("非目标被试文件（真值：不匹配）："))
        file_layout.addWidget(self.non_target_file_list)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # 6. 评估按钮
        self.start_evaluation_btn = QPushButton("开始评估所有已训练算法")
        self.start_evaluation_btn.setStyleSheet("padding: 10px; background-color: #2196F3; color: white;")
        self.start_evaluation_btn.clicked.connect(self.start_evaluation)
        self.start_evaluation_btn.setEnabled(False)
        main_layout.addWidget(self.start_evaluation_btn)

        # 7. 结果显示
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(150)
        self.result_text.setPlaceholderText("测试结果将显示在这里...")
        main_layout.addWidget(self.result_text)

        self.setLayout(main_layout)
        self.update_nav_buttons()

    def update_test_count_label(self):
        """更新测试条数"""
        self.test_count_label.setText(f"当前测试：第 {self.current_test_idx + 1} 条 / 共 {self.test_count} 条")
        logger.debug(f"用户「{self.username}」当前测试进度：第 {self.current_test_idx + 1} 条 / 共 {self.test_count} 条")

    def prev_test(self):
        """上一次测试"""
        if self.current_test_idx > 0:
            old_idx = self.current_test_idx
            self.current_test_idx -= 1
            self.update_test_text()
            self.update_test_count_label()
            self.update_nav_buttons()
            logger.info(f"用户「{self.username}」将测试序号从 {old_idx + 1} 切换为 {self.current_test_idx + 1}")

    def next_test(self):
        """下一次测试"""
        if self.current_test_idx < self.test_count - 1:
            old_idx = self.current_test_idx
            self.current_test_idx += 1
            self.update_test_text()
            self.update_test_count_label()
            self.update_nav_buttons()
            logger.info(f"用户「{self.username}」将测试序号从 {old_idx + 1} 切换为 {self.current_test_idx + 1}")

    def update_test_text(self):
        """更新测试文本"""
        if self.text_type == "standard":
            text_idx = self.current_test_idx % len(self.shuffled_test_texts)
            self.test_text_display.setText(self.shuffled_test_texts[text_idx])
            logger.debug(f"用户「{self.username}」更新测试文本为打乱后列表的第 {text_idx + 1} 条标准文本")

    def update_nav_buttons(self):
        """更新导航按钮状态"""
        self.prev_test_btn.setEnabled(self.current_test_idx > 0)
        self.next_test_btn.setEnabled(self.current_test_idx < self.test_count - 1)

    def start_test_recording(self):
        """开始录制测试音频"""
        timestamp = int(time.time())
        file_name = f"test_{self.current_test_idx + 1}_{timestamp}.wav"
        save_path = os.path.join(self.user_dir, file_name)
        duration = 4 if (self.current_test_idx % 2 == 0) else 5

        self.record_thread = RecorderThread(save_path, duration, config.SAMPLE_RATE, config.CHANNELS, pyaudio.paInt16)
        self.record_thread.status_signal.connect(self.update_test_status)
        self.record_thread.finish_signal.connect(self.on_test_recording_finished)
        self.record_thread.start()

        self.start_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(True)
        self.prev_test_btn.setEnabled(False)
        self.next_test_btn.setEnabled(False)
        logger.info(f"用户「{self.username}」开始录制第 {self.current_test_idx + 1} 次测试音频，保存路径：{save_path}，时长：{duration} 秒")

    def stop_test_recording(self):
        """停止录制"""
        if self.record_thread and self.record_thread.is_recording:
            self.record_thread.stop_recording()
            logger.info(f"用户「{self.username}」手动停止了第 {self.current_test_idx + 1} 次测试音频录制")

    def update_test_status(self, status):
        """更新测试状态"""
        self.test_status_label.setText(f"当前状态：{status}")
        logger.debug(f"用户「{self.username}」第 {self.current_test_idx + 1} 次测试录制状态：{status}")

    def on_test_recording_finished(self, file_path):
        """录制完成回调"""
        if file_path not in self.target_test_files:
            if len(self.target_test_files) > self.current_test_idx:
                self.target_test_files[self.current_test_idx] = file_path
            else:
                self.target_test_files.append(file_path)
        self.refresh_test_file_lists()
        self.test_progress_label.setText(f"进度：{len(self.target_test_files)}/{self.test_count}")

        self.start_test_btn.setEnabled(True)
        self.stop_test_btn.setEnabled(False)
        self.update_nav_buttons()

        # 加载非目标素材
        if len(self.target_test_files) >= self.test_count:
            self.non_target_test_files = self.get_non_target_audio_files(self.test_count)
            self.refresh_test_file_lists()
            self.start_evaluation_btn.setEnabled(True)
            logger.info(f"用户「{self.username}」已完成 {len(self.target_test_files)} 次测试音频录制，已加载 {len(self.non_target_test_files)} 个非目标素材，可启动模型评估")
        else:
            if self.current_test_idx < self.test_count - 1:
                self.next_test()

        if "录制完成" in self.test_status_label.text():
            logger.success(f"用户「{self.username}」第 {self.current_test_idx + 1 if self.current_test_idx < self.test_count else self.test_count} 次测试音频录制成功，文件：{file_path}")
        else:
            logger.warning(f"用户「{self.username}」第 {self.current_test_idx + 1} 次测试音频录制被取消")

    def refresh_test_file_lists(self):
        """刷新测试文件列表"""
        # 目标文件
        self.target_file_list.clear()
        for i, file_path in enumerate(self.target_test_files):
            self.target_file_list.addItem(f"测试{i + 1}: {os.path.basename(file_path)}（真值：匹配）")
        # 非目标文件
        self.non_target_file_list.clear()
        if self.non_target_test_files:
            for i, file_path in enumerate(self.non_target_test_files):
                other_user = os.path.basename(os.path.dirname(file_path))
                self.non_target_file_list.addItem(f"非目标{i + 1}（用户：{other_user}）: {os.path.basename(file_path)}（真值：不匹配）")
        else:
            self.non_target_file_list.addItem("暂无可用非目标被试音频素材")
        logger.debug(f"用户「{self.username}」刷新测试文件列表：目标文件{len(self.target_test_files)}个，非目标文件{len(self.non_target_test_files)}个")

    def start_evaluation(self):
        """启动模型评估（明确真值标签）"""
        if len(self.target_test_files) < self.test_count:
            warning_msg = f"需要{self.test_count}个目标测试文件，当前只有{len(self.target_test_files)}个"
            QMessageBox.warning(self, "测试文件不足", warning_msg, QMessageBox.Ok)
            logger.warning(f"用户「{self.username}」启动评估失败：{warning_msg}")
            return

        # 禁用按钮
        self.start_test_btn.setEnabled(False)
        self.stop_test_btn.setEnabled(False)
        self.prev_test_btn.setEnabled(False)
        self.next_test_btn.setEnabled(False)
        self.start_evaluation_btn.setEnabled(False)

        # 合并目标和非目标文件
        final_test_files = self.target_test_files.copy()
        final_test_files.extend(self.non_target_test_files)
        logger.info(f"用户「{self.username}」合并测试文件：目标{len(self.target_test_files)}个（匹配） + 非目标{len(self.non_target_test_files)}个（不匹配） = 总计{len(final_test_files)}个")

        # 生成结果
        self.result_text.clear()
        self.result_text.append("=== 声纹识别多算法测试报告 ===\n")
        self.result_text.append(f"目标用户：{self.username}（测试文件真值：匹配）")
        if self.non_target_test_files:
            self.result_text.append(f"非目标被试：{len(self.non_target_test_files)} 个（测试文件真值：不匹配）\n")
        else:
            self.result_text.append("未引入非目标被试素材（无可用非目标用户或音频）\n")
        self.result_text.moveCursor(self.result_text.textCursor().End)

        # 测试已训练算法
        for algo in self.trained_algos:
            self.result_text.append(f"\n----- {algo} 算法测试 -----")
            self.result_text.moveCursor(self.result_text.textCursor().End)
            # 传入目标用户名，方便TestThread判断真值
            test_thread = TestThread(self.username, algo, final_test_files)
            test_thread.status_signal.connect(self.update_test_status)
            test_thread.progress_signal.connect(lambda v, a=algo: self.update_algo_test_progress(v, a))
            test_thread.finish_signal.connect(lambda res, a=algo: self.on_algo_evaluation_finished(res, a))
            test_thread.start()
            self.test_thread_dict[algo] = test_thread
            logger.info(f"用户「{self.username}」启动 {algo} 算法测试，测试文件总数：{len(final_test_files)}")

        # 未训练算法提示
        all_algos = ["SHERPA", "ECAPA"]
        untrained_algos = [a for a in all_algos if a not in self.trained_algos]
        if untrained_algos:
            self.result_text.append(f"\n----- 未训练算法提示 -----")
            for algo in untrained_algos:
                self.result_text.append(f"{algo} 算法：尚未训练，无法进行测试")
            self.result_text.moveCursor(self.result_text.textCursor().End)
            logger.info(f"用户「{self.username}」未训练算法：{', '.join(untrained_algos)}")

    def update_algo_test_progress(self, value, algo):
        """更新算法测试进度"""
        self.test_status_label.setText(f"正在评估 {algo} 算法：{value}%")
        logger.debug(f"用户「{self.username}」{algo} 算法测试进度：{value}%")

    def on_algo_evaluation_finished(self, results, algo):
        """算法评估完成回调（核心修改：区分ECAPA算法，展示相似度和阈值）"""
        if results["success"]:
            report = f"\n{algo} 算法测试结果："
            report += f"\n  测试总数：{results['total']}（目标{len(self.target_test_files)}个（匹配） + 非目标{len(self.non_target_test_files)}个（不匹配））"
            report += f"\n  识别正确：{results['correct']}"
            report += f"\n  准确率：{results['accuracy']:.2%}"
            report += f"\n  详细结果："
            for i, detail in enumerate(results["details"]):
                # 基础信息：文件名、真值、预测值、测试结果
                base_info = f"\n    样本{i + 1} ({detail['file']}): 真值={detail['true_label']} | 预测={detail['pred_label']} | 结果={detail['result']}"
                # 核心修改：判断是否为ECAPA算法，若是则补充相似度和阈值
                if "similarity" in detail and "threshold" in detail:
                    # 补充SHERPA和ECAPA相似度（保留4位小数）和阈值（保留4位小数）
                    ecapa_extra = f" | 相似度={detail['similarity']:.4f} | 阈值={detail['threshold']:.4f}"
                    base_info += ecapa_extra
                # 拼接最终信息
                report += base_info
            self.result_text.append(report)
            self.test_status_label.setText(f"{algo} 算法评估完成")
            logger.success(f"用户「{self.username}」{algo} 算法评估完成，准确率：{results['accuracy']:.2%}，正确数：{results['correct']}/{results['total']}")
        else:
            error_msg = f"\n{algo} 算法评估失败：{results['message']}"
            self.result_text.append(error_msg)
            self.test_status_label.setText(f"{algo} 算法评估失败")
            logger.error(f"用户「{self.username}」{algo} 算法评估失败：{results['message']}")
        self.result_text.moveCursor(self.result_text.textCursor().End)

        # 所有算法完成后恢复按钮
        all_finished = all([not thread.isRunning() for thread in self.test_thread_dict.values()])
        if all_finished:
            self.start_test_btn.setEnabled(True)
            self.update_nav_buttons()
            logger.success(f"用户「{self.username}」所有已训练算法测试完成（已明确真值标签评估）")

    # 重写关闭事件，发射信号
    def closeEvent(self, event):
        self.test_finished.emit()  # 窗口关闭时发射信号
        event.accept()