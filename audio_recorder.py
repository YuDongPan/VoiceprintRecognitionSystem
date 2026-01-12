import time
import pyaudio
import wave
from PyQt5.QtCore import QThread, pyqtSignal
from loguru import logger

class RecorderThread(QThread):
    status_signal = pyqtSignal(str)  # 录制状态信号
    finish_signal = pyqtSignal(str)  # 录制完成信号（返回保存路径）

    def __init__(self, save_path, record_seconds=5, rate=16000, channels=1, format=pyaudio.paInt16):
        super().__init__()
        self.save_path = save_path      # 音频完整保存路径
        self.record_seconds = record_seconds  # 录制时长
        self.rate = rate                # 采样率
        self.channels = channels        # 声道数
        self.format = format            # 采样格式
        self.chunk = 1024               # 数据块大小
        self.is_recording = False       # 录制状态标记
        self.frames = []                # 音频数据缓存
        self.p = None                   # PyAudio实例
        self.stream = None              # 音频流实例
        # 记录线程初始化日志（INFO级别，关键配置信息）
        logger.info(
            f"音频录制线程初始化完成\n"
            f"  保存路径：{self.save_path}\n"
            f"  录制时长：{self.record_seconds} 秒\n"
            f"  采样率：{self.rate} Hz\n"
            f"  声道数：{self.channels}\n"
            f"  采样格式：{self.format}（paInt16对应16位深度）"
        )

    def run(self):
        """线程运行入口，执行音频录制逻辑"""
        self.is_recording = True
        self.frames = []
        try:
            # 初始化PyAudio并打开音频流
            logger.info("开始初始化PyAudio实例并创建音频输入流")
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            logger.debug(f"PyAudio实例初始化成功，音频流已打开（数据块大小：{self.chunk}）")

            # 循环采集音频数据
            total_chunks = int(self.rate / self.chunk * self.record_seconds)
            logger.info(f"开始音频采集，总数据块数：{total_chunks}，预计录制{self.record_seconds}秒")
            for i in range(total_chunks):
                if not self.is_recording:
                    break
                data = self.stream.read(self.chunk)
                self.frames.append(data)
                # 计算剩余时长并发送状态
                remaining = self.record_seconds - (i * self.chunk / self.rate)
                self.status_signal.emit(f"正在录制：剩余{remaining:.1f}秒")
                # 每10个数据块记录一次进度日志，避免日志刷屏
                if i % 10 == 0:
                    progress = int(100 * (i + 1) / total_chunks)
                    logger.debug(f"音频采集进度：{progress}%（第{i+1}/{total_chunks}个数据块），剩余{remaining:.1f}秒")
                time.sleep(0.001)  # 避免GUI刷屏

            # 保存音频文件（仅当正常录制完成时）
            if self.is_recording:
                logger.info(f"录制正常完成，开始保存音频文件至：{self.save_path}")
                wf = wave.open(self.save_path, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                logger.success(f"音频文件保存成功：{self.save_path}（文件大小：{len(b''.join(self.frames))} 字节）")
                self.finish_signal.emit(self.save_path)
                self.status_signal.emit(f"录制完成：已保存文件")
            else:
                logger.warning("音频录制被手动取消，未生成有效音频文件")
                self.status_signal.emit("录制已取消")

        except Exception as e:
            err_msg = f"录制失败：{str(e)}"
            logger.error(f"音频录制异常：{err_msg}", exc_info=True)
            self.status_signal.emit(err_msg)
        finally:
            # 释放音频资源
            logger.debug("开始释放PyAudio音频资源（停止流+关闭流+终止实例）")
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                logger.debug("音频流已停止并关闭")
            if self.p:
                self.p.terminate()
                logger.debug("PyAudio实例已终止")
            self.is_recording = False
            logger.debug("音频录制线程资源释放完成，录制状态重置为False")

    def stop_recording(self):
        """外部调用：停止录制"""
        if self.is_recording:
            logger.info(f"接收到手动停止录制指令，当前录制状态将置为False（保存路径：{self.save_path}）")
            self.is_recording = False
        else:
            logger.debug("录制状态已为False，无需重复停止")