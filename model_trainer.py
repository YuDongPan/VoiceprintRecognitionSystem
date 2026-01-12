import os
import pickle
import numpy as np
import sherpa_onnx
import torch
import torchaudio
import soundfile as sf
import torch.nn.functional as F
from PyQt5.QtCore import QThread, pyqtSignal
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.VAD import VAD
from loguru import logger
import config  # 导入全局配置


class TrainThread(QThread):
    status_signal = pyqtSignal(str)  # 训练状态信号
    progress_signal = pyqtSignal(int)  # 训练进度信号（0-100）
    finish_signal = pyqtSignal(bool, str)  # 训练完成信号（是否成功+提示信息）

    def __init__(self, username, algorithm):
        super().__init__()
        self.username = username
        self.algorithm = algorithm.upper()  # 统一为大写（SHERPA/ECAPA）
        self.user_dir = os.path.join(config.DATASET_ROOT, username)
        self.model_path = os.path.join(self.user_dir, f"{self.algorithm}_model.pkl")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # 加载SHERPA预训练模型
        logger.info(f"【{self.username}】SHERPA模型：开始加载预训练模型")
        try:
            self.sherpa_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model="./local_sherpa_onnx_models/wespeaker_zh_cnceleb_resnet34.onnx",
                    num_threads=2,
                    provider="cpu",
                )
            )
            logger.info(f"【{self.username}】SHERPA模型：预训练模型加载成功，使用设备：cpu")

        except Exception as e:
            raise Exception(f"加载SHERPA预训练模型失败：{str(e)}")

        # 加载SHERPA-VAD预训练模型
        sherpa_vad_config = sherpa_onnx.VadModelConfig()
        sherpa_vad_config.silero_vad.model = "./local_sherpa_onnx_models/silero_vad.onnx"
        sherpa_vad_config.sample_rate = 16000
        sherpa_vad_config.silero_vad.min_speech_duration = config.SHERPA_VAD_MIN_SPEECH_DURATION
        sherpa_vad_config.silero_vad.min_silence_duration = config.SHERPA_VAD_MIN_SILENCE_DURATION
        sherpa_vad_config.silero_vad.threshold = config.SHERPA_VAD_THRESHOLD

        '''
        参数	                            影响	                 推荐范围	   你的设置
        min_speech_duration	    最小语音段长度，低于此值会被丢弃	0.1-0.3秒	0.1（较敏感）
        min_silence_duration	最小静音长度，用于分割连续语音	0.1-0.5秒	0.1（较敏感）
        threshold	            语音/非语音决策阈值	        0.1-0.9	   未设置（默认~0.5）
        window_size	            处理窗口大小（样本数）	        256-1024	    512
        '''

        self.sherpa_vad = sherpa_onnx.VoiceActivityDetector(
            sherpa_vad_config,
            buffer_size_in_seconds=10
        )

        logger.info(f"【{self.username}】ECAPA模型：开始加载预训练模型")
        try:
            self.ecapa_extractor = EncoderClassifier.from_hparams(
                source="./local_speechbrain_models/spkrec-ecapa-voxceleb",
                savedir=None,
                run_opts={"device": self.device}
            )
            self.ecapa_extractor.eval()
            logger.info(f"【{self.username}】ECAPA模型：预训练模型加载成功，使用设备：{self.device}")

        except Exception as e:
            raise Exception(f"加载ECAPA预训练模型失败：{str(e)}")

        # 加载ECAPA-VAD预训练模型
        self.ecapa_vad = VAD.from_hparams(
            source="./local_speechbrain_models/vad-crdnn-libriparty",
            savedir=None,
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )

        # 记录初始化日志
        logger.info(f"【{self.username}】训练线程初始化完成，算法：{self.algorithm}，模型保存路径：{self.model_path}")



    def run(self):
        """线程运行入口，执行模型训练逻辑"""
        try:
            # 1. 检查训练数据
            wav_files = [f for f in os.listdir(self.user_dir) if f.endswith(".wav")]
            if not wav_files:
                err_msg = "没有找到音频文件，请先录制训练数据"
                logger.warning(f"【{self.username}】{self.algorithm}训练失败：{err_msg}")
                self.finish_signal.emit(False, err_msg)
                return

            # 2. 检查数据量是否满足算法要求（严格区分：Sherpa-Onnx≥3，ECAPA≥3）
            if self.algorithm == "SHERPA" and len(wav_files) < 3:
                err_msg = "SHERPA算法需要至少3个训练样本"
                logger.warning(f"【{self.username}】{self.algorithm}训练失败：{err_msg}（当前样本数：{len(wav_files)}）")
                self.finish_signal.emit(False, err_msg)
                return

            # 核心：确保ECAPA算法仅需3个样本，明确提示信息
            if self.algorithm == "ECAPA" and len(wav_files) < 3:
                err_msg = "ECAPA算法需要至少3个训练样本"
                logger.warning(f"【{self.username}】{self.algorithm}训练失败：{err_msg}（当前样本数：{len(wav_files)}）")
                self.finish_signal.emit(False, err_msg)
                return

            # 3. 开始训练流程
            self.status_signal.emit(f"开始使用{self.algorithm}算法训练模型...")
            self.progress_signal.emit(10)
            logger.info(f"【{self.username}】{self.algorithm}模型训练启动，有效音频样本数：{len(wav_files)}")

            # 4. 按算法类型执行训练
            if self.algorithm == "SHERPA":
                self._train_sherpa_model(wav_files)
            elif self.algorithm == "ECAPA":
                self._train_ecapa_model(wav_files)
            else:
                err_msg = f"不支持的算法类型：{self.algorithm}（仅支持SHERPA/ECAPA）"
                logger.error(f"【{self.username}】训练失败：{err_msg}")
                self.finish_signal.emit(False, err_msg)
                return

            # 5. 训练完成
            self.progress_signal.emit(100)
            self.status_signal.emit(f"{self.algorithm}模型训练完成")
            succ_msg = f"{self.algorithm}模型训练成功"
            logger.success(f"【{self.username}】{succ_msg}，模型已保存至：{self.model_path}")
            self.finish_signal.emit(True, succ_msg)

        except Exception as e:
            err_msg = f"训练失败: {str(e)}"
            logger.error(f"【{self.username}】{self.algorithm}训练异常：{err_msg}", exc_info=True)
            self.finish_signal.emit(False, err_msg)

    def _train_sherpa_model(self, wav_files):
        """训练SHERPA模型（提取平均嵌入作为模板，仅需3个有效样本）"""

        # 加载SHERPA预训练模型
        self.progress_signal.emit(30)
        self.status_signal.emit("已成功加载SHERPA预训练模型...")

        # 提取音频嵌入（过滤无效嵌入，确保最终有效数量≥3）
        self.status_signal.emit("提取SHERPA音频嵌入特征...")
        logger.info(f"【{self.username}】SHERPA模型：开始提取{len(wav_files)}个音频的嵌入特征")
        embeddings = []
        for i, f in enumerate(wav_files):
            emb = self.extract_sherpa_embedding(os.path.join(self.user_dir, f))
            if emb is not None:
                embeddings.append(emb)
            else:
                logger.warning(f"【{self.username}】SHERPA模型：跳过无效音频文件 {f}（嵌入提取失败）")

            # 更新进度
            self.progress_signal.emit(30 + int(60 * (i + 1) / len(wav_files)))

        # 额外检查：有效嵌入数量是否≥3（避免部分文件提取失败导致有效样本不足）
        if len(embeddings) < 3:
            err_msg = f"SHERPA算法需要至少3个有效音频样本（当前有效样本数：{len(embeddings)}）"
            logger.error(f"【{self.username}】SHERPA训练失败：{err_msg}")
            raise Exception(err_msg)

        logger.debug(
            f"【{self.username}】ECAPA模型：嵌入特征提取完成，有效嵌入数量：{len(embeddings)}，嵌入维度：{embeddings[0].shape}")

        # 计算平均嵌入并保存（作为用户声纹模板）
        self.status_signal.emit("生成SHERPA声纹模板（平均嵌入）...")

        # template = np.mean(np.stack(embeddings), axis=0)

        emb_stack = np.stack(embeddings)  # [N, D]

        # 中心
        center = emb_stack.mean(axis=0, keepdims=True)

        # cosine similarity
        sims = np.sum(emb_stack * center, axis=1) / (
                np.linalg.norm(emb_stack, axis=1) * np.linalg.norm(center)
        )

        mean_sim = sims.mean()
        std_sim = sims.std()

        # 轻量筛选（比 ECAPA 温和）
        valid_mask = sims >= (mean_sim - 0.5 * std_sim)
        filtered_embs = emb_stack[valid_mask]

        if filtered_embs.shape[0] < 3:
            filtered_embs = emb_stack  # 回退

        template = filtered_embs.mean(axis=0)

        # 阈值：小样本固定
        num_emb = filtered_embs.shape[0]

        if num_emb < config.SHERPA_DEFAULT_THRESHOLD_NUM_EMBEDS:
            user_threshold = config.SHERPA_DEFAULT_THRESHOLD  # 小样本：固定

            logger.warning(
                f"【{self.username}】SHERPA样本数={num_emb} < {config.ECAPA_DEFAULT_THRESHOLD_NUM_EMBEDS}，"
                f"使用默认阈值 {user_threshold:.2f}"
            )

        else:
            # 用户内相似度
            user_threshold = mean_sim - 0.5 * std_sim
            user_threshold = max(user_threshold, config.SHERPA_THRESHOLD_MIN)
            user_threshold = min(user_threshold, config.SHERPA_THRESHOLD_MAX)

            logger.info(
                f"【{self.username}】SHERPA自适应阈值学习完成："
                f"{user_threshold:.4f}（样本数={num_emb}）"
            )

        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "template": template,
                    "threshold": user_threshold,
                    "threshold_mode": "default" if num_emb < 5 else "adaptive",
                    "model": "sherpa-onnx",
                },
                f
            )

    def _train_ecapa_model(self, wav_files):
        """训练ECAPA模型（提取平均嵌入作为模板，仅需3个有效样本）"""
        # 加载ECAPA预训练模型
        self.progress_signal.emit(30)
        self.status_signal.emit("已成功加载ECAPA预训练模型...")

        # 提取音频嵌入（过滤无效嵌入，确保最终有效数量≥3）
        self.status_signal.emit("提取ECAPA音频嵌入特征...")
        logger.info(f"【{self.username}】ECAPA模型：开始提取{len(wav_files)}个音频的嵌入特征")
        embeddings = []
        for i, f in enumerate(wav_files):
            wav_path = os.path.join(self.user_dir, f)
            emb = self.extract_ecapa_embedding(self.ecapa_extractor, wav_path)
            if emb is not None:
                embeddings.append(emb)
            else:
                logger.warning(f"【{self.username}】ECAPA模型：跳过无效音频文件 {f}（嵌入提取失败）")

            # 更新进度
            self.progress_signal.emit(30 + int(60 * (i + 1) / len(wav_files)))

        # 额外检查：有效嵌入数量是否≥3（避免部分文件提取失败导致有效样本不足）
        if len(embeddings) < 3:
            err_msg = f"ECAPA算法需要至少3个有效音频样本（当前有效样本数：{len(embeddings)}）"
            logger.error(f"【{self.username}】ECAPA训练失败：{err_msg}")
            raise Exception(err_msg)

        logger.debug(
            f"【{self.username}】ECAPA模型：嵌入特征提取完成，有效嵌入数量：{len(embeddings)}，嵌入维度：{embeddings[0].shape}")

        # 计算平均嵌入并保存（作为用户声纹模板）
        self.status_signal.emit("生成ECAPA声纹模板（平均嵌入）...")

        # ===== 新增：embedding 一致性筛选 =====
        emb_stack = torch.cat(embeddings, dim=0)  # [N, D]
        center = torch.mean(emb_stack, dim=0, keepdim=True)

        # 计算每段与中心的相似度
        sims = F.cosine_similarity(emb_stack, center)  # [N]

        mean_sim = sims.mean().item()
        std_sim = sims.std().item()

        # 筛选规则（经验值）
        valid_mask = sims >= (mean_sim - 1.0 * std_sim)
        filtered_embs = emb_stack[valid_mask]

        logger.info(
            f"【{self.username}】ECAPA模板筛选："
            f"原始 {len(embeddings)} → 保留 {filtered_embs.shape[0]}"
        )

        # 再次检查数量
        if filtered_embs.shape[0] < 3:
            logger.warning("有效嵌入不足，回退使用全部嵌入")
            filtered_embs = emb_stack

        template_emb = torch.mean(filtered_embs, dim=0, keepdim=True)
        template_emb = F.normalize(template_emb, dim=-1)

        # ===== 新增：学习用户自适应阈值 =====
        # ===== 阈值策略：小样本使用默认阈值 =====
        num_emb = filtered_embs.shape[0]

        if num_emb < config.ECAPA_DEFAULT_THRESHOLD_NUM_EMBEDS:
            user_threshold = config.ECAPA_DEFAULT_THRESHOLD  # 例如 0.7
            logger.warning(
                f"【{self.username}】ECAPA样本数={num_emb} < {config.ECAPA_DEFAULT_THRESHOLD_NUM_EMBEDS}，"
                f"使用默认阈值 {user_threshold:.2f}"
            )
        else:
            alpha = config.ECAPA_THRESHOLD_STD_COEF
            user_threshold = mean_sim - alpha * std_sim
            user_threshold = max(user_threshold, config.ECAPA_THRESHOLD_MIN)
            user_threshold = min(user_threshold, config.ECAPA_THRESHOLD_MAX)

            logger.info(
                f"【{self.username}】ECAPA自适应阈值学习完成："
                f"{user_threshold:.4f}（样本数={num_emb}）"
            )

        # 保存模板嵌入和阈值（torch.save兼容张量格式）
        torch.save(
            {
                "template": template_emb,
                "threshold": user_threshold,
                "threshold_mode": "default" if num_emb < 8 else "adaptive",
                "num_embeddings": num_emb
            },
            self.model_path
        )
        logger.info(f"【{self.username}】ECAPA模型：声纹模板生成完成，已保存至{self.model_path}")

    def extract_sherpa_embedding(self, wav_path):
        """优化后的VAD处理方式，包含音频质量检查和动态调整"""
        try:
            # ========== 1. 读取和检查音频 ==========
            samples, sr = sf.read(wav_path, dtype="float32")

            # 确保samples是numpy数组
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)

            # 确保单声道
            if samples.ndim > 1:
                samples = samples[:, 0]  # 取第一列（左声道）

            # 检查采样率，如果不是16kHz则重采样
            if sr != 16000:
                logger.warning(f"音频采样率{sr}Hz，需要重采样到16000Hz")
                try:
                    import librosa
                    # 确保输入是numpy数组
                    samples_np = np.asarray(samples, dtype=np.float32)
                    samples = librosa.resample(samples_np, orig_sr=sr, target_sr=16000)
                    sr = 16000
                    logger.info(f"音频已重采样到16000Hz")
                except ImportError:
                    raise ValueError("音频采样率不是16000Hz，且未安装librosa进行重采样")

            # ========== 2. 音频质量检查 ==========
            # 再次确保samples是numpy数组
            samples = np.asarray(samples, dtype=np.float32)
            audio_duration = len(samples) / sr

            # 2.1 检查音频长度
            if audio_duration < 0.3:  # 小于300ms
                logger.warning(f"音频过短: {audio_duration:.3f}秒，可能影响声纹提取精度")
                # 不直接返回，尝试继续处理

            # 2.2 检查音量（RMS能量）- 修复这里：确保是numpy数组
            # 避免列表类型的samples
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)

            # 计算RMS能量 - 添加类型检查和转换
            try:
                # 确保samples是numpy数组
                samples_np = np.asarray(samples, dtype=np.float32)
                rms_energy = np.sqrt(np.mean(samples_np ** 2))

            except Exception as e:
                logger.warning(f"计算RMS能量失败: {e}, 尝试直接计算")
                # 备选计算方法
                if hasattr(samples, '__iter__'):
                    # 如果是可迭代对象（列表、数组等）
                    squared_sum = sum(float(x) ** 2 for x in samples)
                    rms_energy = np.sqrt(squared_sum / len(samples)) if len(samples) > 0 else 0.0
                else:
                    rms_energy = 0.0

            logger.debug(f"音频信息: 长度={audio_duration:.3f}s, RMS能量={rms_energy:.6f}")

            # 2.3 应用自动增益（如果需要）
            if rms_energy < 0.001 and rms_energy > 0:  # 能量极低但不是零
                logger.warning(f"音频能量极低 (RMS={rms_energy:.6f})，尝试自动增益")
                # 确保samples是numpy数组
                samples_np = np.asarray(samples, dtype=np.float32)
                max_abs = np.max(np.abs(samples_np))
                if max_abs > 0:
                    gain_factor = min(0.1 / rms_energy, 10.0)  # 最大增益10倍
                    samples = samples_np * gain_factor
                    logger.info(f"应用自动增益: {gain_factor:.2f}倍")

            # 2.4 检查是否有有效信号
            samples_np = np.asarray(samples, dtype=np.float32)
            if np.max(np.abs(samples_np)) < 0.0001:
                logger.error(f"音频信号几乎全零，无法处理: {wav_path}")
                return None

            # 现在确保samples是numpy数组用于后续处理
            samples = np.asarray(samples, dtype=np.float32)

            # ========== 3. 动态调整窗口大小 ==========
            # 根据音频长度动态选择窗口大小
            if audio_duration < 1.0:  # 短音频（<1秒）
                window_size = 256  # 16ms窗口，更精细
                logger.debug(f"短音频({audio_duration:.2f}s)，使用小窗口{window_size}(16ms)")
            elif audio_duration < 3.0:  # 中等音频（1-3秒）
                window_size = 512  # 32ms窗口，标准
                logger.debug(f"中等音频({audio_duration:.2f}s)，使用标准窗口{window_size}(32ms)")
            else:  # 长音频（>3秒）
                window_size = 1024  # 64ms窗口，更稳定
                logger.debug(f"长音频({audio_duration:.2f}s)，使用大窗口{window_size}(64ms)")

            # 确保window_size是有效正数且不超过音频长度
            if window_size <= 0:
                window_size = 512
            if window_size > len(samples):
                window_size = max(256, len(samples) // 2)
                logger.debug(f"窗口大小调整到音频长度的一半: {window_size}")

            # ========== 4. VAD处理 ==========
            self.sherpa_vad.reset()

            # 优化分块处理：考虑边界情况
            total_samples = len(samples)
            for i in range(0, total_samples, window_size):
                chunk = samples[i:i + window_size]
                if len(chunk) > 0:
                    self.sherpa_vad.accept_waveform(chunk)

            # 确保所有数据都已处理（处理最后不完整的块）
            remaining = total_samples % window_size
            if remaining > 0:
                self.sherpa_vad.accept_waveform(samples[-remaining:])

            embeddings = []
            segments_found = 0
            segments_skipped = 0

            # ========== 5. 处理VAD检测到的语音段 ==========
            while not self.sherpa_vad.empty():
                seg = self.sherpa_vad.front
                # 确保seg.samples是numpy数组
                seg_samples = np.asarray(seg.samples, dtype=np.float32)
                segment_length = len(seg_samples) / sr

                # 动态调整最短语音段要求
                # 短音频（<2秒）使用更宽松的标准
                min_speech_duration = 0.3 if audio_duration < 2.0 else 0.5

                if segment_length < min_speech_duration:
                    segments_skipped += 1
                    logger.debug(f"跳过过短语音段: {segment_length:.3f}s < {min_speech_duration}s")
                    self.sherpa_vad.pop()
                    continue

                # 检查语音段的能量
                seg_rms = np.sqrt(np.mean(seg_samples ** 2))
                if seg_rms < 0.001:  # 语音段能量过低
                    logger.debug(f"跳过低能量语音段: RMS={seg_rms:.6f}")
                    self.sherpa_vad.pop()
                    segments_skipped += 1
                    continue

                # 提取声纹嵌入
                stream = self.sherpa_extractor.create_stream()
                stream.accept_waveform(sr, seg_samples)
                stream.input_finished()

                if not self.sherpa_extractor.is_ready(stream):
                    logger.debug("提取器未就绪，跳过该段")
                    self.sherpa_vad.pop()
                    segments_skipped += 1
                    continue

                emb = np.asarray(self.sherpa_extractor.compute(stream))
                embeddings.append(emb)
                segments_found += 1

                logger.debug(f"提取到语音段 {segments_found}: {segment_length:.3f}s, RMS={seg_rms:.6f}")
                self.sherpa_vad.pop()

            # 记录VAD检测结果
            logger.debug(f"VAD检测结果: 找到{segments_found}个语音段，跳过{segments_skipped}个")

            # ========== 6. 处理结果 ==========
            if embeddings:
                if len(embeddings) > 1:
                    logger.debug(f"检测到{len(embeddings)}个语音段，计算平均嵌入")
                    # 可选：加权平均（根据语音段长度加权）
                    # 注意：这里简化了权重计算
                    emb_stack = np.stack(embeddings)
                    return np.mean(emb_stack, axis=0)
                else:
                    logger.debug(f"检测到1个语音段，直接使用")
                    return embeddings[0]

            # ========== 7. 回退机制：VAD未检测到语音 ==========
            logger.warning(f"Sherpa VAD 未检测到语音段，回退整段音频: {wav_path}")

            # 对整段音频进行质量检查
            if rms_energy < 0.0005:  # 能量极低
                logger.error(f"整段音频能量过低，无法提取声纹: RMS={rms_energy:.6f}")
                return None

            # 尝试直接使用整段音频
            stream = self.sherpa_extractor.create_stream()
            # 确保samples是numpy数组
            final_samples = np.asarray(samples, dtype=np.float32)
            stream.accept_waveform(sr, final_samples)
            stream.input_finished()

            if not self.sherpa_extractor.is_ready(stream):
                logger.error(f"整段音频提取器也未就绪: {wav_path}")
                return None

            emb = np.asarray(self.sherpa_extractor.compute(stream))
            logger.debug(f"回退机制: 使用整段音频提取嵌入")
            return emb

        except Exception as e:
            logger.warning(f"Sherpa 嵌入提取失败 ({wav_path}): {e}", exc_info=True)
            return None

    def extract_ecapa_embedding(self, classifier, wav_path):
        """优化后的ECAPA嵌入提取，包含音频质量检查和动态调整"""
        try:
            logger.debug(f"开始处理ECAPA音频: {wav_path}")

            # ===== 1. 读取音频并检查质量 =====
            try:
                # 使用torchaudio读取（一次读取，避免重复）
                signal, fs = torchaudio.load(wav_path)

                # 单声道处理
                if signal.shape[0] > 1:
                    signal = signal.mean(dim=0, keepdim=True)

                # 获取音频基本信息
                audio_duration = signal.shape[1] / fs

                # 将张量转换为numpy进行质量检查
                signal_np = signal.numpy().flatten()
                rms_energy = np.sqrt(np.mean(signal_np ** 2))

                logger.debug(f"ECAPA音频信息: 长度={audio_duration:.3f}s, RMS能量={rms_energy:.6f}, 采样率={fs}Hz")

                # 检查音频质量
                if audio_duration < 0.3:
                    logger.warning(f"ECAPA音频过短: {audio_duration:.3f}秒")

                if rms_energy < 0.001:
                    logger.warning(f"ECAPA音频能量低: RMS={rms_energy:.6f}")

                    # 应用自动增益（在张量上操作）
                    max_abs = torch.max(torch.abs(signal))
                    if max_abs > 0:
                        gain_factor = min(0.1 / rms_energy, 5.0)
                        signal = signal * gain_factor
                        logger.debug(f"ECAPA音频应用自动增益: {gain_factor:.2f}倍")

                # 检查是否有有效信号
                if torch.max(torch.abs(signal)) < 0.0001:
                    logger.error(f"ECAPA音频信号几乎全零，无法处理: {wav_path}")
                    return None

            except Exception as e:
                logger.error(f"ECAPA音频读取失败: {e}")
                return None

            # 确保采样率是16kHz（ECAPA需要16k）
            if fs != config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(fs, config.SAMPLE_RATE)
                signal = resampler(signal)
                fs = config.SAMPLE_RATE
                logger.debug(f"ECAPA音频已重采样到{config.SAMPLE_RATE}Hz")

            # ===== 2. VAD语音检测 =====
            boundaries = self.ecapa_vad.get_speech_segments(wav_path)

            if boundaries is None or boundaries.numel() == 0:
                logger.warning(f"ECAPA VAD未检测到语音段，尝试回退整段音频: {wav_path}")
                # 回退到使用整段音频
                try:
                    # 检查音频长度是否足够
                    if signal.shape[1] < int(0.3 * fs):
                        logger.error(f"ECAPA音频过短，无法处理: {signal.shape[1] / fs:.3f}秒")
                        return None

                    # 提取整段音频嵌入
                    chunk = signal.to(classifier.device)
                    emb = classifier.encode_batch(chunk).squeeze(1)
                    emb = F.normalize(emb, dim=-1)
                    logger.debug(f"ECAPA回退成功: 使用整段音频，长度={signal.shape[1] / fs:.3f}秒")
                    return emb

                except Exception as e:
                    logger.error(f"ECAPA回退处理失败: {e}")
                    return None

            # ===== 3. 动态调整参数 =====
            # 根据音频总长度调整最短语音段要求
            if audio_duration < 2.0:  # 短音频
                min_segment_duration = 0.3  # 300ms
            elif audio_duration < 5.0:  # 中等音频
                min_segment_duration = 0.5  # 500ms
            else:  # 长音频
                min_segment_duration = 0.7  # 700ms

            logger.debug(f"ECAPA音频总长: {audio_duration:.3f}s, 使用最小段长: {min_segment_duration:.3f}s")

            embeddings = []
            segment_qualities = []  # 存储语音段质量信息
            valid_segments = 0
            skipped_segments = 0

            # ===== 4. 处理VAD检测到的语音段 =====
            for seg_idx, seg in enumerate(boundaries):
                # seg = [start_sec, end_sec]
                start_sec = seg[0].item()
                end_sec = seg[1].item()
                segment_duration = end_sec - start_sec

                # 跳过过短的语音段
                if segment_duration < min_segment_duration:
                    skipped_segments += 1
                    logger.debug(f"ECAPA跳过短段 {seg_idx}: {segment_duration:.3f}s < {min_segment_duration:.3f}s")
                    continue

                # 转换为样本索引
                start = int(start_sec * fs)
                end = int(end_sec * fs)

                # 边界检查
                if start >= signal.shape[1] or end > signal.shape[1] or start >= end:
                    logger.warning(f"ECAPA语音段边界无效: [{start_sec:.3f}, {end_sec:.3f}]s")
                    skipped_segments += 1
                    continue

                chunk = signal[:, start:end]

                # 检查语音段能量（使用张量计算）
                segment_rms = torch.sqrt(torch.mean(chunk ** 2)).item()

                # 跳过能量过低的语音段
                if segment_rms < 0.0005:
                    skipped_segments += 1
                    logger.debug(f"ECAPA跳过低能量段 {seg_idx}: RMS={segment_rms:.6f}")
                    continue

                # 移动到设备并提取嵌入
                try:
                    chunk = chunk.to(classifier.device)
                    emb = classifier.encode_batch(chunk).squeeze(1)
                    emb = F.normalize(emb, dim=-1)

                    embeddings.append(emb)
                    segment_qualities.append({
                        'duration': segment_duration,
                        'rms': segment_rms,
                        'start': start_sec,
                        'end': end_sec
                    })
                    valid_segments += 1

                    logger.debug(f"ECAPA有效语音段 {seg_idx}: {segment_duration:.3f}s, RMS={segment_rms:.6f}")

                except Exception as e:
                    logger.warning(f"ECAPA语音段处理失败 {seg_idx}: {e}")
                    skipped_segments += 1
                    continue

            logger.debug(f"ECAPA VAD结果: 找到{len(boundaries)}个段，有效{valid_segments}个，跳过{skipped_segments}个")

            # ===== 5. 处理结果 =====
            if len(embeddings) == 0:
                logger.warning(f"ECAPA无有效语音段，尝试回退: {wav_path}")

                # 回退到使用整段音频
                try:
                    # 检查是否需要截取中间部分（避免开头结尾静音）
                    if audio_duration > 2.0:
                        # 取中间1-2秒
                        middle_start = int(fs * max(0.5, audio_duration / 2 - 1.0))
                        middle_end = int(fs * min(audio_duration - 0.5, audio_duration / 2 + 1.0))
                        if middle_end - middle_start > int(0.5 * fs):  # 至少0.5秒
                            signal = signal[:, middle_start:middle_end]
                            logger.debug(f"ECAPA使用音频中间部分: {middle_start / fs:.2f}s-{middle_end / fs:.2f}s")

                    chunk = signal.to(classifier.device)
                    emb = classifier.encode_batch(chunk).squeeze(1)
                    emb = F.normalize(emb, dim=-1)
                    logger.debug(f"ECAPA回退成功: 使用整段音频")
                    return emb
                except Exception as e:
                    logger.error(f"ECAPA回退失败: {e}")
                    return None

            # ===== 6. 智能嵌入融合策略 =====
            emb_stack = torch.cat(embeddings, dim=0)

            # 动态选择融合策略
            if len(embeddings) <= 2:
                # 样本少，全部使用简单平均
                final_emb = torch.mean(emb_stack, dim=0, keepdim=True)
                logger.debug(f"ECAPA使用全部{len(embeddings)}个嵌入的平均")

            elif len(embeddings) <= 5:
                # 中等数量，取质量最好的K个（基于长度和能量）
                # 计算每个段的质量分数
                quality_scores = []
                for quality in segment_qualities:
                    # 质量分数 = 长度 × 能量（归一化）
                    score = quality['duration'] * quality['rms']
                    quality_scores.append(score)

                # 选择前K个最好的
                K = min(3, len(embeddings))
                best_indices = np.argsort(quality_scores)[-K:]  # 取分数最高的K个
                best_embs = torch.cat([embeddings[i] for i in best_indices], dim=0)
                final_emb = torch.mean(best_embs, dim=0, keepdim=True)
                logger.debug(f"ECAPA使用前{K}个高质量语音段的平均")

            else:
                # 样本多，使用加权平均
                weights = []
                for quality in segment_qualities:
                    # 权重 = 长度 × 能量
                    weight = quality['duration'] * quality['rms']
                    weights.append(weight)

                # 转换为张量并归一化
                weights_tensor = torch.tensor(weights, device=emb_stack.device).unsqueeze(1)
                weights_tensor = weights_tensor / weights_tensor.sum()

                # 加权平均
                final_emb = torch.sum(emb_stack * weights_tensor, dim=0, keepdim=True)
                logger.debug(f"ECAPA使用{len(embeddings)}个嵌入的加权平均")

            # 归一化最终嵌入
            final_emb = F.normalize(final_emb, dim=-1)

            logger.debug(f"ECAPA嵌入提取成功: 最终嵌入形状={final_emb.shape}")
            return final_emb

        except Exception as e:
            logger.warning(f"ECAPA+VAD 嵌入提取失败 ({wav_path}): {e}", exc_info=True)
            return None





