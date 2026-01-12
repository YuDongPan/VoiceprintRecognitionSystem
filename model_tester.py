import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from speechbrain.inference.speaker import EncoderClassifier
from loguru import logger
import config  # 导入全局配置
from model_trainer import TrainThread  # 复用特征提取方法，避免冗余


class TestThread(QThread):
    status_signal = pyqtSignal(str)  # 测试状态信号
    progress_signal = pyqtSignal(int)  # 测试进度信号（0-100）
    finish_signal = pyqtSignal(dict)  # 测试完成信号（返回结果字典）

    def __init__(self, username, algorithm, test_files):
        super().__init__()
        self.username = username  # 目标用户（当前登录用户）
        self.algorithm = algorithm.upper()  # 统一为大写
        self.test_files = test_files  # 测试音频文件列表
        self.user_dir = os.path.join(config.DATASET_ROOT, username)
        self.model_path = os.path.join(self.user_dir, f"{self.algorithm}_model.pkl")
        self.trainer = TrainThread(username, algorithm)  # 复用特征提取方法
        # 记录线程初始化日志
        logger.info(
            f"【{self.username}】测试线程初始化完成\n"
            f"  算法：{self.algorithm}\n"
            f"  测试文件数量：{len(self.test_files)}\n"
            f"  模型路径：{self.model_path}\n"
            f"  测试文件列表：{[os.path.basename(f) for f in self.test_files]}"
        )

    def run(self):
        """线程运行入口，执行模型测试逻辑"""
        try:
            # 1. 检查模型是否存在
            if not os.path.exists(self.model_path):
                err_msg = "未找到训练好的模型，请先训练"
                logger.warning(f"【{self.username}】{self.algorithm}测试失败：{err_msg}（模型路径不存在）")
                self.finish_signal.emit({
                    "success": False,
                    "message": err_msg
                })
                return

            # 2. 初始化测试结果
            results = {
                "success": True,
                "message": f"{self.algorithm}模型测试完成",
                "total": len(self.test_files),
                "correct": 0,
                "accuracy": 0.0,
                "details": []
            }

            # 3. 按算法类型执行测试
            self.status_signal.emit(f"开始使用{self.algorithm}算法进行测试...")
            logger.info(f"【{self.username}】{self.algorithm}模型测试启动，待测试文件数：{len(self.test_files)}")
            if self.algorithm == "SHERPA":
                self._test_sherpa_model(results)

            elif self.algorithm == "ECAPA":
                self._test_ecapa_model(results)

            else:
                err_msg = f"不支持的算法类型：{self.algorithm}（仅支持SHERPA/ECAPA）"
                results["success"] = False
                results["message"] = err_msg
                logger.error(f"【{self.username}】测试失败：{err_msg}")
                self.finish_signal.emit(results)
                return

            # 4. 计算准确率（修正后：基于真实标签与预测标签的匹配度）
            results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
            logger.success(
                f"【{self.username}】{self.algorithm}模型测试完成\n"
                f"  总测试数：{results['total']}\n"
                f"  正确数：{results['correct']}\n"
                f"  准确率：{results['accuracy']:.4f}（{results['accuracy'] * 100:.2f}%）"
            )
            self.finish_signal.emit(results)

        except Exception as e:
            err_msg = f"测试失败: {str(e)}"
            logger.error(f"【{self.username}】{self.algorithm}测试异常：{err_msg}", exc_info=True)
            self.finish_signal.emit({
                "success": False,
                "message": err_msg
            })

    def _test_sherpa_model(self, results):
        """测试 SHERPA-ONNX 模型（模板匹配 + VAD）"""

        # ===== 1. 加载 Sherpa-Onnx 声纹模板 =====
        self.status_signal.emit("加载 SHERPA预训练模型和声纹模板...")
        logger.info(f"【{self.username}】SHERPA模型：开始加载模板（路径：{self.model_path}）")

        try:
            with open(self.model_path, "rb") as f:
                model_info = pickle.load(f)

            template = torch.tensor(
                model_info["template"],
                dtype=torch.float32
            )
            threshold = model_info["threshold"]

            logger.info(
                f"【{self.username}】SHERPA模型：模板加载成功\n"
                f"  模板维度：{template.shape}\n"
                f"  匹配阈值：{threshold:.4f}"
            )

        except Exception as e:
            raise Exception(f"加载 SHERPA 模型失败：{str(e)}")

        # ===== 2. 逐个测试音频 =====
        self.status_signal.emit("逐个测试音频文件，提取 Sherpa 声纹嵌入...")
        for i, test_file in enumerate(self.test_files):
            file_basename = os.path.basename(test_file)

            # 真值标签（是否为目标用户）
            file_owner = os.path.basename(os.path.dirname(test_file))
            true_label = "匹配" if file_owner == self.username else "不匹配"

            logger.debug(
                f"【{self.username}】SHERPA模型：测试第 {i + 1}/{len(self.test_files)} 个文件：{file_basename}\n"
                f"  所属用户：{file_owner}\n"
                f"  真值标签：{true_label}"
            )

            try:
                # ===== 3. 提取 Sherpa embedding（带 VAD）=====
                test_emb = self.trainer.extract_sherpa_embedding(test_file)

                if test_emb is None:
                    raise RuntimeError("Sherpa embedding 提取失败")

                test_emb = torch.tensor(test_emb, dtype=torch.float32)

                # ===== 4. 计算余弦相似度 =====
                sim = F.cosine_similarity(
                    template.unsqueeze(0),
                    test_emb.unsqueeze(0)
                ).item()

                pred_label = "匹配" if sim >= threshold else "不匹配"
                is_correct = (pred_label == true_label)

                # ===== EMA 阈值更新（仅限高置信匹配）=====
                if pred_label == "匹配" and sim > threshold + 0.05:
                    new_threshold = (1 - config.SHERPA_EMA_ALPHA) * threshold + config.SHERPA_EMA_ALPHA * sim
                    new_threshold = float(np.clip(new_threshold, config.SHERPA_THRESHOLD_MIN, config.SHERPA_THRESHOLD_MAX))

                    if abs(new_threshold - threshold) > 1e-4:
                        logger.info(
                            f"【{self.username}】Sherpa 阈值 EMA 更新："
                            f"{threshold:.4f} → {new_threshold:.4f}"
                        )
                        threshold = new_threshold
                        model_info["threshold"] = threshold

                        with open(self.model_path, "wb") as f:
                            pickle.dump(model_info, f)


                # ===== 5. 记录结果 =====
                result_detail = {
                    "file": file_basename,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "result": "正确" if is_correct else "错误",
                    "similarity": sim,
                    "threshold": threshold,
                }

                results["details"].append(result_detail)

                if is_correct:
                    results["correct"] += 1

                logger.debug(
                    f"【{self.username}】SHERPA模型：文件 {file_basename} 测试完成\n"
                    f"  相似度：{sim:.4f}\n"
                    f"  阈值：{threshold:.4f}\n"
                    f"  预测标签：{pred_label}\n"
                    f"  是否正确：{is_correct}"
                )

            except Exception as e:
                err_detail = f"测试异常：{str(e)}"
                logger.warning(f"【{self.username}】SHERPA模型：文件 {file_basename} {err_detail}")

                results["details"].append({
                    "file": file_basename,
                    "true_label": true_label,
                    "pred_label": "异常",
                    "result": err_detail,
                    "correct": False
                })

            # ===== 6. 更新进度 =====
            progress = int(100 * (i + 1) / len(self.test_files))
            self.progress_signal.emit(progress)

    def _test_ecapa_model(self, results):
        """测试ECAPA模型（核心修正：真值标签判断 + 正确与否判定逻辑）"""
        # 加载ECAPA分类器和模板嵌入
        self.status_signal.emit("加载ECAPA预训练模型和声纹模板...")
        logger.info(f"【{self.username}】ECAPA模型：开始加载预训练模型和声纹模板")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            classifier = EncoderClassifier.from_hparams(
                source="./local_speechbrain_models/spkrec-ecapa-voxceleb",
                savedir=None,
                run_opts={"device": device}
            )
            classifier.eval()
            saved_obj = torch.load(self.model_path)

            if isinstance(saved_obj, dict):
                template_emb = saved_obj["template"]
                threshold = saved_obj["threshold"]
            else:
                # 兼容老模型
                template_emb = saved_obj
                threshold = config.ECAPA_DEFAULT_THRESHOLD

            logger.info(
                f"【{self.username}】ECAPA模型：加载成功\n"
                f"  使用设备：{device}\n"
                f"  声纹模板维度：{template_emb.shape}\n"
                f"  匹配阈值：{threshold:.4f}"
            )

        except Exception as e:
            raise Exception(f"加载ECAPA模型/模板失败：{str(e)}")

        # 逐个测试音频文件
        self.status_signal.emit("逐个测试音频文件，提取嵌入并计算相似度...")
        for i, test_file in enumerate(self.test_files):
            file_basename = os.path.basename(test_file)
            # 关键修正1：获取当前测试文件的所属用户（确定真值标签）
            file_owner = os.path.basename(os.path.dirname(test_file))
            true_label = "匹配" if file_owner == self.username else "不匹配"
            logger.debug(
                f"【{self.username}】ECAPA模型：开始测试第 {i + 1}/{len(self.test_files)} 个文件：{file_basename}（所属用户：{file_owner}，真值：{true_label}）")

            # 提取测试音频嵌入
            test_emb = self.trainer.extract_ecapa_embedding(classifier, test_file)
            if test_emb is None:
                err_msg = "提取特征失败"
                logger.warning(f"【{self.username}】ECAPA模型：文件 {file_basename} {err_msg}")
                # 异常时记录真值标签
                results["details"].append({
                    "file": file_basename,
                    "true_label": true_label,
                    "pred_label": "异常",
                    "result": err_msg,
                    "correct": False
                })
            else:
                # 计算相似度并判断预测标签
                sim = F.cosine_similarity(template_emb, test_emb).item()
                # 关键修正2：将相似度结果转为标签，与真值标签对比
                pred_label = "匹配" if sim > threshold else "不匹配"
                is_correct = (pred_label == true_label)  # 正确与否的核心判断

                logger.debug(
                    f"  相似度：{sim:.4f} | 用户阈值：{threshold:.4f} | 预测标签：{pred_label}"
                )

                # 记录结果（补充真值、预测值字段）
                result_str = f"{pred_label} (相似度: {sim:.4f})"
                result_detail = {
                    "file": file_basename,
                    "true_label": true_label,  # 真值标签
                    "pred_label": pred_label,  # 预测标签
                    "result": "正确" if is_correct else "错误",
                    "similarity": sim,
                    "threshold": threshold
                }
                results["details"].append(result_detail)
                if is_correct:
                    results["correct"] += 1  # 仅当真值与预测值一致时累加正确数

                # 记录该文件测试日志
                logger.debug(
                    f"【{self.username}】ECAPA模型：文件 {file_basename} 测试完成\n"
                    f"  所属用户：{file_owner}\n"
                    f"  真值标签：{true_label}\n"
                    f"  预测标签：{pred_label}\n"
                    f"  相似度：{sim:.4f}\n"
                    f"  阈值：{threshold:.4f}\n"
                    f"  是否正确：{is_correct}\n"
                    f"  匹配状态：{result_str}"
                )

            # 更新进度
            progress = int(100 * (i + 1) / len(self.test_files))
            self.progress_signal.emit(progress)
            logger.debug(f"【{self.username}】ECAPA模型：测试进度更新至 {progress}%")