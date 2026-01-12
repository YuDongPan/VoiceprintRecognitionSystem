from pathlib import Path
from loguru import logger

def init_logger():
    """初始化全局日志配置（可直接调用）"""
    # 创建日志文件夹
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)

    # 配置loguru日志
    logger.add(
        log_dir / "voiceprint_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module: <15} | {message}",
    )
    logger.info("全局日志配置初始化完成")