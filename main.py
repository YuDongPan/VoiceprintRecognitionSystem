import sys
from PyQt5.QtWidgets import QApplication
from voiceprint_gui.log_config import init_logger
from voiceprint_gui.login_window import LoginWindow

# ================================= 屏蔽警告信息 ===============================================
import os
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::libpng warning'
import warnings
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
# =========================================================================================

if __name__ == "__main__":
    # 初始化日志
    init_logger()

    # 启动Qt应用
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()

    # 运行应用
    sys.exit(app.exec_())