import sys
import os
from PyQt6.QtWidgets import QApplication
from ui_app import MainWindow

if __name__ == "__main__":
    # 【核心修改】：PyInstaller 路径兼容逻辑
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 exe 运行，将工作目录设为 exe 所在的物理目录
        application_path = os.path.dirname(sys.executable)
    else:
        # 如果是 Python 源码运行，设为 main.py 所在目录
        application_path = os.path.dirname(os.path.abspath(__file__))
        
    os.chdir(application_path)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())