"""
集成面试示例脚本

演示如何使用 JingXin 多模态集成模块进行实时面试评估。
"""

# 从 main 模块导入集成器
from main.integrator import JingXinIntegrator


def main():
    """主函数：运行集成面试"""
    print("=" * 60)
    print("JingXin 多模态面试评估系统")
    print("=" * 60)
    print("ℹ️  启动后请保持自然坐姿1~2秒，系统将自动校准肩部基准")
    print("ℹ️  校准完成后，耸肩将被正确检测")
    print("🛑 按 'q' 键退出程序\n")

    integrator = JingXinIntegrator()
    integrator.start_interview_session()


if __name__ == "__main__":
    main()