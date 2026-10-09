import omnigibson as og

# 1.5.0 版本使用字典定义配置
config = {
    # 最简空场景，减少资产加载
    "scene": {
        "type": "Scene"
    },
    # 核心修复：强制光栅化渲染，关闭RTX光追
    "renderer_type": "Storm",
    "enable_rt": False,
    # 降低渲染分辨率，减少显存占用
    "image_width": 960,
    "image_height": 540,
    # 无头模式，规避显示设备相关问题
    "headless": True
}

# 创建环境
env = og.Environment(configs=config)

# 执行一步仿真
env.step([])
print("✅ 环境启动成功，无段错误")

# 正常关闭
og.shutdown()