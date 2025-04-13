# SvsS - AI贪吃蛇对战框架

## 项目概述
基于PyQt5开发的AI贪吃蛇对战框架，支持实时可视化、智能体控制和多策略对战。包含完整的游戏逻辑分析、路径规划算法和可视化模块。

## 项目结构
```
SvsS/
├── analyzer/        # 游戏逻辑分析模块
│   └── board_analyzer.py   # 棋盘状态分析
│   └── path_finder.py      # A*路径规划算法
├── controller/      # 游戏控制模块
│   └── snake_controller.py # 键盘事件模拟
├── view/            # 可视化界面
│   └── card/        # 游戏卡片组件
│       └── snake_card.py   # 主游戏面板UI
├── player/          # 智能体模块
│   └── snake_player.py     # 自动控制逻辑
├── templates/       # 蛇类行为模板
│   └── snake/       # 不同AI策略
│       └── eye/      # 视觉识别策略
│       └── mine/     # 地雷躲避策略
├── log/             # 日志系统
├── model/           # 数据模型
└── main.py          # 程序入口
```

## 主要功能
- 实时游戏画面捕获与分析
- 多策略AI智能体控制
- 可视化调试面板
- 性能监控仪表盘
- 历史截图批量保存

## 使用说明
```bash
# 安装依赖
pip install -r requirements.txt

# 运行程序
python main.py
```

## 配置说明
1. 在`templates/snake/`目录创建AI策略模板
2. 通过`snake_card.py`界面启用智能体控制
3. 使用`DebugHelper`保存分析过程截图

## 开发建议
- 继承`SnakePlayer`类实现自定义AI
- 通过`MapDrawer`扩展可视化效果
- 使用`board_analyzer_test_card.py`进行单元测试