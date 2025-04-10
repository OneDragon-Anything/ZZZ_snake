# 蛇吃蛇游戏AI控制器

## 项目概述
基于PyQt5和OpenCV的蛇吃蛇游戏AI控制器，实现自动游戏控制。

## 功能特性
- 实时游戏画面分析
- 最优路径计算(A*算法)
- 自动化蛇移动控制
- 可视化调试界面

## 项目结构
```
SvsS/
├── analyzer/          # 游戏分析模块
│   ├── board_analyzer.py  # 棋盘分析
│   └── path_finder.py     # 路径查找
├── controller/        # 控制模块
│   └── snake_controller.py # 蛇控制
├── drawer/           # 可视化模块
│   └── map_drawer.py     # 地图绘制
├── log/              # 日志模块
│   ├── debug_helper.py   # 调试工具
│   └── log.py           # 日志记录
├── model/            # 数据模型
│   ├── image_cell.py    # 格子类型
│   └── snake_board.py   # 棋盘模型
├── player/           # 主逻辑
│   └── snake_player.py  # 游戏处理
├── view/             # 界面组件
│   ├── card/
│   │   ├── board_analyzer_test_card.py
│   │   ├── settings_card.py
│   │   ├── snake_card.py      # 游戏控制卡片
│   │   └── theme_manager.py
│   └── mainwindow.py      # 主窗口
└── main.py           # 程序入口
```

## 依赖项
- PyQt5
- OpenCV
- numpy

## 使用说明
1. 安装依赖: `pip install -r requirements.txt`
2. 运行主程序: `python main.py`
3. 在界面中点击"开始"按钮启动AI

## 示例
提供examples目录展示各模块使用方法。