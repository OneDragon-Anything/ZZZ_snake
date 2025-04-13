# SvsS (Snake vs Snake) 贪吃蛇游戏AI

这是一个基于Python的贪吃蛇游戏AI项目，使用PyQt5构建界面，OpenCV进行图像处理，实现了自动识别游戏画面并进行智能操控的功能。

## 项目结构

```
SvsS/
├── analyzer/                 # 分析器模块
│   ├── board_analyzer.py    # 棋盘分析器
│   └── path_finder.py       # 路径寻找器
├── app/                     # 应用配置
│   └── config/             # 配置文件目录
├── controller/             # 控制器模块
│   └── snake_controller.py # 蛇控制器
├── drawer/                 # 绘制器模块
│   └── map_drawer.py      # 地图绘制器
├── log/                    # 日志模块
│   ├── debug_helper.py    # 调试助手
│   └── log.py            # 日志记录器
├── model/                  # 模型模块
│   ├── image_cell.py     # 图像单元类
│   ├── snake_board.py    # 蛇棋盘类
│   └── template/         # 模板相关类
├── player/                # 玩家模块
│   └── snake_player.py   # 蛇玩家类
├── templates/             # 模板资源
│   ├── default_frame/    # 默认框架模板
│   └── snake/            # 蛇相关模板
├── view/                  # 视图模块
│   └── card/             # 卡片组件
├── main.py               # 主程序入口
├── mainwindow.py         # 主窗口
└── requirements.txt      # 项目依赖
```

## 主要功能

1. **棋盘分析**
   - 实时识别游戏画面
   - 分析蛇的位置和方向
   - 识别游戏中的各种元素（食物、障碍物等）

2. **智能寻路**
   - 使用路径寻找算法
   - 自动规划最优路径
   - 避开障碍物

3. **自动控制**
   - 自动操控蛇的移动
   - 实时响应游戏状态
   - 智能决策下一步行动

4. **可视化界面**
   - 实时显示游戏画面
   - 分析结果可视化
   - 性能指标监控

5. **调试功能**
   - 详细的日志记录
   - 调试信息输出
   - 截图保存功能

## 技术栈

- Python 3.x
- PyQt5：GUI框架
- OpenCV：图像处理
- NumPy：数据处理
- QFluentWidgets：现代化UI组件

## 使用说明

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行程序：
```bash
python main.py
```

3. 程序会以管理员权限运行，用于实现游戏画面捕获功能。

## 注意事项

- 需要管理员权限运行
- 确保游戏窗口可见且未被遮挡
- 支持自定义模板和配置

## 已知问题
- 吃了无敌之后暂时没有好的识别办法
- 加速的时候会反应不过来