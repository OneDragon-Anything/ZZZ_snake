import time
from pynput import keyboard
from pynput import mouse
import win32gui
from log import SnakeLogger

class SnakeController:
    """负责根据地图和路径控制小蛇的类"""
    def __init__(self, logger=None):
        self.logger = logger
        self.keyboard_controller = keyboard.Controller()  # 初始化键盘控制器
        self.mouse_controller = mouse.Controller()  # 初始化鼠标控制器
        self.last_press_time = 0  # 上次按键时间
        self.last_click_time = 0  # 上次点击时间
        self.hwnd = None  # 游戏窗口句柄
        
    def set_game_window(self, hwnd):
        """设置游戏窗口句柄"""
        self.hwnd = hwnd
        
    def control_snake(self, direction):
        """
        控制蛇的移动
        :param direction: 移动方向，'w'上，'s'下，'a'左，'d'右
        """
        if not direction:
            return
            
        try:
            # 记录按键时间
            current_time = time.time()
            self.last_press_time = current_time
            
            # 按下并释放按键
            self.keyboard_controller.press(keyboard.KeyCode.from_char(direction))
            time.sleep(0.05)  # 短暂延迟
            self.keyboard_controller.release(keyboard.KeyCode.from_char(direction))
            
            if self.logger:
                self.logger.log(f"控制蛇移动: {direction}")
        except Exception as e:
            if self.logger:
                self.logger.log(f"控制蛇移动出错: {str(e)}")
    
    def click_window_center(self):
        """
        点击游戏窗口中心，用于开始游戏或重新开始
        """
        if not self.hwnd:
            if self.logger:
                self.logger.log("未设置游戏窗口，无法点击")
            return
            
        try:
            # 获取窗口客户区位置和大小
            client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.hwnd)
            # 计算中心点
            center_x = client_left + (client_right - client_left) // 2
            center_y = client_top + (client_bottom - client_top) // 2
            
            # 转换为屏幕坐标
            center_x, center_y = win32gui.ClientToScreen(self.hwnd, (center_x, center_y))
            
            # 移动鼠标到中心点并点击
            self.mouse_controller.position = (center_x, center_y)
            time.sleep(0.1)  # 短暂延迟
            self.mouse_controller.press(mouse.Button.left)
            time.sleep(0.1)  # 短暂延迟
            self.mouse_controller.release(mouse.Button.left)
            
            # 记录点击时间
            self.last_click_time = time.time()
            
            if self.logger:
                self.logger.log("点击游戏窗口中心")
        except Exception as e:
            if self.logger:
                self.logger.log(f"点击游戏窗口出错: {str(e)}")
    
    def determine_next_move(self, board_state, path):
        """
        根据当前路径确定下一步移动方向
        :param board_state: 游戏棋盘状态
        :param path: 计划路径
        :return: 移动方向
        """
        if not path or len(path) < 2:
            return None
            
        # 获取蛇头位置
        head_pos = None
        for r in range(len(board_state)):
            for c in range(len(board_state[r])):
                if board_state[r][c] == 'own_head':
                    head_pos = (c, r)
                    break
            if head_pos:
                break
                
        if not head_pos:
            return None
            
        # 在路径中找到当前蛇头位置的索引
        head_index = -1
        for i, pos in enumerate(path):
            if pos == head_pos:
                head_index = i
                break
                
        # 根据蛇头在路径中的位置决定移动方向
        if head_index != -1 and head_index + 1 < len(path):
            next_pos = path[head_index + 1]
            dx = next_pos[0] - head_pos[0]
            dy = next_pos[1] - head_pos[1]
            
            # 根据位置差确定方向
            if dx > 0:
                return 'd'  # 右
            elif dx < 0:
                return 'a'  # 左
            elif dy > 0:
                return 's'  # 下
            elif dy < 0:
                return 'w'  # 上
        
        return None