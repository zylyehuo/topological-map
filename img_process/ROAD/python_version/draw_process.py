import cv2
import numpy as np

# ================= 配置区域 =================
# 输入刚才画好线的图片，或者原图都可以
INPUT_IMAGE_PATH = '/home/yehuo/topological_map/img_process/ROAD/process/0_clear_process.png'    
OUTPUT_IMAGE_PATH = '/home/yehuo/topological_map/img_process/ROAD/process/0_draw_process.png' 

LINE_COLOR = (0, 0, 0)   # 线条颜色，(0,0,0) 为黑色
LINE_THICKNESS = 4       # 线条粗细
FILL_COLOR = (0, 0, 0)   # 填充颜色，(0,0,0) 为黑色
# ============================================

# 用于存储鼠标左键点击的坐标点
points = []
# 用于存储图像的历史状态，实现撤销功能
history = []

def click_event(event, x, y, flags, params):
    global points, img_copy, history
    
    # 【功能 1】当鼠标左键按下时：画线封闭
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"已记录画线端点: ({x}, {y})")

        # 每收集到两个点（一对），就在原图上绘制一条直线
        if len(points) % 2 == 0:
            # 【重要】在实际修改图片前，把当前状态存入历史记录
            history.append(img_copy.copy())
            
            pt1 = points[-2]
            pt2 = points[-1]
            cv2.line(img_copy, pt1, pt2, LINE_COLOR, LINE_THICKNESS)
            cv2.imshow('Image Editor', img_copy)

    # 【功能 2】当鼠标右键按下时：区域填充（油漆桶）
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"执行区域填充: ({x}, {y})")
        
        # 【重要】在实际修改图片前，把当前状态存入历史记录
        history.append(img_copy.copy())
        
        # 获取图片的宽高
        h, w = img_copy.shape[:2]
        
        # OpenCV 的 floodFill 需要一个比原图大 2 个像素的 mask 矩阵
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 容差范围：由于图片可能有轻微的噪点或抗锯齿边缘，设置一点容差能填得更干净
        lo_diff = (20, 20, 20) # 颜色负向容差
        up_diff = (20, 20, 20) # 颜色正向容差
        
        # 执行泛洪填充
        cv2.floodFill(img_copy, mask, (x, y), FILL_COLOR, lo_diff, up_diff)
        cv2.imshow('Image Editor', img_copy)

# 1. 读取图片
img = cv2.imread(INPUT_IMAGE_PATH)
if img is None:
    print(f"错误：无法读取图片 '{INPUT_IMAGE_PATH}'。请确保图片路径正确。")
    exit()

# 创建一个副本用于显示和绘制
img_copy = img.copy()

# ================= 自适应窗口大小 =================
h, w = img_copy.shape[:2]
max_display_size = 800 
scale = min(max_display_size/w, max_display_size/h)

if scale < 1:
    display_w = int(w * scale)
    display_h = int(h * scale)
else:
    display_w = w
    display_h = h

# 2. 初始化显示窗口并绑定鼠标事件
cv2.namedWindow('Image Editor', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('Image Editor', display_w, display_h) 
# ==================================================

cv2.imshow('Image Editor', img_copy)
cv2.setMouseCallback('Image Editor', click_event)

print("=== 图片编辑器已升级 ===")
print("操作指南：")
print("1. 【画线】用鼠标 左键 点击缺口两端，自动连线封闭边界。")
print("2. 【填充】确认边界封闭后，用鼠标 右键 点击空白区域内部，自动填充为黑色。")
print("3. 【撤销】按下 'z' 键，可以撤销上一次的画线或填充操作。")
print("4. 【保存】完成后，按下 's' 键保存并退出。")
print("5. 【取消】如果操作失误，按下 'q' 键不保存退出，重新运行即可。")
print("------------------------")

# 3. 等待键盘指令
while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        cv2.imwrite(OUTPUT_IMAGE_PATH, img_copy)
        print(f"\n成功！处理后的图片已保存至：{OUTPUT_IMAGE_PATH}")
        break
    elif key == ord('q'):
        print("\n已取消操作，未保存图片。")
        break
    elif key == ord('z'):
        # 撤销逻辑
        if len(points) % 2 != 0:
            # 如果处于“半连线”状态（只点了一个起点），直接取消这个点
            points.pop()
            print("\n撤销：已取消刚才点击的起点。")
        elif len(history) > 0:
            # 如果有历史记录，则恢复上一帧图片
            img_copy = history.pop()
            cv2.imshow('Image Editor', img_copy)
            print("\n撤销：已恢复上一步的图片状态。")
        else:
            print("\n无法撤销：当前已经是最初状态了。")

# 释放窗口资源
cv2.destroyAllWindows()
