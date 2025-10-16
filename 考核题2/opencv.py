# detect_blue_regions_video.py
# 作用：在 video.mp4 中按“蓝色”直接框选所有目标（瓶盖、瓶身都会各自被框），不做任何文案标注。
# 依赖：pip install opencv-python numpy

import cv2
import numpy as np
import os

# ========== 基本路径 ==========
INPUT_VIDEO  = "video.mp4"
OUTPUT_VIDEO = "outputs/blue_regions.mp4"
SNAP_DIR     = "outputs/snaps"
SHOW_WINDOW  = True  # 如不想弹窗显示改为 False

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# ========== 打开视频 ==========
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))
snap_every = max(1, int(fps * 2.0))  # 每 ~2 秒截一张图

# ========== 可调参数（只做蓝色分割 + 轻去噪 + 基本过滤）==========
# 颜色阈值（HSV；H 0-180）
LOWER_BLUE = np.array([85,  70,  60])   # 偏宽一些，包含浅/深蓝
UPPER_BLUE = np.array([135, 255, 255])

# 形态学去噪（不要把瓶盖和瓶身连桥，所以强度适中）
KERNEL = np.ones((3, 3), np.uint8)
CLOSE_ITERS, OPEN_ITERS = 1, 1

# 轮廓过滤（相对整帧像素），避免把极小噪点/大块背景框进来
MIN_AREA_RATIO = 0.0004
MAX_AREA_RATIO = 0.12
SOLIDITY_MIN   = 0.60   # 实体度（area/hull_area）过滤破碎区域

# 可选：按面积只保留最大的前K个蓝色区域，0 表示不过滤
TOP_K = 0
# ===========================================================

frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_id += 1
    vis = frame.copy()

    # 1) 颜色分割（BGR -> HSV）
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # 2) 形态学去噪（轻度）
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=CLOSE_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL, iterations=OPEN_ITERS)

    # 3) 查找外轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4) 面积过滤 + （可选）只保留前K大
    min_area = H * W * MIN_AREA_RATIO
    max_area = H * W * MAX_AREA_RATIO
    contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    if TOP_K > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:TOP_K]

    # 5) 实体度过滤并画框（红色），不区分盖/身、不显示文字
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) or 1
        if area / hull_area < SOLIDITY_MIN:
            continue

        # 旋转最小外接矩形（更贴合）
        rect = cv2.minAreaRect(c)              # ((cx,cy),(rw,rh),angle)
        box  = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)

    # 6) 写帧/显示/截图
    writer.write(vis)
    if SHOW_WINDOW:
        cv2.imshow("Blue Regions (no cap/body split)", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break
    if frame_id % snap_every == 0:
        cv2.imwrite(os.path.join(SNAP_DIR, f"snap_{frame_id:06d}.png"), vis)

cap.release()
writer.release()
if SHOW_WINDOW:
    cv2.destroyAllWindows()

print("✅ 完成！")
print("输出视频：", os.path.abspath(OUTPUT_VIDEO))
print("截图目录：", os.path.abspath(SNAP_DIR))
