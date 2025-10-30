
import cv2
import numpy as np
import os


INPUT_VIDEO  = "video.mp4"
OUTPUT_VIDEO = "outputs/blue_bottle_split.mp4"
SNAP_DIR     = "outputs/snaps"
SHOW_WINDOW  = True
TOP_BOTTLES  = 4

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)


cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频：{INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))
snap_every = max(1, int(fps * 2.0))


LOWER_BLUE = np.array([85,  110,  200])
UPPER_BLUE = np.array([135, 255, 255])


KERNEL_SMALL = np.ones((3, 3), np.uint8)
CLOSE_ITERS, OPEN_ITERS = 1, 1


MIN_COMP_AREA_RATIO = 0.0004
MAX_COMP_AREA_RATIO = 0.12
SOLIDITY_MIN        = 0.60


X_GROUP_THR_RATIO   = 0.06

CAP_MAX_REL_AREA    = 0.35
MIN_X_OVERLAP_RATIO = 0.25


def x_overlap_ratio(a, b):
    ax1, ax2 = a[0], a[0] + a[2]
    bx1, bx2 = b[0], b[0] + b[2]
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    return inter / max(1, min(a[2], b[2]))

frame_id = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_id += 1
    vis = frame.copy()


    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)


    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_SMALL, iterations=CLOSE_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL_SMALL, iterations=OPEN_ITERS)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = H * W * MIN_COMP_AREA_RATIO
    max_area = H * W * MAX_COMP_AREA_RATIO

    comps = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) or 1
        if area / hull_area < SOLIDITY_MIN:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w / 2, y + h / 2
        comps.append(dict(bbox=(x, y, w, h), area=area, center=(cx, cy), contour=c))


    comps.sort(key=lambda d: d["center"][0])
    groups = []
    for comp in comps:
        placed = False
        for g in groups:

            g_xs = [c["center"][0] for c in g]
            gx = sum(g_xs) / len(g_xs)
            if abs(comp["center"][0] - gx) < X_GROUP_THR_RATIO * W:
                g.append(comp); placed = True; break
        if not placed:
            groups.append([comp])


    bottles = []
    for g in groups:
        if not g:
            continue

        body = max(g, key=lambda d: d["area"])
        bx, by, bw, bh = body["bbox"]

        cap_cand = None
        for c in g:
            if c is body:
                continue
            cx, cy, cw, ch = c["bbox"]
            if c["center"][1] >= body["center"][1]:
                continue
            if x_overlap_ratio(c["bbox"], body["bbox"]) < MIN_X_OVERLAP_RATIO:
                continue
            if c["area"] <= CAP_MAX_REL_AREA * body["area"]:

                if (cap_cand is None or
                    abs(c["center"][0] - body["center"][0]) <
                    abs(cap_cand["center"][0] - body["center"][0])):
                    cap_cand = c

        bottles.append(dict(body=body, cap=cap_cand))


    if TOP_BOTTLES > 0 and len(bottles) > TOP_BOTTLES:
        bottles.sort(key=lambda b: b["body"]["area"], reverse=True)
        bottles = bottles[:TOP_BOTTLES]


    for b in bottles:
        x, y, w, h = b["body"]["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)     # Body 红
        if b["cap"] is not None:
            x2, y2, w2, h2 = b["cap"]["bbox"]
            cv2.rectangle(vis, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 255), 2)  # Cap 黄


    writer.write(vis)
    if SHOW_WINDOW:
        cv2.imshow("Blue Bottle Split (Body=Red, Cap=Yellow)", vis)
        if cv2.waitKey(1) & 0xFF == 27:
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
