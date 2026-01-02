import cv2
import numpy as np
from pathlib import Path


def generate_sheet(
    outfile: str = "aruco_4x4_50_4x6_300dpi.png",
    dpi: int = 300,
    page_size_inch=(4, 6),
    rows: int = 3,
    cols: int = 2,
    first_id: int = 0,
):
    """
    生成一张 4x6 英寸相纸大小的 ArUco 码排版图（2x3 共 6 个）。
    默认使用 DICT_4X4_50 字典，从 first_id 开始连续编号。
    """
    aruco = cv2.aruco

    # 选择字典：与你现在那套 4x4 小码一致
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # 页尺寸（像素）
    page_w = int(page_size_inch[0] * dpi)  # 4 inch * 300 = 1200
    page_h = int(page_size_inch[1] * dpi)  # 6 inch * 300 = 1800

    # 创建白底画布（灰度）
    page = np.full((page_h, page_w), 255, dtype=np.uint8)

    # 页面边距和行列间距（可以按需要微调）
    margin_x = int(page_w * 0.10)   # 左右各 10%
    margin_y = int(page_h * 0.08)   # 上下各 8%

    # 计算单个 marker 的边长（像素），保证摆得下 rows x cols
    available_w = page_w - (cols + 1) * margin_x
    available_h = page_h - (rows + 1) * margin_y
    marker_size = int(min(available_w / cols, available_h / rows))

    print(f"Page size: {page_w}x{page_h} px @ {dpi} dpi")
    print(f"Marker size: {marker_size} px (~ {marker_size / dpi * 2.54:.1f} cm)")

    id_now = first_id

    for r in range(rows):
        for c in range(cols):
            # 生成 ArUco 图片
            try:
                # 新版 OpenCV（4.7+）
                marker = aruco.generateImageMarker(dictionary, id_now, marker_size)
            except AttributeError:
                # 兼容旧版 OpenCV（4.6 及以前）
                marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
                aruco.drawMarker(dictionary, id_now, marker_size, marker, 1)

            # 计算在大画布上的放置位置（左上角坐标）
            x = margin_x + c * (marker_size + margin_x)
            y = margin_y + r * (marker_size + margin_y)

            page[y : y + marker_size, x : x + marker_size] = marker

            # 在下面写 ID，方便识别
            text = f"id={id_now}"
            text_y = y + marker_size + int(margin_y * 0.6)
            if text_y < page_h - 10:  # 防止越界
                cv2.putText(
                    page,
                    text,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    0,      # 黑色
                    2,
                    cv2.LINE_AA,
                )

            id_now += 1

    # 保存为 PNG（无损）
    out_path = Path(outfile).resolve()
    cv2.imwrite(str(out_path), page)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_sheet()
