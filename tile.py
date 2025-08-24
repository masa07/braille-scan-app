from ultralytics import YOLO

def tile_predect(model, image_path, tile_size=640, overlap=0.2, image=None):
    # モデルをロード
    # model = YOLO('yolov8n.pt')

    # タイルに分割
    # image_path = 'path/to/your/large_image.jpg'
    # tiles = split_image_into_tiles(image_path, tile_size=640, overlap=0.2)
    tiles = split_image_into_tiles(image_path, tile_size, overlap, image)

    all_results = []
    for tile in tiles:
        # タイルで推論を実行
        results = model(tile['image'], verbose=False)
        
        # 検出結果を元の画像座標に変換
        for res in results:
            for box in res.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0]
                
                # 座標をタイルの位置分ずらす
                x_offset, y_offset, _, _ = tile['box']
                original_x_min = x_min + x_offset
                original_y_min = y_min + y_offset
                original_x_max = x_max + x_offset
                original_y_max = y_max + y_offset
                
                all_results.append({
                    'box': (original_x_min, original_y_min, original_x_max, original_y_max),
                    'conf': box.conf[0],
                    'cls': box.cls[0]
                })
    return all_results

from PIL import Image
import os
import math

def split_image_into_tiles(image_path, tile_size, overlap, image=None):
    # img = Image.open(image_path)
    if image is not None:
        img = Image.fromarray(image)
    else:
        img = Image.open(image_path)
    width, height = img.size
    tiles = []
    
    # オーバーラップをピクセル単位に変換
    overlap_pixels = int(tile_size * overlap)
    
    # Y軸方向のループ
    for y in range(0, height, tile_size - overlap_pixels):
        # X軸方向のループ
        for x in range(0, width, tile_size - overlap_pixels):
            # タイル領域を定義
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tile = img.crop(box)
            tiles.append({'image': tile, 'box': box})
    return tiles

import numpy as np
from ultralytics.utils.ops import non_max_suppression
import torch

def nms_predictions(all_results):
    # 検出結果をnumpy配列に変換
    if all_results:
        boxes   = np.array([r['box'] for r in all_results], dtype=np.float32)
        scores  = np.array([[r['conf']] for r in all_results], dtype=np.float32)
        classes = np.array([[r['cls']]  for r in all_results], dtype=np.float32)
        detections = np.hstack((boxes, scores, classes))
    else:
        detections = np.empty((0, 6), dtype=np.float32)

    # NMSを実行 - 修正箇所: detectionsをリストでラップ
    nms_results = non_max_suppression(torch.tensor(detections).unsqueeze(0), conf_thres=0.25, iou_thres=0.45)
    final_detections = nms_results[0]

    return final_detections

from ultralytics import YOLO
from PIL import Image, ImageDraw
def visualize_detections(image_path, detections, model_names, image=None):
    # img = Image.open(image_path).convert("RGB")
    if image is not None:
        img = Image.fromarray(image)
    else:
        img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    labels = []

    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det[:4])
        confidence = det[4].item()  # Tensorの場合は.item()で値を取得
        class_id = int(det[5])        
        class_name = model_names.get(class_id, str(class_id))
        label = f"{class_name}: {confidence:.2f}"

        # Draw bounding box
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)

        # Draw label background
        text_x = x_min
        text_y = y_min - 10
        draw.rectangle([(text_x, text_y - 10), (text_x + len(label) * 10, text_y)], fill="red")

        # Draw label text
        draw.text((text_x, text_y - 10), label, fill="white")

        labels.append(label)

    return img, labels