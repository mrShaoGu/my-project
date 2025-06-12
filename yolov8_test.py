from ultralytics import YOLO
import cv2

# 加载预训练 YOLOv11 模型（可用 yolo11n.pt, yolo11s.pt 等）
model = YOLO('yolo11n.pt')

# 读取无人机视角图片
img_path = 'your_drone_image.jpg'  # 替换为你的无人机图片路径
img = cv2.imread(img_path)

# 推理
results = model(img)

# 可视化检测结果
annotated_img = results[0].plot()
cv2.imshow('YOLOv11 Drone Detection', annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()