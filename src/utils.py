import json
import random, os
import numpy as np
import torch
from PIL import Image
import cv2
from typing import List, Tuple
import uuid

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def pil_to_cv2(img: Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def is_coor_valid(img: np.ndarray, box_location: tuple, box_size: tuple, angle: float, save_path:str='test/') -> bool:
    save_path = save_path + str(uuid.uuid4()) + '.jpg'
    img_h, img_w = img.shape[:2]
    x, y = box_location
    box_w, box_h = box_size

    cx = x + box_w / 2.0
    cy = y + box_h / 2.0

    corners = np.array([
        [-box_w/2, -box_h/2],
        [ box_w/2, -box_h/2],
        [ box_w/2,  box_h/2],
        [-box_w/2,  box_h/2]
    ])

    theta = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    rotated_corners = corners.dot(rotation_matrix.T)
    abs_corners = rotated_corners + np.array([cx, cy])
    valid = True
    for (px, py) in abs_corners:
        if px < 0 or px > img_w or py < 0 or py > img_h:
            valid = False
            break
    if save_path is not None:
        img_copy = img.copy()
        pts = abs_corners.reshape((-1, 1, 2)).astype(np.int32)
        # Nếu box hợp lệ, vẽ với màu xanh; nếu không, vẽ với màu đỏ
        color = (0, 255, 0) if valid else (0, 0, 255)
        cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=2)
        cv2.imwrite(save_path, img_copy)
    return valid

import cv2
import numpy as np
import matplotlib.pyplot as plt



def putText(img, text, font_size, color, position, thickness, angle, font=cv2.FONT_HERSHEY_SIMPLEX):
  (text_w, text_h), baseline = cv2.getTextSize(text, font, font_size, thickness)
  
  (x_min_text, y_min_text) = (position[0], position[1] - text_h)
  (x_max_text, y_max_text) = (position[0] + text_w, position[1] + baseline)
  x_center_text = (x_min_text + x_max_text) // 2
  y_center_text = (y_min_text + y_max_text) // 2
  R = np.sqrt((x_center_text - x_min_text) ** 2 + (y_center_text - y_min_text) ** 2)
  
  if R - int(R) >= 0.5:
    R = int(R) + 1
  else:
    R = int(R)
  
  (x_square_min_text, y_square_min_text) = (x_center_text - R, y_center_text - R)
  (x_square_max_text, y_square_max_text) = (x_center_text + R, y_center_text + R)
    
  
  text_black_img = np.zeros((2 * R, 2 * R, 3), dtype=np.uint8)
  position_new = (R - text_w // 2, R + text_h // 2)
  cv2.putText(text_black_img, text, position_new, font, font_size, (1, 1, 1), thickness)
  
  center = (R, R)
  rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated_image = cv2.warpAffine(text_black_img, rotation_matrix, (2*R, 2*R), 
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(0, 0, 0))


  
  cv2.imwrite("text_black_img.png", rotated_image)

  patch_tmp = img[y_square_min_text:y_square_max_text, x_square_min_text:x_square_max_text, :].copy()
  
  mask_ones = np.ones_like(patch_tmp, dtype=np.uint8)
  img[y_square_min_text:y_square_max_text, x_square_min_text:x_square_max_text, :] = patch_tmp * (mask_ones - rotated_image) + patch_tmp * rotated_image * color
  cv2.imwrite("rotated.png", img)
        
        
    
    

    