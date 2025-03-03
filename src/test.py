import cv2
import numpy as np

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
        


img = cv2.imread("resize.png")
putText(img, "hello", 1, (0, 0, 255), (50, 100), 1, 20)
