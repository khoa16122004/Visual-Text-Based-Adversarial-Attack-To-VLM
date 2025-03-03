from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import random
import math
# import PIL.Image as Image
from PIL import Image as PILImage
from PIL.Image import Image
import numpy as np

@dataclass
class TextIndividual:
    content: str
    location: Tuple[int, int]
    box_size: Tuple[int, int]
    angle: int = 0
    blend_factor: float = 255
    
    def blend_colors(self, target_color, blend_factor):
        r = int(target_color[0] * (blend_factor))
        g = int(target_color[1] * (blend_factor))
        b = int(target_color[2] * (blend_factor))
        return (r, g, b)

    def get_average_color_from_mask(self, mask):
        if mask.size == 0:
            assert False, f"Mask is empty with {self.location} and {self.box_size}"
        avg_color = np.mean(mask, axis=(0, 1)).astype(int) 
        if len(mask.shape) == 3:
            return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
        else:
            avg_color = np.mean(mask)  
            return (int(avg_color), int(avg_color), int(avg_color))

    def get_loc_after_rotate(self, loc, angle):
        x, y, w, h = loc
        top_left = np.array([x, y])
        top_right = np.array([x + w, y])
        bottom_left = np.array([x, y + h])
        bottom_right = np.array([x + w, y + h])
        
        points = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)        
        center = (x + w // 2, y + h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_points = cv2.transform(np.array([points]), rotation_matrix)[0]
        x_min = np.min(rotated_points[:, 0])
        y_min = np.min(rotated_points[:, 1])
        x_max = np.max(rotated_points[:, 0])
        y_max = np.max(rotated_points[:, 1])
        
        width = x_max - x_min
        height = y_max - y_min
        x = int(x_min)
        y = int(y_min)
        
        return x, y, int(width), int(height)
    def rotate_with_padding(self, img, angle_degrees):
        (h, w) = img.shape[:2]
        pad_h = h // 4
        pad_w = w // 4
        padded_img = cv2.copyMakeBorder(
            img, pad_h, pad_h, pad_w, pad_w, 
            borderType=cv2.BORDER_CONSTANT, value=(0,0,0)
        )

        (hp, wp) = padded_img.shape[:2]
        center = (wp//2, hp//2)

        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated = cv2.warpAffine(padded_img, M, (wp, hp))
        
        return rotated, pad_h, pad_w
    def rotate_image(self, img, angle_degrees, pad_h, pad_w):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        h, w = rotated.shape[:2]
        rotated = rotated[pad_h:h-pad_h, pad_w:w-pad_w]
        return rotated
    def add_text_to_image(self, img: np.ndarray) -> Tuple[Image, np.ndarray]:
        h, w = img.shape[:2]
        box_w, box_h = self.box_size
        x, y = self.location

        words = self.content.split()

        total_width = 0
        word_widths = []
        font_scale = 1.0

        for word in words:
            size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            word_width = size[0][0]
            word_widths.append(word_width)
            total_width += word_width
        word_spacing = (box_w - total_width) // (total_width )
        total_width += word_spacing * (len(words) - 1)

        if total_width > 0:
            font_scale = min(box_w / total_width, box_h / (cv2.getTextSize("Tg", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][1])) * 0.8

        actual_word_widths = [int(w * font_scale) for w in word_widths]
        actual_total_width = sum(actual_word_widths) + word_spacing * (len(words) - 1)

        start_x = x + (box_w - actual_total_width) // 2
        center_y = y + box_h // 2

        rotated_img, pad_h, pad_w  = self.rotate_with_padding(img, self.angle)
        
        for i, word in enumerate(words):
            word_size = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            word_width = word_size[0][0]
            word_height = word_size[0][1]

            word_y = center_y - word_height // 2  
            word_x = start_x + sum(word_widths[:i]) + i * word_spacing
            word_loc = (word_x+pad_w, word_y+pad_h, word_width, word_height)

            x_aft, y_aft, w_aft, h_aft = self.get_loc_after_rotate(word_loc, self.angle)

            word_roi = rotated_img[y_aft:y_aft + h_aft, x_aft:x_aft + w_aft]
            color = self.get_average_color_from_mask(word_roi)
            color_blended = self.blend_colors(color, self.blend_factor)

            cv2.putText(word_roi, word, (0, h_aft), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_blended, 2)
            rotated_img[y_aft:y_aft + h_aft, x_aft:x_aft + w_aft] = word_roi
            start_x += word_width + word_spacing
        final_img = self.rotate_image(rotated_img, -self.angle, pad_h, pad_w)
        mask_region = final_img[y:y+box_h, x:x+box_w]
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        final_img = PILImage.fromarray(final_img_rgb)
        return final_img, mask_region

    def add_text_to_image(self, img: np.ndarray) -> Tuple[Image, np.ndarray]:
        img_pil = Image.fromarray(img)



    
    
if __name__ == "__main__":
    test_img_path = r'D:\codePJ\RESEARCH\Flow-Based-Attack-To-VLM\src\images\lionsea.jpg'
    img = cv2.imread(test_img_path)
    text_test = "A fox is jumping"
    img_shape = img.shape
    # color = (255, 0, 0)  
    
    # box_size = (min(random.randint(int(0.25*img_shape[1]), img_shape[1] - 100), img_shape[1]), # width
    #             min(random.randint(int(0.1*img_shape[0]), img_shape[0] - 100), img_shape[0])) # height
    # location = (random.randint(0, img_shape[1] - box_size[0]), 
    #             random.randint(0, img_shape[0] - box_size[1]))
    box_size = (3467, 372)
    location = (99, 718)
    # random_angle = random.randint(-30, 30)
    # random_blend = random.uniform(0.5, 1.0)
    random_angle = -4
    random_blend = 0.9
    individual = TextIndividual(
        content=text_test, 
        # color=color, 
        location=location, 
        box_size=box_size,
        angle=random_angle,
        blend_factor=random_blend
    )
    
    img_test_aft, mask = individual.add_text_to_image(img)
    # Show the PIL Image
    img_test_aft.show()

    # Alternatively, display with matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(img_test_aft)
    plt.axis('off')
    plt.title("Image with Text")
    plt.show()

    cv2.imshow('Mask', cv2.resize(mask, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

