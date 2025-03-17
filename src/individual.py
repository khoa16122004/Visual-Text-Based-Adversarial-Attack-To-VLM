from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import random
import math
from PIL import ImageDraw, ImageFont, ImageOps
from PIL import Image as PILImage
from PIL.Image import Image
import numpy as np

@dataclass
class TextIndividual:
    location: Tuple[int, int]
    font_size: int = 32
    angle: int = 0
    blend_factor: float = 0.9

    def blend_colors(self, target_color, blend_factor):
        r = int(target_color[0] * (blend_factor))
        g = int(target_color[1] * (blend_factor))
        b = int(target_color[2] * (blend_factor))
        return (r, g, b)

    def get_average_color_from_mask(self, mask):
        if mask.size == 0:
            assert False, f"Mask is empty with shape {mask.shape}."
        avg_color = np.mean(mask, axis=(0, 1)).astype(int)
        if len(mask.shape) == 3:
            avg_color = self.blend_colors(avg_color, self.blend_factor)
            return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
        else:
            avg_color = np.mean(mask)
            avg_color = self.blend_colors(avg_color, self.blend_factor)
            return (int(avg_color), int(avg_color), int(avg_color))

    def add_text_to_image(self, img: np.ndarray, text: str) -> Tuple[Image, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        if isinstance(img, np.ndarray):
            original_img = img.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                original_pil = PILImage.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = PILImage.fromarray(img)
                original_pil = PILImage.fromarray(original_img)
        else:
            pil_img = img.copy()
            original_pil = img.copy()
        
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
            original_pil = original_pil.convert('RGBA')
        
        txt_img = PILImage.new('RGBA', pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            font = ImageFont.load_default()
        
        words = text.split()
        word_positions = []
        word_sizes = []
        
        current_position = self.location
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=font)
            textwidth = bbox[2] - bbox[0]
            textheight = bbox[3] - bbox[1]
            x = int(current_position[0] - textwidth / 2)
            y = int(current_position[1] - textheight / 2)
            
            word_positions.append((x, y))
            word_sizes.append((textwidth, textheight))
            current_position = (current_position[0] + textwidth + 10, current_position[1])
        
        binary_mask = PILImage.new('L', pil_img.size, 0)
        binary_mask_draw = ImageDraw.Draw(binary_mask)
        
        for i, word in enumerate(words):
            x, y = word_positions[i]
            binary_mask_draw.text((x, y), word, font=font, fill=255)
        
        rotated_binary_mask = binary_mask.rotate(self.angle, expand=0, center=self.location)
        rotated_binary_mask_np = np.array(rotated_binary_mask)
        y_indices, x_indices = np.where(rotated_binary_mask_np > 0)
        
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(pil_img.width - 1, x_max + padding)
            y_max = min(pil_img.height - 1, y_max + padding)
            
            box_width = x_max - x_min + 1
            box_height = y_max - y_min + 1
            box_coords = (x_min, y_min, x_max, y_max)
        else:
            box_coords = (0, 0, pil_img.width - 1, pil_img.height - 1)
        
        rotated_word_masks = []
        word_colors = []
        for i, word in enumerate(words):
            x, y = word_positions[i]
            word_mask = PILImage.new('L', pil_img.size, 0)
            word_mask_draw = ImageDraw.Draw(word_mask)
            word_mask_draw.text((x, y), word, font=font, fill=255)
            rotated_word_mask = word_mask.rotate(self.angle, expand=0, center=self.location)
            rotated_word_masks.append(rotated_word_mask)
            rotated_mask_np = np.array(rotated_word_mask)
            y_indices, x_indices = np.where(rotated_mask_np > 0)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(pil_img.width - 1, x_max)
                y_max = min(pil_img.height - 1, y_max)
                
                img_np = np.array(pil_img)
                region = img_np[y_min:y_max+1, x_min:x_max+1]
                mask_region = rotated_mask_np[y_min:y_max+1, x_min:x_max+1]
                
                if region.size > 0 and mask_region.size > 0:
                    if len(region.shape) == 3:
                        mask_3d = np.stack([mask_region] * region.shape[2], axis=2)
                        masked_region = region * (mask_3d > 0)
                        non_zero = mask_3d > 0
                        if np.any(non_zero):
                            region_sum = np.sum(masked_region, axis=(0, 1))
                            pixel_count = np.sum(non_zero[:,:,0])
                            avg_color = region_sum / pixel_count
                            avg_color = self.blend_colors(avg_color, self.blend_factor)
                        else:
                            avg_color = (255, 255, 255)
                    else:
                        masked_region = region * (mask_region > 0)
                        non_zero = mask_region > 0
                        if np.any(non_zero):
                            avg_color = np.sum(masked_region) / np.sum(non_zero)
                            avg_color = int(self.blend_colors(avg_color, self.blend_factor)[0])
                            avg_color = (avg_color, avg_color, avg_color)
                        else:
                            avg_color = (255, 255, 255)
                else:
                    avg_color = (255, 255, 255)
            else:
                avg_color = (255, 255, 255)
                
            word_colors.append(avg_color)
        
        for i, word in enumerate(words):
            x, y = word_positions[i]
            color = word_colors[i]
            draw.text((x, y), word, font=font, fill=color)
        
        txt_img = txt_img.rotate(self.angle, expand=0, center=self.location)
        result = PILImage.alpha_composite(pil_img, txt_img)
        
        x_min, y_min, x_max, y_max = box_coords
        original_box = original_pil.crop(box_coords)
        result_box = result.crop(box_coords)
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 3:
                return (
                    # cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR),
                    # PILImage.fromarray(result.convert('RGB')),
                    result,
                    cv2.cvtColor(np.array(result_box.convert('RGB')), cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(np.array(original_box.convert('RGB')), cv2.COLOR_RGB2BGR),
                    box_coords 
                )
            else:
                return (
                    # PILImage.fromarray(np.array(result.convert('RGB'))),
                    result,
                    np.array(result_box.convert('RGB')),
                    np.array(original_box.convert('RGB')),
                    box_coords
                )
        else:
            return result, result_box, original_box, box_coords

if __name__ == "__main__":
    img = cv2.imread(r"D:\codePJ\RESEARCH\Visual-Text-Based-Adversarial-Attack-To-VLM\images\lionsea.jpg")
    img = cv2.resize(img, (375, 375))
    text_test = "A fox"
    location = (155, 50)
    random_angle = 45
    random_blend = 0.99
    
    individual = TextIndividual(
        location=location,
        font_size=120,
        angle=random_angle,
        blend_factor=random_blend
    )
    
    img_test_aft, box1, box2, coor = individual.add_text_to_image(img=img, text=text_test)
    cv2.imshow("Rotated text", img_test_aft)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Converted Mask", box1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Background Mask", box2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(coor)
