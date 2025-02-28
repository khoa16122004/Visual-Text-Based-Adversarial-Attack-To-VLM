import cv2
import numpy as np
import math

def draw_caption(img, text, text_org, angle, font_scale=1.0, color=(0, 0, 255),
                 font_face=cv2.FONT_HERSHEY_COMPLEX, thickness=2):
    """
    Draw rotated text on the image.
    
    Parameters:
      img       : Input image (numpy array).
      text      : The caption text.
      text_org  : Tuple (x, y) where the text should be placed.
      angle     : Rotation angle in radians.
      font_scale: Scale factor for the font (controls size).
      color     : Text color in BGR (e.g., (0,0,255) for red).
      font_face : OpenCV font type (default: cv2.FONT_HERSHEY_COMPLEX).
      thickness : Thickness of the text stroke.
      
    Returns:
      img_with_cap: Image with the rotated text blended in.
    """
    # Get text size and baseline
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    
    # Compute a center position for the text on a blank canvas
    text_center = ((img.shape[1] - text_width) // 2, (img.shape[0] + text_height) // 2)
    
    # Create a blank image to draw the text
    img_txt = np.zeros_like(img)
    cv2.putText(img_txt, text, text_center, font_face, font_scale, color, thickness, cv2.LINE_AA)
    
    # Compute the rotation matrix for the affine transform
    cosA = math.cos(angle)
    sinA = math.sin(angle)
    tx = text_org[0] - (cosA * text_center[0] - sinA * text_center[1])
    ty = text_org[1] - (sinA * text_center[0] + cosA * text_center[1])
    M = np.array([[cosA, -sinA, tx],
                  [sinA,  cosA, ty]], dtype=np.float32)
    
    # Rotate the text image so the text is at the desired location and angle
    img_dst = cv2.warpAffine(img_txt, M, (img.shape[1], img.shape[0]))
    
    # Create a mask from the text (white text on black background)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.putText(mask, text, text_center, font_face, font_scale, 255, thickness, cv2.LINE_AA)
    mask = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))
    
    # Combine the rotated text with the original image using the mask.
    # This replicates the bitwise_and blend from the C++ code.
    img_with_cap = cv2.bitwise_and(img, img_dst, mask=mask)
    
    return img_with_cap

# Example usage:
if __name__ == '__main__':
    # Load the image using cv2
    img = cv2.imread("resize.png")
    if img is None:
        print("Image not found.")
    else:
        text = "A. Einstein"
        angle = -math.pi / 4 
        text_org = (100, 100) 
        font_scale = 1.0      
        color = (0, 0, 255)   
        font_face = cv2.FONT_HERSHEY_COMPLEX
        
        img_with_caption = draw_caption(img, text, text_org, angle, font_scale, color, font_face)

        cv2.imread("resize.png", img_with_caption
