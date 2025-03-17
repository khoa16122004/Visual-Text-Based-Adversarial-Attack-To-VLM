import numpy as np
import torch
import torch.nn as nn
from paddleocr import PaddleOCR, draw_ocr
import cv2
from utils import pil_to_cv2
from model import OCR

class Fitness:
    def __init__(self, 
                org_img,
                clf_model,
                ocr_model,
                gt_text,):
        self.gt_text = gt_text
        self.clf_model = clf_model
        self.ocr_model = ocr_model
        self.org_img = org_img
    
    def L2(self, roi, mask):
        return np.mean((roi - mask) ** 2)

    def PSNR(self, roi, mask):
        l2 = self.L2(roi, mask)
        return l2, 10 * np.log10(255**2 / l2)

    def OCR(self, adv_img, adv_text, box_coor):
        adv_img = pil_to_cv2(adv_img)
        conf = self.ocr_model.evaluate(adv_img)
        if len(conf) == 0:
            return 0, 0, None
        # conf = [line for line in res] # list of [text, confidence]
        encoded_adv = self.clf_model.text_encode(adv_text)
        encoded_adv = encoded_adv.to('cuda')
        def compute_iou(box1, box2):
            ix1 = max(box1[0], box2[0])
            iy1 = max(box1[1], box2[1])
            ix2 = min(box1[2], box2[2])
            iy2 = min(box1[3], box2[3])
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area else 0

        max_iou = 0
        max_det = None
        for det in conf:
            pts = det[0]  
            ocr_box = [min(pt[0] for pt in pts),
                       min(pt[1] for pt in pts),
                       max(pt[0] for pt in pts),
                       max(pt[1] for pt in pts)]
            iou = compute_iou(box_coor, ocr_box)
            if iou > max_iou:
                max_iou = iou
                max_det = det
        
        max_text, max_conf = max_det[1][0], max_det[1][1]
        encoded_ocr = self.clf_model.text_encode([max_text])
        encoded_ocr = encoded_ocr.to("cuda")
        sim_text = self.clf_model.cos(encoded_adv, encoded_ocr)
        
        return sim_text, max_conf, max_det
    
    def ADV(self, individual, adv_text):

        # print("BUG INDIVIDUAL", individual)
        watermarked, box_text, box_org, box_coor = individual.add_text_to_image(self.org_img, adv_text)
        # sim(adv, gt) >< sim(adv, content)
        adv_sim_score = self.clf_model.evaluate(watermarked, adv_text)
        gt_sim_score = self.clf_model.evaluate(watermarked, self.gt_text)
        success = adv_sim_score > gt_sim_score

        sim_text, max_conf, max_det = self.OCR(adv_img=watermarked, 
                                               adv_text=adv_text, 
                                               box_coor=box_coor)
        ocr_res = sim_text + max_conf
        return {
            "success": success,
            "PSNR": self.PSNR(box_text, box_org),
            "sim_text": sim_text,
            "max_conf": max_conf,
            "best_det": max_det,
            "ocr_res": ocr_res,
            "fitness_score": adv_sim_score - gt_sim_score + ocr_res
        }
        # return  success, self.PSNR(roi, mask), sim_text, max_conf, max_det, adv_sim_score - gt_sim_score + ocr_res


if __name__ == "__main__":
    import clip
    from PIL import Image
    from model import CLIP
    from individual import TextIndividual
    # img_path = "D:\codePJ\RESEARCH\Flow-Based-Attack-To-VLM\src\images\lionsea.jpg"
    img_path = "D:\codePJ\RESEARCH\Visual-Text-Based-Adversarial-Attack-To-VLM\images\dog.jpg"
    img = cv2.imread(img_path)
    # print("IMG", img)
    model = CLIP(model_name="ViT-B/16")
    ocr = OCR()
    individual = TextIndividual(
        content="A fox", 
        # color=color, 
        location=(600, 100), 
        box_size=(200, 200),
        angle=-10,
        blend_factor=0.9
    )
    # text detection location (300, 230, 300+260, 230+100)
    adv_img, _ = individual.add_text_to_image(img)
    import matplotlib.pyplot as plt
    adv_img.save("test/test_adv_img_5.jpg")
    print("Saved adv_img as adv_img.jpg")
    f = Fitness(
        org_img=img,
        clf_model=model,
        ocr_model=ocr,
        gt_text="A lion sea is looking at the sea"
    )
    print("ADV", f.ADV(individual))
    # print("OCR", f.OCR(adv_img, individual))