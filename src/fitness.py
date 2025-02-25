import numpy as np
import torch
import torch.nn as nn

class Fitness:
    def __init__(self, 
                org_img,
                model,
                gt_text,):
        self.gt_text = gt_text
        self.model = model
        self.org_img = org_img
    def get_roi(self, loc, box):
        x, y = loc
        w, h = box
        np_img = np.array(self.org_img)
        roi = np_img[y:y+h, x:x+w]
        return roi
    
    def L2(self, roi, mask):
        return np.mean((roi - mask) ** 2)

    def PSNR(self, roi, mask):
        l2 = self.L2(roi, mask)
        return l2, 10 * np.log10(255**2 / l2)

    def ADV(self, individual):
        loc = individual.location
        box = individual.box_size
        roi = self.get_roi(loc, box)
        adv_text = individual.content
        watermarked, mask = individual.add_text_to_image(self.org_img)
        # sim(adv, gt) >< sim(adv, content)
        adv_sim_score = self.model.evaluate(watermarked, adv_text)
        gt_sim_score = self.model.evaluate(watermarked, self.gt_text)
        return  self.PSNR(roi, mask), adv_sim_score - gt_sim_score