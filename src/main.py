import argparse
import numpy as np
from utils import seed_everything
from model import OpenCLIP, CLIP, OCR
from algorithm import POPOP
import cv2
from fitness import Fitness
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--max_iter', type=int, default=100, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--model_name', type=str, default='ViT-B/16')
    parser.add_argument("--cross_rate", type=float, default=0.9)
    parser.add_argument("--mutation_rate", type=float, default=0.9)
    parser.add_argument("--annotation_file", type=str, default='D:\codePJ\RESEARCH\Visual-Text-Based-Adversarial-Attack-To-VLM\images\new_annotations.csv')
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--arch", type=str, choices=['clip', 'openclip'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    success_rate = 0
    seed_everything(args.seed)
    
    img_path = r"D:\codePJ\RESEARCH\Visual-Text-Based-Adversarial-Attack-To-VLM\images\img1.jpg"    
    img = cv2.resize(cv2.imread(img_path), (224, 224))
    vlm = CLIP(args.model_name)
    ocr = OCR()
    fitness = Fitness(org_img=img,
                        clf_model=vlm,
                        ocr_model=ocr,
                        gt_text='A baby girl in an orange dress gets wet as she stands next to a water sprinkler')
    
    popop = POPOP(
        i=0,
        population_size=args.pop_size,
        f_fit=fitness,
        cross_rate=args.cross_rate,
        model=vlm,
        org_img=img,
        mutation_rate=args.mutation_rate,
        generations=args.max_iter,
        gt_text='A baby girl in an orange dress gets wet as she stands next to a water sprinkler',
        adv_text="two white dogs"
    )
    best_individual = popop.run()
    popop.save_log('popop.txt')


    
    
    # with open(args.annotation_file, "r") as f:
    #     lines = [line.strip().split("\t") for line in f.readlines()]
    
    # for i, [img_id, c_gt, c_tar] in enumerate(lines):
    #     img_path = os.path.join(args.img_dir, img_id)
        
    #     img = cv2.resize(cv2.imread(img_path), (224, 224))
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     if args.arch == 'clip':
    #         vlm = CLIP(args.model_name)
    #     elif args.arch == 'openclip':
    #         vlm = OpenCLIP(args.model_name)

    #     fitness = Fitness(org_img=img,
    #                       model=vlm,
    #                       gt_text=c_gt)
        
    #     popop = POPOP(
    #         i=i,
    #         population_size=args.pop_size,
    #         f_fit=fitness,
    #         cross_rate=args.cross_rate,
    #         model=vlm,
    #         org_img=img,
    #         mutation_rate=args.mutation_rate,
    #         generations=args.max_iter,
    #         gt_text=c_gt,
    #         adv_text="A fox"
    #     )
        
    #     best_individual = popop.run()
    #     popop.save_log('popop.txt')
    #     # break        
    
    # print("Success rate: ", success_rate / 100)  