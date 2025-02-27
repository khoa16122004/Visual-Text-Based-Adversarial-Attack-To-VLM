import argparse
import numpy as np
from utils import seed_everything
from model import OpenCLIP
from algorithm import POPOP
import cv2
from fitness import Fitness
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--max_iter', type=int, default=100, help="Number of iterations for the genetic algorithm")
    parser.add_argument('--seed', type=int, default=22520691)
    parser.add_argument('--model_name', type=str, default='ViT-H-14')
    parser.add_argument("--cross_rate", type=float, default=0.9)
    parser.add_argument("--mutation_rate", type=float, default=0.9)
    parser.add_argument("--annotation_file", type=str)
    parser.add_argument("--img_dir", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    success_rate = 0
    seed_everything(args.seed)

    
    
    with open(args.annotation_file, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
    
    for i, [img_id, c_gt, c_tar] in enumerate(lines):
        img_path = os.path.join(args.img_dir, img_id)
        
        img = cv2.resize(cv2.imread(img_path), (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        vlm = OpenCLIP(args.model_name)

        fitness = Fitness(org_img=img,
                          model=vlm,
                          gt_text=c_gt)
        
        popop = POPOP(
            population_size=args.pop_size,
            f_fit=fitness,
            cross_rate=args.cross_rate,
            model=vlm,
            org_img=img,
            mutation_rate=args.mutation_rate,
            generations=args.max_iter,
            gt_text=c_gt,
            adv_text=c_tar
        )
        
        break        
    
    print("Success rate: ", success_rate / 100)  