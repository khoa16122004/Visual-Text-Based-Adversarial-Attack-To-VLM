import random
import numpy as np
import cv2
from typing import Tuple
from PIL import Image
from GA.individual import TextIndividual
from timm import create_model, list_models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision import models

MODEL = create_model('vit_base_patch16_clip_224', pretrained=True)

class GA_text:
    def __init__(self,
                pop_size: int,
                org_img: np.ndarray,
                mut: float = 0.1,
                generations: int = 100):
        self.pop_size = pop_size
        self.mut = mut
        self.generations = generations
        self.population = []
        self.logging = {}
        self.org_img = org_img
        self.text = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.initialize_population(org_img.shape[:2])

    def initialize_population(self, img_size: Tuple[int, int]):
        for _ in range(self.pop_size):
            content = ''.join(random.choices(self.text, k=10))
            color = tuple(random.choices(range(256), k=3))
            location = tuple(random.choices(range(img_size[0]), k=2))
            box_size = tuple(random.choices(range(10, 50), k=2))
            self.population.append(TextIndividual(content, color, location, box_size))
    def fitness(self, individual: TextIndividual):
        