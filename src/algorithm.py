from typing import List, Tuple
import random
from individual import TextIndividual
from model import CLIP
from fitness import Fitness
from settings import (W_ADV, 
                      W_PSNR, 
                      MIN_ANGLE, 
                      MAX_ANGLE, 
                      MARGIN, 
                      MIN_FONT_SIZE, 
                      MAX_FONT_SIZE)
import numpy as np
from PIL import Image
from typing import Tuple
from tqdm import tqdm
import cv2
from utils import is_coor_valid
class GABase:
    '''
    Base class for Genetic Algorithm, including:
    - Initialization of population
    - Fitness calculation
    - Mutation
    - Crossover
    - Parent selection
    - Logging
    '''
    def __init__(self, 
                 i: int, 
                 population_size: int,
                 f_fit: Fitness,
                 model: CLIP,
                 org_img: np.ndarray,
                 mutation_rate: float = 0.2,
                 cross_rate: float = 0.5,
                 generations: int = 100,
                 gt_text: str = '',
                 adv_text: str = '',
                 ):
        self.f_fit = f_fit
        self.cross_rate = cross_rate
        self.model = model
        self.adv_text = adv_text
        self.gt_text = gt_text
        self.population_size = population_size
        self.org_img = org_img
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.logger = []
        self.i = i    
    def create_font_size(self):
        return random.randint(MIN_FONT_SIZE, MAX_FONT_SIZE)
    
    def create_location(self):
        w, h = self.org_img.shape[1], self.org_img.shape[0]
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        return x, y
    
    def create_angle(self):
        return random.randint(MIN_ANGLE, MAX_ANGLE)
    
    def create_blended_fac(self):
        return random.uniform(0.9, 1.0)
    
    def create_individual(self,  text_content):
        font_size = self.create_font_size()
        location = self.create_location()
        angle = self.create_angle()
        blended = 0.9
        return TextIndividual(
            location=location,
            font_size=font_size,
            angle=angle,
            blend_factor=blended
        )
    def initialize_population(self, text_content: str) -> List[TextIndividual]:
        population = []
        for _ in range(self.population_size):
            individual = self.create_individual(text_content)
            population.append(individual)
        return population

    def calculate_fitness(self, individual: TextIndividual) -> float:
        """
            fitness output = {
                'success': bool,
                'PSNR': float,
                'sim_text': float, # sim text is the similarity between the adv text and the OCR text
                'max_conf': float,
                'best_det': List,
                'ocr_res': float,
                'fitness_score': float # fitness_score = adv_sim - gt_sim + ocr_res
            }
            Ex:{
                'success': tensor([True], device='cuda:0'), 
                'PSNR': (3.3874416666666667, 42.83208535675605), 
                'sim_text': tensor([0.9995], device='cuda:0', dtype=torch.float16), 
                'max_conf': 0.7628968954086304, 
                'max_det': [[[592.0, 168.0], [741.0, 154.0], [745.0, 195.0], [596.0, 209.0]], ('A fox', 0.7628968954086304)], 
                'ocr_res': tensor([1.7627], device='cuda:0', dtype=torch.float16), 
                'fitness_score': tensor([1.7852], device='cuda:0', dtype=torch.float16)
            }
        """
        res = self.f_fit.ADV(individual=individual, adv_text=self.adv_text)
        return res["success"], res["fitness_score"], res
            

    def mutate(self, individual: TextIndividual) -> TextIndividual:
        if random.random() < self.mutation_rate:
            # mutate angle
            new_angle = self.create_angle()
            individual.angle = new_angle
        if random.random() < self.mutation_rate:
            # mutate box size# width
            new_font_size = self.create_font_size()
            individual.font_size = new_font_size
        if random.random() < self.mutation_rate:
            # mutate location
            new_loc = self.create_location()
            individual.location = new_loc
        return individual
    def BLX_alpha(self, p1, p2, alpha=0.5):
        d = abs(p1 - p2)
        return np.random.uniform(min(p1, p2) - alpha*d, max(p1, p2) + alpha*d)
    def SBX_crossover(self, p1, p2, eta=2):
        u = np.random.rand()
        beta = (2*u)**(1/(eta+1)) if u <= 0.5 else (1/(2*(1 - u)))**(1/(eta+1))
        
        if isinstance(p1, int):
            child1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            child2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            return child1, child2
        
        child1_x = 0.5 * ((1 + beta) * p1[0] + (1 - beta) * p2[0])
        child2_x = 0.5 * ((1 - beta) * p1[0] + (1 + beta) * p2[0])

        u = np.random.rand()
        beta = (2*u)**(1/(eta+1)) if u <= 0.5 else (1/(2*(1 - u)))**(1/(eta+1))

        child1_y = 0.5 * ((1 + beta) * p1[1] + (1 - beta) * p2[1])
        child2_y = 0.5 * ((1 - beta) * p1[1] + (1 + beta) * p2[1])
        w, h = self.org_img.shape[1], self.org_img.shape[0]
        x_c1, y_c1 = int(np.clip(child1_x, 0, w - 1)), int(np.clip(child1_y, 0, h - 1))
        x_c2, y_c2 = int(np.clip(child2_x, 0, w - 1)), int(np.clip(child2_y, 0, h - 1))

        return (x_c1, y_c1), (x_c2, y_c2)
    def crossover(self, parent1: TextIndividual, parent2: TextIndividual) -> Tuple[TextIndividual, TextIndividual]:
        if random.random() < self.cross_rate:
            child1_location = parent2.location
            child2_location = parent1.location
        else:
            child1_location, child2_location = self.SBX_crossover(parent1.location, parent2.location)
        # Font size crossover
        if random.random() < self.cross_rate:
            # Swap box sizes
            child1_font_size = parent2.font_size
            child2_font_size = parent1.font_size
        else:
            child1_font_size, child2_font_size = self.SBX_crossover(parent1.font_size, parent2.font_size)
            
        # Angle crossover
        if random.random() < self.cross_rate:
            # Swap angles
            child1_angle = parent2.angle
            child2_angle = parent1.angle
        else:
            child1_angle = self.BLX_alpha(parent1.angle, parent2.angle)
            child2_angle = self.BLX_alpha(parent2.angle, parent1.angle)

        # Blend factor crossover
        if random.random() < self.cross_rate:
            # Swap blend factors
            child1_blend_factor = parent2.blend_factor
            child2_blend_factor = parent1.blend_factor
        else:
            child1_blend_factor = self.BLX_alpha(parent1.blend_factor, parent2.blend_factor)    
            child2_blend_factor = self.BLX_alpha(parent2.blend_factor, parent1.blend_factor)

        child1 = TextIndividual(
            location=child1_location,
            font_size=child1_font_size,
            angle=child1_angle,
            blend_factor=child1_blend_factor
        )
        
        child2 = TextIndividual(
            location=child2_location,
            font_size=child2_font_size,
            angle=child2_angle,
            blend_factor=child2_blend_factor
        )
        
        return child1, child2

    def select_parents(self, population: List[TextIndividual], 
                      fitness_scores: List[float]) -> List[TextIndividual]:
        # Tournament selection
        tournament_size = 3
        parents = []
        
        while len(parents) < len(population):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
            
        return parents
    def logging(self, individual_dict, save_img=False):
        root = f'results/{self.i}'
        import os
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

        last_best = self.logger[-1] if self.logger else None
        if last_best:
            if last_best['fitness'] < individual_dict['fitness']:
                if save_img:
                    individual_dict['adv'].save(f'{root}/{individual_dict["id"]}.png')   
                    del individual_dict['adv']
                self.logger.append(individual_dict)
        else:
            if save_img:
                individual_dict['adv'].save(f'{root}/{individual_dict["id"]}.png')
            del individual_dict['adv']
            self.logger.append(individual_dict)
        return
    def save_log(self, path):
        result_root = f'results/{self.i}'
        import os
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        path = os.path.join(result_root, path)
        with open(path, 'w') as f:
            for log in self.logger:
                f.write(f"{log}\n")
        return

class POPOP(GABase):
    def __init__(self, 
                 i: int,
                 population_size: int,
                 f_fit: Fitness,
                 model: CLIP,
                 cross_rate: float,
                 org_img: np.ndarray,
                 mutation_rate: float = 0.2,
                 generations: int = 100,
                 gt_text: str = '',
                 adv_text: str = ''):
        super().__init__(
            i=i,
            population_size=population_size,
            f_fit=f_fit,
            model=model,
            org_img=org_img,
            mutation_rate=mutation_rate,
            cross_rate=cross_rate,
            generations=generations,
            gt_text=gt_text,
            adv_text=adv_text
        )

    def run(self) -> TextIndividual:
        population = self.initialize_population(self.adv_text)
        print(f"\nDone initializing population")
        best_individual = None
        best_fitness = float('-inf')

        for gen in tqdm(range(self.generations), desc="Running generations"):
            fitness_scores = []
            _success = []
            for ind in population:
                success, fitness, res = self.calculate_fitness(ind)
                _success.append(success)
                fitness_scores.append(fitness.cpu().item())
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx]
                best_success = _success[current_best_idx]
                individual_params = best_individual.__dict__
                self.logging({
                    'id': gen,
                    'success': best_success,
                    'fitness': best_fitness,
                    'fitness_res': {
                        'success': res["success"].cpu().item(),
                        'PSNR': res["PSNR"],
                        # 'sim_text': res["sim_text"].cpu().item(),
                        'sim_text': res["sim_text"],
                        'max_conf': res["max_conf"],
                        'best_det': res["best_det"],
                        # 'ocr_res': res["ocr_res"].cpu().item(),
                        'ocr_res': res["ocr_res"],
                        'fitness_score': res["fitness_score"].cpu().item()
                    },
                    'adv': best_individual.add_text_to_image(self.org_img, self.adv_text)[0],
                    'params': individual_params
                }, save_img=True)
            parents = self.select_parents(population, fitness_scores)

            offspring = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[(i+1) % len(parents)]

                c1, c2 = self.crossover(p1, p2)
                # print(f"\nCrossover: Offspring at {gen}: {c1}, {c2}\n")

                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                # print(f"\nMutation: Offspring at {gen}: {c1}, {c2}\n")

                offspring.append(c1)
                offspring.append(c2)
            # print(f"\nOffspring at {gen}: {offspring}")
            offspring_fitness_scores = [self.calculate_fitness(ind)[1] for ind in offspring]

            combined_population = population + offspring
            combined_fitness = fitness_scores + offspring_fitness_scores

            sorted_indices = sorted(
                range(len(combined_population)), 
                key=lambda i: combined_fitness[i], 
                reverse=True
            )

            new_population = [combined_population[i] for i in sorted_indices[:self.population_size]]
            population = new_population
            if gen % 10 == 0:
                print(f"\nGeneration {gen+1}/{self.generations}, Best Fitness: {best_fitness:.4f}")
        return best_individual
    
if __name__ == "__main__":
    test_img_path = r'D:\codePJ\RESEARCH\Flow-Based-Attack-To-VLM\src\images\lionsea.jpg'
    img = cv2.imread(test_img_path)
    gt_test = "A sea dog is seeing the sea"
    adv_test = "A fox is jumping"
    f_fit = Fitness(org_img=img, 
                    model=CLIP(), 
                    gt_text=gt_test)
    popop = POPOP(
           population_size=10,
            f_fit=f_fit,
            cross_rate=0.6,
            model=CLIP(),
            org_img=img,
            mutation_rate=0.2,
            generations=10,
            gt_text=gt_test,
            adv_text=adv_test
    )
    best_individual = popop.run()
    popop.save_log('popop.txt')
    print(best_individual)
