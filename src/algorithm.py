from typing import List, Tuple
import random
from individual import TextIndividual
from model import CLIP
from fitness import Fitness
from settings import W_ADV, W_PSNR, MIN_ANGLE, MAX_ANGLE, MARGIN
import numpy as np
from PIL import Image
from typing import Tuple
from tqdm import tqdm
import cv2

class GABase:
    def __init__(self, 
                 population_size: int,
                 f_fit: Fitness,
                 model: CLIP,
                 org_img: np.ndarray,
                 mutation_rate: float = 0.2,
                 cross_rate: float = 0.5,
                 generations: int = 100,
                 gt_text: str = '',
                 adv_text: str = ''):
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
    
    def create_box(self):
        img_shape = self.org_img.shape
        margin = MARGIN
        min_width = 0.25 * img_shape[1]
        min_height = 0.1 * img_shape[0]
        box_size = (min(random.randint(int(min_width), img_shape[1] - margin), img_shape[1]), # width
                min(random.randint(int(min_height), img_shape[0] - margin), img_shape[0])) # height
        return box_size
    def create_location(self, box_size):
        img_shape = self.org_img.shape
        location = (random.randint(0, img_shape[1] - box_size[0]), 
                random.randint(0, img_shape[0] - box_size[1]))
        return location
    def create_angle(self):
        return random.randint(MIN_ANGLE, MAX_ANGLE)
    def create_blended_fac(self):
        return random.uniform(0.5, 1.0)
    def create_individual(self, text_content):
        box_size = self.create_box()
        location = self.create_location(box_size)
        angle = self.create_angle()
        blended = self.create_blended_fac()
        return TextIndividual(
            content=text_content,
            location=location,
            box_size=box_size,
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
        l2_psnr, adv = self.f_fit.ADV(individual)
        l2 = l2_psnr[0]
        psnr = l2_psnr[1]
        return W_PSNR * psnr + W_ADV * adv
            

    def mutate(self, individual: TextIndividual) -> TextIndividual:
        if random.random() < self.mutation_rate:
            # mutate angle
            new_angle = self.create_angle()
            individual.angle = new_angle
        if random.random() < self.mutation_rate:
            # mutate box size# width
            new_box_size = self.create_box()
            individual.box_size = new_box_size
        if random.random() < self.mutation_rate:
            # mutate location
            new_loc = self.create_location(individual.box_size)
            individual.location = new_loc
        if random.random() < self.mutation_rate:
            # mutate blend factor
            individual.blend_factor = self.create_blended_fac()
        return individual

    def crossover(self, parent1: TextIndividual, parent2: TextIndividual) -> Tuple[TextIndividual, TextIndividual]:
        # Color crossover
        # if random.random() < 0.5:
        #     # Swap colors
        #     child1_color = parent2.color
        #     child2_color = parent1.color
        # else:
        #     # Mix colors
        #     child1_color = tuple((c1 + c2) // 2 for c1, c2 in zip(parent1.color, parent2.color))
        #     child2_color = tuple((c1 + c2) // 2 for c1, c2 in zip(parent2.color, parent1.color))
            
        # Location crossover
        if random.random() < self.cross_rate:
            # Swap locations
            child1_location = parent2.location
            child2_location = parent1.location
        else:
            # Average locations
            child1_location = tuple((l1 + l2) // 2 for l1, l2 in zip(parent1.location, parent2.location))
            child2_location = tuple((l1 + l2) // 2 for l1, l2 in zip(parent2.location, parent1.location))
        
        # Box size crossover
        if random.random() < self.cross_rate:
            # Swap box sizes
            child1_font_size = parent2.box_size
            child2_font_size = parent1.box_size
        else:
            # Average box sizes
            child1_font_size = tuple((s1 + s2) // 2 for s1, s2 in zip(parent1.box_size, parent2.box_size))
            child2_font_size = tuple((s1 + s2) // 2 for s1, s2 in zip(parent2.box_size, parent1.box_size))
        
        # Angle crossover
        if random.random() < self.cross_rate:
            # Swap angles
            child1_angle = parent2.angle
            child2_angle = parent1.angle
        else:
            # Average angles
            child1_angle = (parent1.angle + parent2.angle) // 2
            child2_angle = (parent1.angle + parent2.angle) // 2

        # Blend factor crossover
        if random.random() < self.cross_rate:
            # Swap blend factors
            child1_blend_factor = parent2.blend_factor
            child2_blend_factor = parent1.blend_factor
        else:
            # Average blend factors
            child1_blend_factor = (parent1.blend_factor + parent2.blend_factor) / 2
            child2_blend_factor = (parent1.blend_factor + parent2.blend_factor) / 2

        child1 = TextIndividual(
            content=parent1.content,
            location=child1_location,
            box_size=child1_font_size,
            angle=child1_angle,
            blend_factor=child1_blend_factor

        )
        
        child2 = TextIndividual(
            content=parent2.content,
            location=child2_location,
            box_size=child2_font_size,
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
        last_best = self.logger[-1] if self.logger else None
        if last_best:
            if last_best['fitness'] < individual_dict['fitness']:
                if save_img:
                    individual_dict['adv'].save(f'{individual_dict["id"]}.png')
                self.logger.append(individual_dict)
        else:
            if save_img:
                individual_dict['adv'].save(f'{individual_dict["id"]}.png')
            self.logger.append(individual_dict)
        return
    def save_log(self, path):
        with open(path, 'w') as f:
            for log in self.logger:
                f.write(f"{log}\n")
        return

class POPOP(GABase):
    def __init__(self, 
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
        
        best_individual = None
        best_fitness = float('-inf')

        for gen in tqdm(range(self.generations), desc="Running generations"):
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            fitness_scores = [score.cpu().item() for score in fitness_scores]
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx]
                self.logging({
                    'id': gen,
                    'fitness': best_fitness,
                    'adv': best_individual.add_text_to_image(self.org_img)[0]
                }, save_img=True)
            parents = self.select_parents(population, fitness_scores)

            offspring = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[(i+1) % len(parents)]

                c1, c2 = self.crossover(p1, p2)

                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                offspring.append(c1)
                offspring.append(c2)

            offspring_fitness_scores = [self.calculate_fitness(ind) for ind in offspring]

            combined_population = population + offspring
            combined_fitness = fitness_scores + offspring_fitness_scores

            sorted_indices = sorted(
                range(len(combined_population)), 
                key=lambda i: combined_fitness[i], 
                reverse=True
            )

            new_population = [combined_population[i] for i in sorted_indices[:self.population_size]]
            population = new_population
            print(f"\nGeneration {gen+1}/{self.generations}, Best Fitness: {best_fitness:.4f}")
        return best_individual
    
if __name__ == "__main__":
    test_img_path = r'D:\codePJ\RESEARCH\Flow-Based-Attack-To-VLM\src\images\lionsea.jpg'
    img = cv2.imread(test_img_path)
    gt_test = "A sea dog is seeing the sea"
    adv_test = "A quick brown fox jumps over the lazy dog"
    f_fit = Fitness(org_img=img, 
                    model=CLIP(), 
                    gt_text=gt_test)
    popop = POPOP(
           population_size=10,
            f_fit=f_fit,
            cross_rate=0.5,
            model=CLIP(),
            org_img=img,
            mutation_rate=0.2,
            generations=2,
            gt_text=gt_test,
            adv_text=adv_test
    )
    best_individual = popop.run()
    popop.save_log('popop.txt')
    print(best_individual)
