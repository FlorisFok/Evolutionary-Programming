import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'evoman')
from evoman.environment import Environment
from demo_controller import player_controller, enemy_controller
from random import sample
from tensorflow.keras.models import model_from_json, load_model


ENV = Environment(experiment_name="test",
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemy_controller=enemy_controller(),
                  level=2,
                  speed="fastest",
                  contacthurt='player',
                  logs='on')
TRUE_EVAL = .05


class Individual:
    dom_u = 1
    dom_l = -1
    n_hidden = 10
    n_vars = (ENV.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5  # multilayer with 50 neurons

    def __init__(self):
        self.age = 0
        self.weights = list()
        self.fitness = None
        self.enemy_life = None
        self.player_life = None
        self.time = None
        self.children = 0
        self.parents = None

    def set_weights(self, weights=None):
        if weights is None:
            self.weights = np.random.uniform(self.dom_l, self.dom_u, self.n_vars)
        else:
            self.weights = weights

    def evaluate(self):
        if np.random.random() > TRUE_EVAL:
            self.fitness = MODEL.predict([[self.weights]])[0][0] * 100
        else:
            f, p, e, t = ENV.play(pcont=self.weights)
            self.fitness = f
            self.player_life = p
            self.enemy_life = e
            self.time = t

    def check_limits(self):
        new_weights = list()
        for weight in self.weights:
            if weight > self.dom_u:
                new_weights.append(self.dom_u)
            elif weight < self.dom_l:
                new_weights.append(self.dom_l)
            else:
                new_weights.append(weight)

        self.weights = np.asarray(new_weights)

    def birthday(self):
        self.age += 1

    def mutate(self, mutation_rate):
        for i in range(0, len(self.weights)):
            # if np.random.random() <= mutation_rate ** 2:
            #     self.weights[i] = np.random.normal(0, 1)
            if np.random.random() <= mutation_rate:
                self.weights[i] = self.weights[i] * np.random.normal(0, 1.27)
            # if np.random.random() <= mutation_rate:
            #     self.weights[i] = self.weights[i] + np.random.normal(0, .1)
        # if np.random.random() <= mutation_rate ** 3:
        #     np.random.shuffle(self.weights)
        self.check_limits()


class Population:
    def __init__(self, size=5):
        self.individuals = list()
        self.size = size
        self.generation = 1
        self.mutation_rate = 0.3
        self.mean_age = None
        self.mean_children = None
        self.mean_fit = None
        self.max_fit = None
        self.worst_fit = None
        self.mean_fit_history = list()
        self.max_fit_history = list()
        self.worst_fit_history = list()

    def append(self, individual):
        self.individuals.append(individual)
        self.update_stats()

    def extend(self, population):
        self.individuals.extend(population)
        self.update_stats()

    def kill(self, individual):
        self.individuals.remove(individual)
        self.update_stats()

    def update_stats(self):
        population_fit = [i.fitness for i in self.individuals]
        self.mean_fit = np.mean(population_fit)
        self.max_fit = np.max(population_fit)
        self.worst_fit = np.min(population_fit)
        self.mean_age = np.mean([i.age for i in self.individuals])
        self.mean_children = np.mean([i.children for i in self.individuals])

    def display_population(self):
        i = 1
        for individual in self.individuals:
            if individual.parents is not None:
                parent1, parent2 = individual.parents
                mean_parents = (parent1.fitness + parent2.fitness) / 2
            else:
                mean_parents = 0
            print(f'{i}: fitness = {round(individual.fitness, 4)}, age = {individual.age}, '
                  f'children = {individual.children}, parent_fit = {round(mean_parents, 2)}')
            i += 1

        print(f'Mean fitness: {round(self.mean_fit, 4)}, Mean age: {round(self.mean_age, 2)}, ',
              f'Mean children = {round(self.mean_children, 2)} \n')

    def initialize(self):
        for i in range(self.size):
            individual = Individual()
            individual.set_weights()
            individual.evaluate()
            individual.birthday()
            self.individuals.append(individual)

        self.update_stats()
        self.display_population()

    def select_parents(self, n, type_='random'):
        if type_ == 'random':
            return sample(self.individuals, n)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            ranks = [sorted(pop_fitness).index(ind) for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            return np.random.choice(self.individuals, size=n, replace=False, p=probabilities)

    def trim(self, type_='fit'):
        if type_ == 'random':
            self.individuals = sample(self.individuals, self.size)
        elif type_ == 'fit':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            probabilities = [(fit / sum(pop_fitness)) for fit in pop_fitness]
            self.individuals = list(np.random.choice(self.individuals, size=self.size, replace=False, p=probabilities))
        elif type_ == 'rank':
            pop_fitness = [(individual.fitness + 7) for individual in self.individuals]
            ranks = [(sorted(pop_fitness).index(ind) + 0.1) for ind in pop_fitness]
            probabilities = [rank / sum(ranks) for rank in ranks]
            self.individuals = list(np.random.choice(self.individuals, size=self.size, replace=False, p=probabilities))

        self.update_stats()

    def sex(self, type_='mean', selection_type='fit'):

        parent1, parent2 = self.select_parents(2, type_=selection_type)

        cross_prop = np.random.random()

        if type_ == 'mean':
            children = [np.array(parent1.weights) * cross_prop + np.array(parent2.weights) * (1 - cross_prop)]

        elif type_ == 'recombine':
            split_loc = int(len(parent1.weights) * cross_prop)
            child1 = np.append(parent1.weights[:split_loc], parent2.weights[split_loc:])
            child2 = np.append(parent2.weights[:split_loc], parent1.weights[split_loc:])
            children = [child1, child2]

        else:  # type == 'uniform'
            child = list(np.zeros(len(parent1.weights)))
            for i in range(len(child)):
                if np.random.random() <= .5:
                    child[i] = parent1.weights[i]
                else:
                    child[i] = parent2.weights[i]
            children = [child]

        for child_weights in children:
            child = Individual()
            child.set_weights(child_weights)
            child.mutate(mutation_rate=self.mutation_rate)
            child.evaluate()
            child.parents = (parent1, parent2)
            self.individuals.append(child)

            parent1.children += 1
            parent2.children += 1

    def mutation_rate_change(self, type_='exp'):
        if type_ == 'linear':
            self.mutation_rate -= 0.01
        elif type_ == 'exp':
            self.mutation_rate *= 0.99
        elif type_ == 'log':
            self.mutation_rate = np.log(self.mutation_rate)

    def next_generation(self):
        self.generation += 1
        for individual in self.individuals:
            individual.birthday()
        self.mean_fit_history.append(self.mean_fit)
        self.max_fit_history.append(self.max_fit)
        self.worst_fit_history.append(self.worst_fit)

    def plot_generations(self):
        plt.figure(figsize=(12, 12))
        plt.plot(self.max_fit_history, label="best")
        plt.plot(self.mean_fit_history, label="avg")
        plt.plot(self.worst_fit_history, label='worst')
        plt.ylim((-10, 100))
        plt.legend()
        plt.title("First run score")
        plt.show()


def main(size=5, generations=5, children_per_gen=5):
    population = Population(size)
    population.initialize()

    for i in range(generations):
        print('Generation:', population.generation)

        for j in range(children_per_gen):
            population.sex(type_='recombine', selection_type='rank')

        population.trim(type_='rank')
        population.display_population()

        population.mutation_rate_change()
        population.next_generation()

    population.plot_generations()


def get_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")

    return model


MODEL = get_model()
main(size=100, generations=50, children_per_gen=50)
