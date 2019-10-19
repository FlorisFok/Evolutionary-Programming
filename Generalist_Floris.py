import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json, load_model
import sys
from scipy import stats
import datetime
import time
import pickle
import sys
import random
import os
sys.path.insert(0, 'evoman')

from environment import Environment
from demo_controller import player_controller, enemy_controller

dom_u = 1
dom_l = -1
MU = 0.03
enemy = [1,2,3,4,6,7,8]
# enemy = [2,4,6]
use_model = False

class Environment2(Environment):
    def cons_multi(self, values):
        return values.sum()/len(self.enemies)

    def fitness_single(self):
        if self.get_enemylife() == 0:
            bonus = 100
        else:
            bonus = 0
        f = 0.6*(100 - self.get_enemylife()) + 0.4*self.get_playerlife() - 10*(self.get_time()/1500)

        return f + bonus

def get_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    return model

# Start environment and first population
def start_env():
    # initializes simulation for coevolution evolution mode.
    env = Environment2(experiment_name="test",
                   enemies=enemy,
                   multiplemode="yes",
                   playermode="ai",
                   player_controller=player_controller(),
                   enemy_controller=enemy_controller(),
                   level=2,
                   speed="fastest")

    # If model learning is enabled
    if use_model:
        MODEL = get_model()

    # System settings
    n_hidden = 10
    n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 50 neurons
    env.update_parameter('contacthurt','player')

    return env, n_vars



class Unit():

    def __init__(self, env, weights):
        self.env = env
        self.weights = weights
        self.victorious = False
        self.fitness = self.score(weights)

    def score(self, weights):
        f,p,e,t = self.env.play(pcont=weights)
        print(f,p,e,t)
        if e == 0:
            self.victorious = True
        return f

    def limits(self, x):
        if x > 1:
            return 1
        elif x < -1:
            return -1
        else:
            return x

    def mutate(self):
        mean = 0
        sigma = 0.5 # HIER NOG NIET ECHT EEN REDEN VOOR.
        for i in range(len(self.weights)):
            if np.random.random() < MU:
                self.weights[i] += random.gauss(mean, sigma)
                self.weights[i] = self.limits(self.weights[i])


class game():
    def __init__(self, env, n_vars, npop, gens):
        self.env = env
        self.n_vars = n_vars
        self.npop = npop
        self.gens = gens
        self.cur_pop = self.random_initialize()
        self.size_tour = 4

    def random_initialize(self):
        pop = np.random.uniform(1, -1, (self.npop, self.n_vars))
        return [Unit(self.env, i) for i in pop]

    def not_r_initialize(self, pop):
        self.cur_pop = [Unit(self.env, i) for i in pop]

    def sort_f(self, x):
        return x.fitness

    def selection(self):
        self.cur_pop.sort(key=self.sort_f, reverse=True)
        self.cur_pop = self.cur_pop[:self.npop]

    def tournament_selection(self):
        random.shuffle(self.cur_pop)

        for i in range(0, len(self.cur_pop), self.size_tour):
            tournament = self.cur_pop[i:i + self.size_tour]
            tournament.sort(key=self.sort_f, reverse=True)
            parent1 = tournament[0]
            parent2 = tournament[random.randint(1,2)]
            self.sex(parent1, parent2)



    def sex(self, parent1, parent2):
        cross_prop = random.random()
        offspring1 = Unit(self.env, np.array(parent1.weights)*cross_prop +
                            np.array(parent2.weights)*(1-cross_prop))

        offspring2 = Unit(self.env, np.array(parent1.weights)*(1-cross_prop) +
                            np.array(parent2.weights)*(cross_prop))

        offspring1.mutate()
        offspring2.mutate()

        self.cur_pop.append(offspring1)
        self.cur_pop.append(offspring2)

    def stats(self):
        return [i.fitness for i in self.cur_pop]

    def best(self):
        self.cur_pop.sort(key=self.sort_f, reverse=True)
        return self.cur_pop[0]

def load_previous():
    x = pickle.load(open("datax1evo.p", "rb"))
    y = pickle.load(open("datay1evo.p", "rb"))
    d = list(zip(x,y))
    def sortit(x):
        return x[1]
    d.sort(key=sortit, reverse=True)
    x, y = zip(*d)

def main():
    MU = 0.03
    gens = 3
    npop = 8

    env, n_vars = start_env()
    run = game(env, n_vars, npop, gens)
    # run.not_r_initialize(x[:npop])

    plot_a = []
    plot_min = []
    plot_max = []

    for gen in range(gens):
        print("GEN", gen)
        run.tournament_selection()
        run.selection()

        best = run.best()
        best.score(best.weights)
        with open("test.txt", "w") as file:
            file.write("best: " + str(best.fitness) + '\n \n \n' + str(best.weights))

        scores = [i.fitness for i in run.cur_pop]
        plot_a.append(sum(scores)/len(scores))
        plot_min.append(max(scores))
        plot_max.append(min(scores))

    plt.plot(plot_a, label='average')
    plt.plot(plot_min, label='min')
    plt.plot(plot_max, label='max')
    plt.legend()
    plt.xlabel("gens")
    plt.ylabel("fitness")


if __name__ == '__main__':
    main()
