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
#enemy = [1,2,3,4,6,7,8]
enemy = [2,5,6]
use_model = False
runmode='train'

# class Environment2(Environment):
#     def cons_multi(self, values):
#         values = values * np.array([1, 0.5, 1, 1, 1, 1, 0.5])
#         return values.sum()/len(self.enemies)
#
#     def fitness_single(self):
#         if self.get_enemylife() == 0:
#             bonus = 50
#         else:
#             bonus = 0
#         f = 0.6*(100 - self.get_enemylife()) + 0.4*self.get_playerlife() - 10*(self.get_time()/1500)
#
#         return f + bonus

def get_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    return model

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def simulation(env,x):
    wins = 0
    f,p,e,t = env.play(pcont=x)

    #file_aux = open('test/results.txt')
    if not e>0:
        wins=1

    print('fitness ' + str(f) + ' player_life ' + str(p) + ' enemy_life ' + str(e) + ' time ' + str(t) + ' wins ' + str(wins))
    # file_aux.write(
    #     '\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    # file_aux.close()

    return f

# Start environment and first population
def start_env():
    # initializes simulation for coevolution evolution mode.
    env = Environment(experiment_name="test",
                   enemies=enemy,
                   multiplemode="yes",
                   playermode="ai",
                   player_controller=player_controller(),
                   enemy_controller=enemy_controller(),
                   level=2,
                   logs="off",
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
        self.fitness, self.p_life, self.e_life = self.score(weights)

    def score(self, weights):
        f,p,e,t = self.env.play(pcont=weights)
        # print(f)
        if e == 0:
            self.victorious = True
        return f,p,e

    def limits(self, x):
        if x > 1:
            return 1
        elif x < -1:
            return -1
        else:
            return x

    def mutate(self, MU):
        mean = 0
        sigma = 0.5 # HIER NOG NIET ECHT EEN REDEN VOOR.
        for i in range(len(self.weights)):
            if np.random.random() < MU:
                self.weights[i] += random.gauss(mean, sigma)
                self.weights[i] = self.limits(self.weights[i])


class game():
    def __init__(self, env, n_vars, npop, gens, MU):
        self.env = env
        self.n_vars = n_vars
        self.npop = npop
        self.gens = gens
        self.mutation_rate = MU
        self.cur_pop = self.random_initialize()
        self.size_tour = 3

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

    def tournament_selection(self, top, bottom):
        random.shuffle(self.cur_pop)

        for i in range(0, len(self.cur_pop), self.size_tour):
            tournament = self.cur_pop[i:i + self.size_tour]
            tournament.sort(key=self.sort_f, reverse=True)
            parent1 = tournament[0]
            parent2 = tournament[1]

            position = 1

            # print("p = ", position)
            #for i in range(position):
            self.sex_whole_arithmetic_xover(parent1, parent2, self.mutation_rate)

    def sex_whole_arithmetic_xover(self, parent1, parent2, MU):
        cross_prop = random.random()
        offspring1 = Unit(self.env, np.array(parent1.weights)*cross_prop +
                            np.array(parent2.weights)*(1-cross_prop))

        offspring2 = Unit(self.env, np.array(parent1.weights)*(1-cross_prop) +
                            np.array(parent2.weights)*(cross_prop))

        offspring1.mutate(MU)
        offspring2.mutate(MU)

        self.cur_pop.append(offspring1)
        self.cur_pop.append(offspring2)

    def sex_blend_xover(self, parent1, parent2, MU):
        weights_child1 = [0]*len(parent1.weights)
        weights_child2 = [0]*len(parent1.weights)

        rand = [np.random.randint(0,1) for i in range(0,len(parent1.weights))]

        for i in range(0,len(parent1.weights)):
            d_i = abs(parent2.weights[i] - parent1.weights[i])

            if rand[i] < 0.5:
                weights_child1[i] = parent1.weights[i] - 0.5*d_i
                weights_child2[i] = parent1.weights[i] + 0.5*d_i
            else:
                weights_child1[i] = parent1.weights[i] + 0.5*d_i
                weights_child2[i] = parent1.weights[i] - 0.5*d_i

        offspring1 = Unit(self.env, np.array(weights_child1))
        offspring2 = Unit(self.env, np.array(weights_child2))

        offspring1.mutate(MU)
        offspring2.mutate(MU)

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

def name_check(name, ext, file_loc):
    i = 1
    while name+str(i)+ext in os.listdir(file_loc):
        i+=1

    return os.path.join(file_loc, name+str(i)+ext)

def main():
    env, n_vars = start_env()

    if runmode == 'test':
        env.update_parameter('multiplemode', 'no')
        for i in range(1,2):
            bsol = np.loadtxt('tourn_6/best/tourn_' + str(i) + '.txt')
            print('\n RUNNING SAVED BEST SOLUTION '+str(i)+'\n')
            for j in [1,2,3,4,5,6,7,8]:
                env.update_parameter('enemies', [j])
                print('ENEMY: ' + str(j))
                for k in range(0,5):
                    evaluate(env, [bsol])

        env.update_parameter('enemies', [1,2,3,4,5,6,7,8])
        env.update_parameter('multiplemode', 'yes')
        env.update_parameter('logs', 'on')

        for k in range(0,5):
            evaluate(env, [bsol])

        sys.exit(0)

    # CHANGE TO YOUR DIRECTORY
    file_loc = r"D:\Documenten\College_2019_2020\Evolutionary_Computing\evoman_framework-master\tourn_3"

    gens = 50
    npop = 50

    env, n_vars = start_env()

    MU = 1/n_vars
    plot_string = f"Mu={MU}_lowerfitness"

    run = game(env, n_vars, npop, gens, MU)
    scores = [i.fitness for i in run.cur_pop]
    energy = [i.p_life for i in run.cur_pop]
    top = max(scores)
    bottom = min(scores)

    mean_energy = [np.array(energy).mean()]
    std_energy = [np.array(energy).std()]
    max_energy = [max(energy)]
    min_energy = [min(energy)]

    plot_a = []
    plot_min = []
    plot_max = []
    plot_mean_energy = []
    plot_std_energy =[]

    for gen in range(gens):
        print("\nGEN", gen)
        run.tournament_selection(top, bottom)
        run.selection()

        best = run.best()
        # best.score(best.weights)

        # with open("test.txt", "w") as file:
        #     file.write("best: " + str(best.fitness) + '\n \n \n' + str(best.weights))
        print("best: ", str(best.fitness))

        scores = [i.fitness for i in run.cur_pop]
        energy = [i.p_life for i in run.cur_pop]
        # print(str(energy))
        top = max(scores)
        bottom = min(scores)
        mean_p =np.array(energy).mean()
        std_p = np.array(energy).std()
        mean_energy.append(mean_p)
        std_energy.append(std_p)
        max_energy.append(max(energy))
        min_energy.append(min(energy))

        plot_a.append(sum(scores)/len(scores))
        plot_min.append(min(scores))
        plot_max.append(max(scores))
        plot_mean_energy.append(mean_p)
        plot_std_energy.append(std_p)

    fig = plt.figure()
    plt.plot(plot_a, label='average')
    plt.plot(plot_min, label='min')
    plt.plot(plot_max, label='max')
    plt.plot(plot_mean_energy,label='mean energy')
    #plt.plot(plot_std_energy, label='mean energy')
    plt.legend()
    plt.title("Fitness and energy over generations")
    plt.xlabel("gens")
    plt.ylabel("fitness and energy")
    fig.savefig(os.path.join(file_loc, name_check("generalist_tourn_size_3_plot", '.png', file_loc)))

    loc_data = os.path.join(file_loc, name_check("data_tourn_size_3", ".txt", file_loc))
    loc_scores = os.path.join(file_loc, name_check("scores_tourn_size_3", ".txt", file_loc))
    loc_best = os.path.join(file_loc, name_check("best_tourn_size_3", ".txt", file_loc))
    loc_energy = os.path.join(file_loc, name_check("energy_tourn_size_3", ".txt", file_loc))

    with open(loc_data, "w") as file:
        file.write(str(plot_a) + "\n\n" + str(plot_max) + "\n\n" + str(plot_min))

    with open(loc_scores, "w") as file:
        file.write(str(scores))

    with open(loc_best, "w") as file:
        file.write("best: " + str(best.fitness) + '\n \n \n' + str(best.weights))

    with open(loc_energy, "w") as file:
        file.write('Mean over generations: ' +str(mean_energy) + '\n\nStd energy over generations: ' +str(std_energy)+ '\n\nMax energy over generations :'+str(max_energy)
                   + '\n\nMin over generations :'+str(min_energy))

if __name__ == '__main__':
    main()
