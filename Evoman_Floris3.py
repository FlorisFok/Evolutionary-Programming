import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats
import datetime
import time
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller, enemy_controller
# initializes simulation for coevolution evolution mode.
env = Environment(experiment_name="test",
               enemies=[2],
               playermode="ai",
               player_controller=player_controller(),
               enemy_controller=enemy_controller(),
               level=2,
               speed="fastest")

# System settings
n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 50 neurons
dom_u = 1
dom_l = -1
env.update_parameter('contacthurt','player')

# User dependent
file_loc = r"C:\Users\FlorisFok\Documents\Master\Evo Pro\evoman_framework\individual_demo\evoman_log"

# evaluation
def evaluate(x1):
    return np.array(list(map(lambda y: simulation(env, y), x1)))

# runs simulation
def simulation(env,x1):
    f,p,e,t = env.play(pcont=x1)
    return f

# limits
def limits(x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x

def sortfit(x):
    return x[1]

def survival(final_len, final_num):
    '''
    Makes a gradient list ove the new gens'''
    a = []
    i = 0
    p = 0.98

    dp = (p - 0.4)/final_len
    while len(a) < final_len and i < final_num - 1:
        p -= dp
        if p > np.random.random():
            a.append(i)
        i += 1
    return a

def sex(parent1, parent2, scores):
    '''
    Crossover and mutation function
    Input: parent1 [np.array()], parent2 [np.array()]
    Output: list of offsprings [np.array()]
    '''

    offsprings = []

    for i in range(NCHILDS):
        cross_prop = np.random.random()

        if MODE == 'float':
            # Crossover by float
            offspring = np.array(parent1[0])*cross_prop+np.array(parent2[0])*(1-cross_prop)
        elif MODE == 'bit':
            # Crossover by bits
            split_i = int(cross_prop*len(parent1))
            offspring = np.array(list(parent1[0])[:split_i] + list(parent2[0][split_i:]))
        elif MODE == 'uniform':
            # Crossover by bits
            offspring = np.zeros(parent1[0].shape)
            for i, a in enumerate(parent1[0]):
                if np.random.random() < 0.5:
                    offspring[i] = a #parent1
                else:
                    offspring[i] = parent2[0][i]


        offsprings.append(offspring)

    return offsprings

def mutate(individual, mutation_rate):
    '''
    Mutate the individual
    Input: individual [np.array()], mutation rate [float]
    Output: mutated individual [np.array()]
    '''
    for i in range(0,len(individual)):
        if np.random.random() <= mutation_rate:
            individual[i] = np.random.normal(0, 1)

    individual = np.array(list(map(lambda y: limits(y), individual)))
    return individual

def initialisatie(npop):
    '''
    Input: number of individuals
    Output: The population in shape [np,.array shape(npop, 265)]
            The fitness of the population [np.array()]
    '''
    pop_p = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop_p = evaluate(pop_p)
    first_pop = list(zip(pop_p, fit_pop_p))
    return first_pop

def parent_selection(all_pop):
    '''
    Choose who dies, keep npop constant and select the best parents by order
    Input: all_pop [zip(individuals[np.array()], fitnesss[float])]
    Output: parent_pop [zip(individuals[np.array()], fitnesss[float])]
                    (each parent in pairs of two in correct order)
                    , survival_pop [zip(individuals[np.array()], fitnesss[float])]
                    (Doesnt gets a child)
    '''
    all_pop.sort(key=sortfit, reverse=True)
    pop_len = len(all_pop)
    shindlerslist = survival(NPOP, pop_len)
    cur_gen = [all_pop[i] for i in shindlerslist]
    np.random.shuffle(cur_gen)

    lucky_list = [i for i in range(pop_len) if not i in shindlerslist and np.random.random() > 0.75]
    lucky_gen = [all_pop[i] for i in lucky_list]

    return np.array(cur_gen), np.array(lucky_gen)

def mutation_rate_change(mu):
    '''
    Defines the change of the mutation rate
    Input: mu [float]
    Ouput: mu [float]
    '''
    return mu - delta_mu

def plot_scores(best_scores, avg_scores):
    plt.figure(figsize=(12,12))
    plt.plot(np.array(best_scores)[:,1], '.-', label="best", color='green')
    plt.plot(np.array(best_scores)[:,0], '.-', label="worst", color='red')
    plt.plot(avg_scores, linewidth=2, label="avg")
    plt.ylim((-10,100))
    plt.legend()
    plt.title("First run score")
    plt.show()

def save_log(inp, log, result):
    loc = file_loc + str(inp) +".txt"
    with open(loc, "w") as f:
        f.write("LOG:\n" + str(log) + "\nResults: \n" + str(result))



# Algorithm settings
NPOP = 20
GENS = 20
MU = 0.8
delta_mu = MU/GENS
NCHILDS = 2
MODE = 'uniform'

ts = time.time()
cur_gen = initialisatie(NPOP)

best_scores = []
avg_scores = []
var_scores = []
mutation_rate = MU

for gen in range(GENS):
    print(f"GEN {gen}")

    parent_gen, survival_gen = parent_selection(cur_gen)

    scores = []
    new_pop = []
    more = 5
    max_len = len(parent_gen) - 1

    for num in range(0, len(parent_gen), 2):
        if num == max_len:
            continue
        parent1 = parent_gen[num]
        parent2 = parent_gen[num+1]

        childs = sex(parent1, parent2, scores)
        childs = [mutate(i, mutation_rate) for i in childs]
        new_pop += childs

    mutation_rate = mutation_rate_change(mutation_rate)

    fit_new_pop = evaluate(np.array(new_pop))
    new_gen = list(zip(new_pop, fit_new_pop))

    cur_gen = list(new_gen) + list(parent_gen) + list(survival_gen)

    scores = [i[1] for i in cur_gen]
    scores.sort(reverse=True)
    scores = np.array(scores)

    dis = stats.describe(np.array(scores))
    best_scores.append(dis.minmax)
    avg_scores.append(dis.mean)
    var_scores.append(dis.variance)

cur_gen.sort(key=sortfit, reverse=True)
sollution = cur_gen[0][0]
total_time = time.time() - ts

print(f"took: {total_time} to find \n {sollution}")

save_log("test", {
    "NPOP": NPOP,
    "GENS": GENS,
    "MUTA start": MU,
    "MUTA step": delta_mu,
    "NCHILDS": NCHILDS,
    "Cross MODE": MODE,
    "time": datetime.date.today(),
    "total_time": total_time
}, {
    "Best Score": best_scores[-1],
    "Scores": scores,
    "Avg Scores": avg_scores,
    "sollution":sollution,
})
plot_scores(best_scores, avg_scores)
