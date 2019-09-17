import numpy as np
import matplotlib.pyplot as plt
import sys
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

# Algorithm settings
NPOP = 20
GENS = 10
mutation_rate = 0.4
NCHILDS = 2


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
    p = 0.99

    dp = (p - 0.5)/final_len
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
        # Crossover by floar
        cross_prop = np.random.random()
        offspring = np.array(parent1[0])*cross_prop+np.array(parent2[0])*(1-cross_prop)
        # Crossover by bits
        #     split_i = int(cross_prop*len(parent1))
        #     offspring = np.array(list(parent1[0])[:split_i] + list(parent2[0][split_i:]))
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

    pop_p = np.array([all_pop[i][0] for i in range(NPOP)])
    fit_pop_p = np.array([all_pop[i][1] for i in range(NPOP)])

    old_pop = list(zip(pop_p, fit_pop_p))
    np.random.shuffle(old_pop)

    return old_pop, []

def mutation_rate_change(mu):
    '''
    Defines the change of the mutation rate
    Input: mu [float]
    Ouput: mu [float]
    '''
    return mu - 0.01

def plot_scores(best_scores, avg_scores):
    plt.figure(figsize=(12,12))
    plt.plot(best_scores, label="best")
    plt.plot(avg_scores, label="avg")
    plt.ylim((0,100))
    plt.legend()
    plt.title("First run score")
    plt.show()

cur_gen = initialisatie(NPOP)

best_scores = []
avg_scores = []

for gen in range(GENS):
    print(f"GEN {gen}")

    parent_gen, survival_gen = parent_selection(cur_gen)

    scores = []
    new_pop = []
    more = 5

    for num in range(0, len(parent_gen), 2):
        parent1 = parent_gen[num]
        parent2 = parent_gen[num+1]

        childs = sex(parent1, parent2, scores)
        childs = [mutate(i, mutation_rate) for i in childs]
        new_pop += childs

    mutation_rate = mutation_rate_change(mutation_rate)

    fit_new_pop = evaluate(np.array(new_pop))
    new_gen = list(zip(new_pop, fit_new_pop))

    cur_gen = new_gen + parent_gen + survival_gen

    scores = [i[1] for i in cur_gen]
    scores.sort(reverse=True)
    print(scores)
    best_scores.append(max(scores))
    avg_scores.append(sum(scores)/len(scores))

plot_scores(best_scores, avg_scores)
