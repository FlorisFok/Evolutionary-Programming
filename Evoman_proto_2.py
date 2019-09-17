import numpy as np
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
npop = 10
nchild_max = 2
gens = 5
mutation_rate = 0.4
nparents = int(npop/4)


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

def sex(parent1, parent2, mutation_rate):

    cross_prop = np.random.random()
    offspring = np.array(parent1[0])*cross_prop+np.array(parent2[0])*(1-cross_prop)

    # mutation
    for i in range(0,len(offspring)):
        if np.random.random() <= mutation_rate:
            offspring[i] = np.random.normal(0, 1)

    offspring = np.array(list(map(lambda y: limits(y), offspring)))

    return offspring


pop_p = np.random.uniform(dom_l, dom_u, (npop, n_vars))
fit_pop_p = evaluate(pop_p)

best_scores = []
avg_scores = []

for gen in range(gens):
    print(f"GEN {gen}")

    old_pop = list(zip(pop_p, fit_pop_p))
    old_pop.sort(key=sortfit, reverse=True)

    scores = []
    pop_p = []
    more = 5

    for parent_num in range(nparents):
        parent1 = old_pop[parent_num]
        parent2 = old_pop[np.random.randint(int(npop/4))]

        for nchild in range(int(nchild_max*more)):
            child = sex(parent1, parent2, mutation_rate)
            pop_p.append(np.array(child))
        more -= (5/nparents)

    mutation_rate -= 0.01

    fit_pop_p = evaluate(pop_p)
    new_pop = list(zip(pop_p, fit_pop_p))

    all_pop = new_pop + old_pop
    all_pop.sort(key=sortfit)

    pop_p = np.array([all_pop[i][0] for i in range(npop)])
    fit_pop_p = np.array([all_pop[i][1] for i in range(npop)])

    scores += [i[1] for i in all_pop]
    best_scores.append(max(scores))
    avg_scores.append(sum(scores)/len(scores))
