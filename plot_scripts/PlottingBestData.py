import os
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
# plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(12,6))

def get_values(data):
    avg0 = [sum([data[n][i] for n in range(len(data))])/len(data)
           for i in range(len(datapart[0]))]
    max0 = [max([data[n][i] for n in range(len(data))])
           for i in range(len(datapart[0]))]
    min0 = [min([data[n][i] for n in range(len(data))])
           for i in range(len(data[0]))]
    return avg0, max0, min0


for nplot, level in enumerate([1,5,7]):
    location = r"C:\Users\FlorisFok\Documents\Github\Evolutionary-Programming\Specialist_results\Enemy_" + str(level)
    log = 'evo_Log'
    files = [i for i in os.listdir(location) if log in i]

    count = 0
    datapart = []
    datafrac = []

    for file in files:
        if 'Log' in file:
            with open(os.path.join(location, file), 'r') as f:
                text = f.read()
                i1 = text.index("'Best Scores'") + 16
                i2 = text[i1:].index("],") + i1
                if "'Cross MODE': 'parts'," in text:
                    strip_text = text[i1:i2].replace("(", "").replace(")", "")
                    datapart.append([float(x) for n, x in enumerate(strip_text.split(',')) if n % 2 == 1])
                else:
                    strip_text = text[i1:i2].replace("(", "").replace(")", "")
                    datafrac.append([float(x) for n, x in enumerate(strip_text.split(',')) if n % 2 == 1])
                count+=1

    plt.subplot(1 ,3 ,nplot + 1)
    if nplot+1 == 1:
        plt.ylabel("fitness", fontsize=14)
    elif nplot+1 == 2:
        plt.title("Best fitness over the generations", fontsize=16)

    avg1, max1, min1 = get_values(datapart)
    avg2, max2, min2 = get_values(datafrac)

    plt.plot(avg1, color='black', label=f'part level:{level}')
    plt.fill_between(range(len(avg1)), min1, max1,
                 color='grey', alpha=0.2)

    plt.plot(avg2, color='red', label=f'fraction level:{level}')
    plt.fill_between(range(len(avg2)), min2, max2,
                 color='red', alpha=0.2)

    plt.legend(loc=4)
    plt.xlabel("generations [n]", fontsize=14)
    plt.ylim((-10,100))

fig.savefig("TriplePlotBest.png")
