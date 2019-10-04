import os
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,6))
ax = plt.subplot(111)

def make_box(data):
    return

box_data = []

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
                i1 = text.index("'Best Score'") + 15
                i2 = text[i1:].index("),") + i1
                try:
                    if "'Cross MODE': 'parts'," in text:
                        pass
                        datapart.append(float(text[i1:i2].split(',')[1].strip()))
                    else:
                        pass
                        datafrac.append(float(text[i1:i2].split(',')[1].strip()))
                except:
                    print(text[i1:i2])
                count+=1
    box_data.append(datafrac)
    box_data.append(datapart)

ax.boxplot(box_data)
ax.set_xticklabels(["p1", "f1", "p5", "f5", "p7", "f7"])
ax.set_ylabel("Fitness")
ax.set_title("Boxplots of the best score")

fig.savefig("BoxplotBestScore.png")
