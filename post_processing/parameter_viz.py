import os
from glob import glob
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def get_score(agent_dir):
  path = "{}/parameter_info.txt".format(agent_dir)
  with open(path) as handle:
    dictdump = json.loads(handle.read())
    return dictdump['score']


def get_parameters(agent_dir):
  path = "{}/parameters.txt".format(agent_dir)
  with open(path) as handle:
    return json.loads(handle.read())


parameters = defaultdict(list)

pattern = os.path.join("../saves", "*")
for item in glob(pattern):
  p = get_parameters(item)
  score = get_score(item)

  for key, val in p.items():
    parameters[key].append((val, score))


fig, axs = plt.subplots(2,3, figsize=(15, 6), facecolor='w', edgecolor='k') # TODO: 2x3=6. Work for different amount of parameters
fig.subplots_adjust(hspace = .5, wspace=.01)


for ax, item in zip(axs.ravel(), parameters.items()):
  key, val = item
  ax.set_title(key)
  ax.locator_params(nbins=6)
  ax.scatter(*zip(*val))

plt.show()