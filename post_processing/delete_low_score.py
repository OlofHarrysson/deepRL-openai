import sys
import os
from glob import glob
import json
import shutil

''' Deleted all the runs and saves with a score lower than the input threshold '''

def get_score(agent_dir):
  path = "{}/parameter_info.txt".format(agent_dir)
  with open(path) as handle:
    dictdump = json.loads(handle.read())
    return dictdump['score']


if not len(sys.argv) == 2:
  print("Input the score threshhold to keep")
  sys.exit(1)

keep_score = int(sys.argv[1])

pattern = os.path.join("../saves", "*")
for item in glob(pattern):
  agent_name = os.path.basename(item)
  score = get_score(item)

  if score < keep_score:
    shutil.rmtree(item)
    print("Deleting {}".format(item))

    pattern = os.path.join("../runs", "{}*".format(agent_name))
    for item in glob(pattern):
      shutil.rmtree(item)
      print("Deleting {}".format(item))