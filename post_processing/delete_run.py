import shutil
import sys
from glob import glob
import os

''' Run this program with the hexadigit that the folders in run/ or saves/ starts with it to delete them more easily '''

if not len(sys.argv) == 2:
  print("Input the start of the agent folder's name that's to be deleted from /runs and /saves")
  sys.exit(1)

agent_name = sys.argv[1]
pattern = os.path.join("../saves", "{}*".format(agent_name))
for item in glob(pattern):
  shutil.rmtree(item)
  print("Deleting {}".format(item))


pattern = os.path.join("../runs", "{}*".format(agent_name))
for item in glob(pattern):
  shutil.rmtree(item)
  print("Deleting {}".format(item))
