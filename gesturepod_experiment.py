from __future__ import print_function
import sys
import subprocess
import os
import torch
import numpy as np

try:
  import wget
  print("[0m[1;37;44m"+"wget already installed"+"[0m")
except:
  print("[0m[1;37;41m"+"installing wget"+"[0m")
  subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                         'wget'])

import myUtils
myUtils.directory_create(myUtils.GesturePod)
myUtils.download_file(myUtils.GesturePod)
myUtils.extract_file(myUtils.GesturePod)

