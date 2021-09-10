import os
import sys
import setproctitle

from fire import Fire

sys.path.append(os.getcwd())
from net.train import train
from IPython import embed

if __name__ == '__main__':
    program_name = 'zrq train ' + os.getcwd().split('/')[-1]
    setproctitle.setproctitle(program_name)
    train
