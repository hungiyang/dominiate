# model evaluation 
# load the previous models and compute their winrate against random/Big Money/SmithyBot
# can we compute the elo ratings?
import numpy as np
from dominion import *
import glob
import os

def evaluate_agent_evolution(model_name, ngame=10, version = 'SarsaAgent', pre_witch = 0):
  # find all the model files
  test = glob.glob('model/'+model_name + '*')
  idx = int(np.where([x.isdigit() for x in test[0].split('_')])[0])
  iteration = np.sort(np.unique([int(fn.split('_')[idx]) for fn in test]))
  pb = BigMoney()
  [p, dql] = load_rl_bot('model/{:s}_iteration_002'.format(model_name), version=version,  pre_witch = pre_witch)
  w1 = []
  w2 = []
  rl1= []
  rl2= []
  b1 = []
  b2 = []
  success_iterations = []
  ngame = ngame
  for i in iteration:
    print(i)
    try:  
      p, _ = load_rl_bot('model/{:s}_iteration_{:03d}'.format(model_name,i), dql, version = version,  pre_witch = pre_witch)
      p.epsilon = 0
      wins1, fs1 = compare_bots([p,pb],num_games=ngame,order=1)
      wins2, fs2 = compare_bots([pb,p],num_games=ngame,order=1)
      w1.append(wins1[p])
      w2.append(wins2[p])
      rl1.append(fs1[p])
      rl2.append(fs2[p])
      b1.append(fs1[pb])
      b2.append(fs1[pb])
      success_iterations.append([i])
    except:
      print('loading {:d} failed.'.format(i))
      pass
    
  iteration = np.asarray(success_iterations, dtype=float)
  w1 = np.asarray(w1, dtype=float)
  w2 = np.asarray(w2, dtype=float)
  rl1 = np.asarray(rl1, dtype=float)
  rl2 = np.asarray(rl2, dtype=float)
  b1 = np.asarray(b1, dtype=float)
  b2 = np.asarray(b2, dtype=float)
  if not os.path.exists('evolution'):
    os.makedirs('evolution')
  fn = 'evolution/' + model_name + '_{:d}'.format(ngame)
  with open(fn, 'wb') as f:
    pickle.dump((iteration,w1,w2,rl1,rl2,b1,b2,ngame), f)
  return (iteration,w1,w2,rl1,rl2,b1,b2,ngame)

def load_model_evolution(model_name, ngame = 50):
  fn = 'evolution/' + model_name + '_{:d}'.format(ngame)
  with open(fn, 'rb') as f:
    return pickle.load(f)




if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Model parameters')
  parser.add_argument('-n', '--name')
  parser.add_argument('-v','--version', default = 'SarsaAgent')
  parser.add_argument('--ngame', type=int, default = 10)
  parser.add_argument('--pre_witch', type=bool, default = False)
  args = parser.parse_args()


  evaluate_agent_evolution(args.name, args.ngame, args.version, args.pre_witch)
