# calculate the elo rating for different agents.
# Let the players randomly match up with each other and adjust the ratings accourding to the formula
from dominion import *


def calculate_elo_vs_BigMoney(p, ngame = 50):
  """
  Let BigMoney() have elo of 1500. 
  Caluculate elo of some bot
  """
  ngame = ngame
  pb = BigMoney() 
  wins1, fs1 = compare_bots([p,pb],num_games=ngame,order=1)
  wins2, fs2 = compare_bots([pb,p],num_games=ngame,order=1)
  winrate = (wins1[p] + wins2[p])/(2*ngame)
  if winrate == 0:
    return 0
  elif winrate == 1:
    return np.inf
  return -np.log10(1/winrate - 1)*400 


def elo_to_winrate(diff):
  return 1/(1+10**(-diff/400))


def calculate_elo_mixed_bot(plist):
  """
  plist is a list of different bots to compare
  """
  # calculate the winrate of bot vs. bot
  ngame = 100
  winrate = {}
  for i in range(len(plist)-1):
    for j in range(i+1, len(plist)):
      p1 = plist[i]
      p2 = plist[j]
      wins1, fs1 = compare_bots([p1, p2],num_games=ngame,order=1)
      wins2, fs2 = compare_bots([p2, p1],num_games=ngame,order=1)
      winrate[(i,j)] = (wins1[p1] + wins2[p1])/(2*ngame)

  # iterate to get the elo's
  K = 10
  elolist = [1500 for _ in plist]
  num_iter = 100000
  for _ in range(num_iter):
    for i in range(len(plist)-1):
      for j in range(i+1, len(plist)):
        # expected winrate
        wi = elo_to_winrate(elolist[i] - elolist[j])
        wj = 1-wi
        # update elo 
        elolist[i] += K*(winrate[(i,j)] - wi)
        elolist[j] += K*(1-winrate[(i,j)] - wj)

  return elolist


    



