# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

import pdb

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  estimator = PositionEstimator()
  return [eval(first)(firstIndex,estimator), eval(second)(secondIndex,estimator)]

class PositionEstimator:
    def __init__(self):
        """Create the position estimator, but we don't have enough info to really initialize yet"""
        self.walls = None
        self.width = 0
        self.height = 0
        self.enemies = None
        self.initial_positions = None
        self.beliefs = None
        self.neighbors = None
        self.initialized = False
        
    def initialize(self,walls,enemies,initial_positions):
        """
        call this from registerInitialState(self,gameState)
        walls is a Grid (see game.py) with walls marked by 'true'
        enemies is a sequence with enemy indices - used for initial positions
        enemypos is a sequence with enemy initial positions
        """
        #pdb.set_trace()
        self.initialized = True
        self.walls = walls
        self.width = walls.width
        self.height = walls.height
        self.enemies = enemies
        self.initial_positions = initial_positions
        #un-normalized array of occupancy beliefs--0 everywhere except initial enemy positions
        self.beliefs = [[[0.0 for y in xrange(self.height)] for x in xrange(self.width)] for n in xrange(len(enemies))]
        #we know (from mazeGenerator.py) where each agent starts
        for ie in xrange(len(enemies)):
          x,y = self.initial_positions[enemies[ie]]
          self.beliefs[ie][x][y] = 1.0
        #how many predecessors does each maze position have?
        #ie. for each maze position, how many positions are there that after 1 move, an agent could be in the position
        # = neighbors + self
        #self.neighbors = [[1*(not walls[x][y]) for y in xrange(self.height)] for x in xrange(self.width)]
        self.neighbors = [[4*(not walls[x][y]) for y in xrange(self.height)] for x in xrange(self.width)]
        for x  in xrange(self.width):
          for y in xrange(self.height):
            if not walls[x][y]:
              self.neighbors[x][y] -= walls[x-1][y] + walls[x+1][y] + walls[x][y-1] + walls[x][y+1]
    
    def observe_noisy(self,enemy,mypos,dist):
        """Update with a distance measurement from pos to an enemy"""
        bel = self.beliefs[self.enemies.index(enemy)]
        #B(enemy is at x,y) = B(enemy is at x,y)*P(observed dist | enemy is at x,y)
        #P(dist | enemy at x,y) = 1/13 * [L1(mypos, (x,y)) is within 6 of dist, inclusive]
        for x in xrange(self.width):
            for y in xrange(self.height):
                if not self.walls[x][y]:
                    d = util.manhattanDistance(mypos,(x,y)) - dist
                    bel[x][y] *= 1.0/13 *(d >= -6 and d <= 6)
        
    def observe_exact(self,enemy,enemypos):
        """We exactly observed the enemy"""
        tmp = [[0.0 for y in xrange(self.height)] for x in xrange(self.width)]
        x,y = enemypos
        tmp[x][y] = 1.0
        self.beliefs[self.enemies.index(enemy)] = tmp
    
    def observe_death(self,enemy):
        """We saw an enemy die!! It will return to its initial position"""
        #print 'death: resetting agent to ', self.initial_positions[enemy]
        self.observe_exact(enemy, self.initial_positions[enemy])
        
    def move(self,enemy):
        """update for enemy movements"""
        #we know they can't move into walls
        #otherwise we're using a uniform movement model (move in any direction with equal propability)
        ie = self.enemies.index(enemy)
        bel = self.beliefs[ie]
        tmp = [[0.0 for y in xrange(self.height)] for x in xrange(self.width)]
        #B(enemy is at x,y now) = sum over all i,j {B(enemy was at i,j before)*P(enemy moved from i,j to x,y)}
        for x in xrange(self.width):
            for y in xrange(self.height):
                #enemy can't be in walls, so there's no point in looking at them
                if not self.walls[x][y]:
                    #we only need to look at the neighbors of x,y bcs. we can't move more than that
                    tmp[x][y] = (bel[x-1][y] + bel[x+1][y] + bel[x][y-1] + bel[x][y+1])/self.neighbors[x][y] #+bel[x][y] if we can stay still
        self.beliefs[ie] = tmp
        
    def normalize(self):
        for arr in self.beliefs:
            s = sum(sum(col) for col in arr)
            if s != 0:
              for x in xrange(self.width):
                arr[x] = map(lambda x: x/s, arr[x])
    
    def to_counters(self):
      ctrs = []
      for ie in xrange(len(self.enemies)):
        ctrs.append(util.Counter())
        for x in xrange(self.width):
          for y in xrange(self.height):
            if not self.walls[x][y]:
              ctrs[-1][(x,y)] = self.beliefs[ie][x][y]
      return ctrs
    
    def best_guess(self):
        """The best guess of the positions of the enemies"""
        best_pos = [(0,0)]*len(self.enemies)
        best_bel = [0]*len(self.enemies)
        for x in xrange(self.width):
            for y in xrange(self.height):
                for ie in xrange(len(self.enemies)):
                    if self.beliefs[ie][x][y] > best_bel[ie]:
                        best_pos[ie] = (x,y)
                        best_bel[ie] = self.beliefs[ie][x][y]
        return zip(self.enemies,best_pos)
    
    def to_str(self,e):
      s = ''
      for y in xrange(1,self.height+1):
        s+=' '.join('%1.3f' % self.beliefs[e][x][-y] for x in xrange(self.width)) + '\n'
      return s
        
##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def __init__(self,index,posEstimator):
    CaptureAgent.__init__(self,index)
    self.position_estimator = posEstimator
    
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self,gameState)
    if not self.position_estimator.initialized:
      self.position_estimator.initialize(gameState.getWalls(), self.getOpponents(gameState),
      [gameState.getInitialAgentPosition(i) for i in xrange(gameState.getNumAgents())])
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    #pdb.set_trace()
    agents = range(gameState.getNumAgents())
    enemies = self.getOpponents(gameState)
    positions = [gameState.getAgentPosition(a) for a in agents]
  
    #update the position estimator for whoever moved before us
    prev_agent = (self.index - 1) % 4
    self.position_estimator.move(prev_agent)
    
    #observe our enemies
    distances = gameState.getAgentDistances()
    for e in enemies:
      if positions[e] is not None:
        self.position_estimator.observe_exact(e,positions[e])
      else:
        self.position_estimator.observe_noisy(e,positions[self.index],distances[e])
    
    #self.position_estimator.normalize()
    #print self.position_estimator
    #self.displayDistributionsOverPositions(self.position_estimator.to_counters())
    guess = self.position_estimator.best_guess()
    self.debugDraw([guess[0][1],guess[1][1]],(1,1,1),clear=True)
    
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    #choose action:
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    act = random.choice(bestActions)
    
    #does the action we choose kill an enemy?
    state = gameState.getAgentState(self.index)
    succ = self.getSuccessor(gameState,act)
    next_state = succ.getAgentState(self.index)
    
    next_positions = [succ.getAgentPosition(a) for a in agents]
    #pdb.set_trace()
    for e in enemies:
      #we were close together
      if positions[e] is not None and util.manhattanDistance(positions[self.index],positions[e]) <=2:
        #now we're far apart, if I'm back in my initial position then I died
        if next_positions[e] ==gameState.getInitialAgentPosition(e) and next_positions[self.index] != gameState.getInitialAgentPosition(self.index):
          self.position_estimator.observe_death(e)

    return act

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
