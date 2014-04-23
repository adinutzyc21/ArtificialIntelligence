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



# GLOBALS
# state_count = 0

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveRLAgent', second = 'DefensiveRLAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class RLAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    """
    self.red = gameState.isOnRedTeam(self.index)
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)

    # comment this out to forgo maze distance computation and use manhattan distances
    self.distancer.getMazeDistances()

    import __main__
    if '_display' in dir(__main__):
      self.display = __main__._display
  
    inferenceType = util.lookup('ParticleFilter', globals())
    self.inferenceModules = [inferenceType(o,self.index) for o in self.getOpponents(gameState)]
    for inference in self.inferenceModules: inference.initialize(gameState)
    self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    self.firstMove = True

  def updateBelief(self, gameState):
    "Updates beliefs, then chooses an action based on updated beliefs."
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      inf.observeState(gameState)
      self.ghostBeliefs[index] = inf.getBeliefDistribution()
    if self.index == 1:
        self.displayDistributionsOverPositions(self.ghostBeliefs)
    
  
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    
    print self.index
    
    self.updateBelief(gameState)
    
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

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
  

class OffensiveRLAgent(RLAgent):
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

class DefensiveRLAgent(RLAgent):
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




#----------------#
# Ghost tracking #
#----------------#
# observationDistributions = {}
# def getObservationDistribution(noisyDistance):
#     global observationDistributions
#     
#     if noisyDistance not in observationDistributions:
#         distribution = util.Counter()
#         for val in SONAR_NOISE_VALUES:
#             distribution[(max(1, noisyDistance + val))] += 1.0/SONAR_NOISE_RANGE # might not be correct
#         observationDistributions[noisyDistance] = distribution
#     return observationDistributions[noisyDistance]


class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  
  def __init__(self, ghostAgent):
    "Sets the ghost agent for later access"
    self.ghostAgent = ghostAgent
    self.index = ghostAgent.index
    
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.
    
    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    ghostPosition = gameState.getGhostPosition(self.index) # The position you set
    actionDist = self.ghostAgent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist
  
  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index] = game.AgentState(conf, False)
    return gameState
  
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getNoisyGhostDistances()
    if len(distances) >= self.index: # Check for missing observations
      obs = distances[self.index - 1]
      self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]   
    self.initializeUniformly(gameState)
    
  ######################################
  # Methods that need to be overridden #
  ######################################
  
  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass
  
  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass
  
  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass
    
  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass



class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """
  
  ############################################
  # Useful methods for all inference modules #
  ############################################
  #DONE
  def __init__(self, enemyAgentIndex, captureAgentIndex):
    "Sets the ghost agent for later access"
    self.enemyAgentIndex = enemyAgentIndex
    self.captureAgentIndex = captureAgentIndex

  #DONE
  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.
    
    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    #ghostPosition = gameState.getGhostPosition(self.enemy) # The position you set
    ghostPosition = gameState.getAgentPosition(self.enemyAgentIndex)
    #actionDist = self.ghostAgent.getDistribution(gameState)
    actionDist = self.getGhostDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist
  
  #DONE (added)
  def getGhostDistribution( self, state ):
      dist = util.Counter()
      # print "EAI" + str(self.enemyAgentIndex)
      for a in state.getLegalActions( self.enemyAgentIndex ): dist[a] = 1.0
      dist.normalize()
      return dist

  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.enemyAgentIndex] = game.AgentState(conf, False)
    return gameState
    
  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    #distances = gameState.getNoisyGhostDistances()
    distances = gameState.getAgentDistances()
    if len(distances) >= self.captureAgentIndex: # Check for missing observations
      obs = distances[self.enemyAgentIndex]
      self.observe(obs, gameState)
      
  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]   
    self.initializeUniformly(gameState)
    
  ######################################
  # Methods that need to be overridden #
  ######################################
  
  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass
  
  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass
  
  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass
    
  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass


# ------------ #


class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single ghost.
  
  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  """
  
  def initializeUniformly(self, gameState, numParticles=300):
    "Initializes a list of particles."
    
    self.numParticles  = numParticles
    self.particles     = []# util.Counter()
    self.belief        = util.Counter()
    legal              = self.legalPositions
    for part in range(self.numParticles):
        p = random.choice(legal)
        self.particles.append(p)
        self.belief[p] += 1
    self.belief.normalize()
  
  def observe(self, observation, gameState):
    "Update beliefs based on the given distance observation."
    ###emissionModel  = busters.getObservationDistribution(observation)
    # pacmanPosition = gameState.getPacmanPosition()
    captureAgentPosition = gameState.getAgentPosition(self.captureAgentIndex)
    newP           = self.belief.copy()

    
    "*** YOUR CODE HERE ***"
    for part in self.belief.keys():
        g_to_p = util.manhattanDistance(part, captureAgentPosition)
        
        # p(e | x)
        #w = emissionModel[g_to_p]
        w = gameState.getDistanceProb(g_to_p, observation)
        newP[part] = w*(self.particles.count(part))/(self.numParticles)

    # if sum(newP.values()) == 0:
    #     self.initializeUniformly(gameState)
    #     return
    
    self.belief = newP
    
  def elapseTime(self, gameState):
    """
    Update beliefs for a time step elapsing.

    As in the elapseTime method of ExactInference, you should use:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    to obtain the distribution over new positions for the ghost, given
    its previous position (oldPos) as well as Pacman's current
    position.
    """
    "*** YOUR CODE HERE ***"
    newP           = self.belief.copy()
    count = 0
    newParticles = []
    for part in self.particles:
        newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, part))
        # newPosDist = self.getPositionDistribution(gameState, part)
        position   = util.sample(newPosDist)
        newParticles.append(position)


       
    self.particles = newParticles
    
  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    "*** YOUR CODE HERE ***"
    
    self.belief.normalize()
    copy_particles = self.belief.copy()
    return copy_particles

