# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

import time
import pdb

from game import Agent

def foodLeft(dA):
  width = dA.width
  height = dA.height
  n = 0
  for i in range(width):
    for j in range(height):
      if dA[i][j] == True:
        n += 1
  return n

def foodDist(pos, dA):
  width = dA.width
  height = dA.height
  dist = [] 
  for i in range(width):
    for j in range(height):
      if dA[i][j] == True:
        dist.append(util.manhattanDistance(pos,[i,j]))
  if dist!=[]:
    return min(dist)
  return -1

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    
    mFL = foodLeft(newFood)
    mFD = foodDist(newPos, newFood)
    
    for pos in successorGameState.getGhostPositions():
        if abs(pos[0] - newPos[0]) + abs(pos[1] - newPos[1]) < 2:
            return 0
    
    """if len(successorGameState.getCapsules()) < len(currentGameState.getCapsules()):
        return 10000000000
      
    for cap in successorGameState.getCapsules():
        return 1000000000 - abs(newPos[0]-cap[0]) - abs(newPos[1]-cap[1])"""
        
    return 10000000 - mFL*1000 - mFD*10
    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def miniMax (self, agent, curPlayer, gameState, depth):
    myAction = None
    result = [] # [action,score]
    score = None
    
    if gameState.isWin() or gameState.isLose() or depth == self.depth:
      result =[myAction,self.evaluationFunction(gameState)]
      return result
    
    # current player maximizes
    if curPlayer == agent:
      bScore = -999999 # best score is max
      curActs = gameState.getLegalActions(agent) # the actions of the current
      for action in curActs:
        gameSuc=gameState.generateSuccessor(agent,action)
        nextPlayer = (curPlayer+1)%4
        score = self.miniMax(agent, nextPlayer, gameSuc, depth+1)
        score = score[1]
        if score > bScore: # max is playing
          bScore = score
          myAction = action
    
    # other player minimizes
    else:
      bScore = 999999 #best score is min
      curActs = gameState.getLegalActions(agent) # the actions of the current
      for action in curActs:
        gameSuc=gameState.generateSuccessor(agent,action)
        nextPlayer = (curPlayer+1)%4
        score = self.miniMax(agent, nextPlayer, gameSuc, depth+1)
        score = score[1]
        if score < bScore: #min is playing
          bScore = score
          myAction = action
    
    result = [myAction,bScore]
    return result
    
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    result = self.miniMax(0, 0, gameState, 0)
    return result[0]
    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def miniMaxAB (self, agent, curPlayer, gameState, depth, alpha, beta):
    #alpha is minimum score Max will get
    #beta is maximum score Min will get
    myAction = None
    result = [] # [action,score]
    score = None
    
    if gameState.isWin() or gameState.isLose() or depth == self.depth:
      result =[myAction,self.evaluationFunction(gameState)]
      return result
    
    # current player maximizes
    if curPlayer == agent:
      bScore = -999999 # best score is max
      curActs = gameState.getLegalActions(agent) # the actions of the current
      for action in curActs:
        gameSuc=gameState.generateSuccessor(agent,action)
        nextPlayer = (curPlayer+1)%4
        score = self.miniMaxAB(agent, nextPlayer, gameSuc, depth+1, -beta, -max(alpha,bScore))
        score = score[1]
        if score > bScore: # max is playing
          bScore = score
          myAction = action
    
    # other player minimizes
    else:
      bScore = 999999 #best score is min
      curActs = gameState.getLegalActions(agent) # the actions of the current
      for action in curActs:
        gameSuc=gameState.generateSuccessor(agent,action)
        nextPlayer = (curPlayer+1)%4
        score = self.miniMaxAB(agent, nextPlayer, gameSuc, depth+1,-min(beta,bScore), -alpha)
        score = score[1]
        if score < bScore: #min is playing
          bScore = score
          myAction = action
    
    result = [myAction,bScore]
    return result
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    result = self.miniMaxAB(0, 0, gameState, 0, -999999, 999999)
    return result[0]
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def expectiMax (self, agent, curPlayer, gameState, depth):
    myAction = Directions.STOP
    result = [] # [action,score]
    score = None
    
    if gameState.isWin() or gameState.isLose() or depth == self.depth:
      result =[myAction,self.evaluationFunction(gameState)]
      return result
    
    # Max nodes as in minimax search
    if curPlayer == agent:
      bScore = -999999 # best score is max
      curActs = gameState.getLegalActions(agent) # the actions of the current
      for action in curActs:
        gameSuc=gameState.generateSuccessor(agent,action)
        nextPlayer = (curPlayer+1)%4
        score = self.expectiMax(agent, nextPlayer, gameSuc, depth+1)
        score = score[1]
        if score > bScore: # max is playing
          bScore = score
          myAction = action
    
    # Chance nodes are like min nodes
    # Chance nodes take average (expectation) of value of children
    else:
      curActs = gameState.getLegalActions(agent) # the actions of the current
      tScore = 0 #total score
      for action in curActs:
        gameSuc=gameState.generateSuccessor(agent,action)
        nextPlayer = (curPlayer+1)%4
        score = self.expectiMax(agent, nextPlayer, gameSuc, depth+1)
        tScore += score[1]
      bScore = tScore/len(curActs)
      myAction = curActs[0]
    
    result = [myAction,bScore]
    return result
  
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    result = self.expectiMax(0, 0, gameState, 0)
    return result[0]
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    The distance to the food and the number of food bubbles are very important points
    Also if a gost is very close to pacman and the current state is a capsule then you get a reward
    
    Solution Description:
    I wasn't sure what to do here, so I just played some games of the real Pacman, and I just
    wrote an evaluation function to do what I do:  go for the capsules first!  Then, it just
    goes after the remaining food.  As it does, it has a decent chance of accidentally running into
    ghosts.  It also makes sure that it runs away from ghosts if things get bad.
  """
  "*** YOUR CODE HERE ***"
  pos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  foodL = foodLeft(food) # the number of food bubbles left
  minFD = foodDist(pos, food) # the distance to the closest food
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  for states in ghostStates:
    if states.scaredTimer == 0:
      for gpos in currentGameState.getGhostPositions():
        if abs(gpos[0] - pos[0]) + abs(gpos[1] - pos[1]) < 1:
          return 0
  caps = currentGameState.getCapsules()
  import random
  return 100000000 - foodL*100000  - len(caps)*10000000 - minFD*1000 + random.random()*1000
#  pdb.set_trace()
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


