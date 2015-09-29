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

from game import Agent

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
    #newFood in an instant, newFood[0]-newFood[4] is list len10, newFood[0][0]is bool
    #print "newPos: " is a tuple
    #print "newGhostStates: type list, len 1, list content is object (2.0, 4.0), East " 
    #print "newFood show where is food in the scope of whole panel"

    score = 0
    i = -1
    distance = 0
    for row in newFood:
      i += 1
      j = -1
      for colume in row:
        j += 1
        distance = manhattanDistance((i, j), newPos)
        if (distance == 0 and colume):
          score += 50
        if colume:
          score += 5/distance

    
    ghostPositions =  successorGameState.getGhostPositions()

    k = -1
    for ghostPosition in ghostPositions:
      k += 1;
      if manhattanDistance(ghostPosition, newPos) < 2:
        score -= 70

    return successorGameState.getScore() + score


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

    #return value and action
    def miniMax(miniMaxState, depth, agent):
      numAgent = miniMaxState.getNumAgents()
      newAgent = 0
      newDepth = 0
      valueMax = float("-inf")
      valueMin = float("inf")
      movePac = Directions.STOP
      moveGhost = Directions.STOP

      if depth == self.depth:
        return self.evaluationFunction(miniMaxState), None


      if agent == numAgent - 1:
        newAgent = 0
        newDepth = depth + 1

      else:
        newAgent = agent + 1
        newDepth = depth


      if agent == 0:
        actions = miniMaxState.getLegalActions(agent)
        if actions == []:
          return self.evaluationFunction(miniMaxState), None
        for action in actions:
          if action is Directions.STOP:
            continue
          state = miniMaxState.generateSuccessor(agent, action)
          ret_value, ret_action = miniMax(state, newDepth, newAgent)
          if ret_value > valueMax:
            valueMax = ret_value
            movePac = action

      if agent > 0:
        actions = miniMaxState.getLegalActions(agent)
        if actions == []:
          return self.evaluationFunction(miniMaxState), None
        for action in actions:
          state = miniMaxState.generateSuccessor(agent, action)
          ret_value, ret_action = miniMax(state, newDepth, newAgent)
          if ret_value < valueMin:
            valueMin = ret_value
            moveGhost = action

      if agent == 0:
        return valueMax, movePac
      else:
        return valueMin, moveGhost

    ret =  miniMax(gameState, 0, 0)[1]
    return ret


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def alphabeta(miniMaxState, depth, agent, alpha, beta):
      numAgent = miniMaxState.getNumAgents()
      newAgent = 0
      newDepth = 0
      valueMax = float("-inf")
      valueMin = float("inf")
      movePac = Directions.STOP
      moveGhost = Directions.STOP

      if depth == self.depth:
        return self.evaluationFunction(miniMaxState), None


      if agent == numAgent - 1:
        newAgent = 0
        newDepth = depth + 1

      else:
        newAgent = agent + 1
        newDepth = depth


      if agent == 0:
        actions = miniMaxState.getLegalActions(agent)
        if actions == []:
          return self.evaluationFunction(miniMaxState), None
        for action in actions:
          if action is Directions.STOP:
            continue
          state = miniMaxState.generateSuccessor(agent, action)
          ret_value, ret_action = alphabeta(state, newDepth, newAgent, alpha, beta)
          if ret_value > valueMax:
            valueMax = ret_value
            movePac = action

          if alpha < valueMax:
            alpha = valueMax

          if beta <= alpha:
            break;

      if agent > 0:
        actions = miniMaxState.getLegalActions(agent)
        if actions == []:
          return self.evaluationFunction(miniMaxState), None
        for action in actions:
          state = miniMaxState.generateSuccessor(agent, action)
          ret_value, ret_action = alphabeta(state, newDepth, newAgent, alpha, beta)
          if ret_value < valueMin:
            valueMin = ret_value
            moveGhost = action

          if beta > valueMin:
            beta = valueMin

          if beta <= alpha:
            break;

      if agent == 0:
        return valueMax, movePac
      else:
        return valueMin, moveGhost

    ret =  alphabeta(gameState, 0, 0, float("-inf"), float("inf"))[1]
    return ret


    

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    def expectimax(expectState, depth, agent):
      numAgent = expectState.getNumAgents()
      newAgent = 0
      newDepth = 0
      valueMax = float("-inf")
      valueMin = float("inf")
      alpha = 0
      movePac = Directions.STOP
      moveGhost = Directions.STOP

      if depth == self.depth:
        return self.evaluationFunction(expectState), None


      if agent == numAgent - 1:
        newAgent = 0
        newDepth = depth + 1

      else:
        newAgent = agent + 1
        newDepth = depth


      if agent == 0:
        actions = expectState.getLegalActions(agent)
        if actions == []:
          return self.evaluationFunction(expectState), None
        for action in actions:
          if action is Directions.STOP:
            continue
          state = expectState.generateSuccessor(agent, action)
          ret_value, ret_action = expectimax(state, newDepth, newAgent)
          if ret_value > valueMax:
            valueMax = ret_value
            movePac = action

      if agent > 0:
        actions = expectState.getLegalActions(agent)
        if actions == []:
          return self.evaluationFunction(expectState), None
        actions_len = len(actions)
        for action in actions:
          state = expectState.generateSuccessor(agent, action)
          ret_value, ret_action = expectimax(state, newDepth, newAgent)
          alpha = alpha + ret_value/actions_len
          if alpha < valueMin:
            valueMin = alpha
            moveGhost = action

      if agent == 0:
        return valueMax, movePac
      else:
        return valueMin, moveGhost

    ret =  expectimax(gameState, 0, 0)[1]
    return ret



    


def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  successorGameState = currentGameState
  newPos = successorGameState.getPacmanPosition()
  newFood = successorGameState.getFood()
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
  score = 0
  i = -1
  distance = 0
  for row in newFood:
    i += 1
    j = -1
    for colume in row:
      j += 1
      distance = manhattanDistance((i, j), newPos)
      if (distance == 0 and colume):
        score += 50
      if colume:
        score += 5/distance

  
  ghostPositions =  successorGameState.getGhostPositions()
  k = -1
  for ghostPosition in ghostPositions:
    k += 1;
    if manhattanDistance(ghostPosition, newPos) < 2:
      score -= 70

  return successorGameState.getScore() + score

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
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = 0
    i = -1
    distance = 0
    for row in newFood:
      i += 1
      j = -1
      for colume in row:
        j += 1
        distance = manhattanDistance((i, j), newPos)
        if (distance == 0 and colume):
          score += 50
        if colume:
          score += 5/distance

    
    ghostPositions =  successorGameState.getGhostPositions()

    k = -1
    for ghostPosition in ghostPositions:
      k += 1;
      if manhattanDistance(ghostPosition, newPos) < 2:
        score -= 70

    return successorGameState.getScore() + score
    

