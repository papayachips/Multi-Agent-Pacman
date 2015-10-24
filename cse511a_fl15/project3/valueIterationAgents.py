# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    
    "*** YOUR CODE HERE ***"
    '''
    def valueIteration(mdp, discount, depth, iterations, state, action):
      maxValues = float("-inf")
      maxAction = action
      retValues = util.Counter()

      if depth == iterations:
        return retValues, maxAction

      if isTerminal(state):
        newDepth = depth + 1
        state = mdp.getStartState()
        actions = mdp.getPossibleActions(state)        
      else:
        newDepth = depth

      actions = mdp.getPossibleActions(state)
      for action in actions:
        nextStates_probs = mdp.getTransitionStatesAndProbs(state, action)
        rewards = util.Counter()
        probs = util.Counter()
        for nextState_prob in nextStates_probs:
          rewards[nextState_prob[0]] = mdp.getReward(state, action, nextState_prob[0])
          probs[nextState_prob[1]] = nextState_prob[1]
        value = (rewards + (valueIteration(mdp, discount, newDepth, iterations, nextState_prob[0], action).divideAll(1.0/discount))) * probs
        if value > maxValues:
          maxValues = value
          retValues = (valueIteration(mdp, discount, newDepth, iterations, nextState_prob[0], action)
          maxAction = action


    return valueIteration(mdp, discount, 0, iterations, mdp.getStartState(), None)[0]
    '''
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    sum = 0
    actions = self.mdp.getPossibleActions(state)
    for action in actions:
      nextStates_probs = self.mdp.getTransitionStatesAndProbs(state, action)
      for nextState_prob in nextStates_probs:
        print "nextState_prob[0]", self.getValue(nextState_prob[0])
        print self.getValue(nextState_prob[0])
        sum += nextState_prob[1] * (self.mdp.getReward(state, action, nextState_prob[0]) + self.discount * self.getValue(nextState_prob[0]))
        print sum
    return sum

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    
    maxvalue = float("-inf")
    actions = self.mdp.getPossibleActions(state)

    if actions == ():
      return None

    maxaction = actions[0]
    for action in actions:
      nextStates_probs = self.mdp.getTransitionStatesAndProbs(state, action)
      for nextState_prob in nextStates_probs:
        temp = nextState_prob[1] * (self.mdp.getReward(state, action, nextState_prob[0]) + self.discount * self.getValue(nextState_prob[0]))
        if temp > maxvalue:
          maxaction = action
    return maxaction
    

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
