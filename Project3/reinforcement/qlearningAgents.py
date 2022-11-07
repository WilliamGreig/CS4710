# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if self.q_values == {}:
            return 0

        # if it is not found, set value equal to 0
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0
            return 0
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        if len(self.getLegalActions(state)) == 0:
            return 0
        max_val = float('-inf')
        for a in self.getLegalActions(state):
            max_val = max(max_val, self.getQValue(state, a))
        return max_val


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if len(self.getLegalActions(state)) == 0:
            return None

        # get maximum q value
        max_val = float('-inf')
        for a in self.getLegalActions(state):
            qVal = self.getQValue(state, a)
            if qVal > max_val:
                max_val = qVal

        # now save the best actions and select one at random
        best_actions = []
        for a in self.getLegalActions(state):
            qVal = self.getQValue(state, a)
            if qVal == max_val:
                best_actions.append(a)
        return random.choice(best_actions)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if len(legalActions) == 0:
            return None

        result = util.flipCoin(self.epsilon)
        if result:
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        discount = self.discount
        alpha = self.alpha
        current_qVal = self.getQValue(state, action)
        max_qVal = self.getValue(nextState)
        # applying formula for q-learning
        new_qVal = current_qVal + alpha * (reward + discount * max_qVal - current_qVal)
        # update value
        self.q_values[(state, action)] = new_qVal

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        # self.weights is dictionary mapping features to weights: to access weights, need features -- use getFeatures(state, action) to get features to get weights
        featureVector = self.featExtractor.getFeatures(state, action)
        qVal = 0
        for i in featureVector:
            qVal = qVal + self.weights[i] * featureVector[i]
        return qVal

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # wi = wi + alpha * difference * fi(s,a)
        # difference = (reward + discount * max(Q (sprime, aprime)) - Q(s,a)

        discount = self.discount
        alpha = self.alpha
        current_qVal = self.getQValue(state, action)
        max_qVal = self.getValue(nextState)
        difference = reward + discount * max_qVal - current_qVal
        # applying given formula
        features = self.featExtractor.getFeatures(state, action)
        for i in features:
            self.weights[i] = self.weights[i] + alpha * difference * features[i]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            print(self.featExtractor)
            print(self.featExtractor.getFeatures(state, "North"))
