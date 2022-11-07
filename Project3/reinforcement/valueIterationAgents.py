# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        self.values[0] = 0
        for k in range (self.iterations):

            # we need to copy the vector of values so we can implement batch version
            # whereby we use the vector of values from k-1 to update values
            """ From the PDF:
            This means that when a state's value is updated in iteration k based on the values of its successor states, 
            the successor state values used in the value update computation should be those from iteration k-1 
            (even if some of the successor states had already been updated in iteration k).
            """
            v = self.values.copy()
            # for all states in the mdp
            for s in self.mdp.getStates():
                self.values[s] = float('-inf')
                # if there are no actions to take (terminal state), set to value of state to zero
                if self.mdp.isTerminal(s):
                    self.values[s] = 0
                # get all actions to compute value
                for a in self.mdp.getPossibleActions(s):
                    # can't use computeQValueFromValues because self.values is not updated...
                    sum_val = 0
                    for s_prime, probability in self.mdp.getTransitionStatesAndProbs(s, a):
                        reward = self.mdp.getReward(s, a, s_prime)
                        discounted_val = self.discount * v[s_prime]
                        sum_val = sum_val + (reward + discounted_val) * probability
                    self.values[s] = max(self.values[s], sum_val)



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qVal = 0
        # apply formula
        for s_prime, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, s_prime)
            discounted_val = self.discount * self.values[s_prime]
            qVal += (reward + discounted_val) * probability
        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if terminal, return None
        if self.mdp.isTerminal(state):
            return None

        # get maximum value and corresponding action
        max_val = float('-inf')
        action = None
        # for all actions in state
        for a in self.mdp.getPossibleActions(state):
            # compute q value
            qVal = self.computeQValueFromValues(state, a)
            # compare -- if qval better than current max, take max and get that action
            if qVal > max_val:
                max_val = qVal
                action = a
        return action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # get all states
        states = self.mdp.getStates()
        for k in range(self.iterations):

            # state (s) is determined by kth index (but modulo the size of the state space -- otherwise out of bounds)
            s = states[k % len(states)]
            # if not terminal
            if not self.mdp.isTerminal(s):
                # get max value for self.values[s]
                v = float('-inf')
                for a in self.mdp.getPossibleActions(s):
                    v = max(v, self.computeQValueFromValues(s, a))
                self.values[s] = v
            else: # if terminal, set value to 0
                self.values[s] = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # step 1: compute predecessors of all states
        # compute all predecessors
        # predecessors are defined, for a given state s, as all the states that have non-zero probability of reaching s
        predecessors = {}
        queue = util.PriorityQueue()
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for sprime, probability in self.mdp.getTransitionStatesAndProbs(s, a):
                    # all non-zero probabilities
                    if probability != 0:
                        # if sprime already in predecessors, add the state to update the predecessor
                        if sprime in predecessors:
                            predecessors[sprime].add(s)
                        else: # otherwise, if it is not in predecessors, update the predecessor of sprime as s
                            predecessors[sprime] = {s}

        # now, predecessor should be a set of sets; the key should be sprime (predecessors) and the items should be all states that lead to that predecessor (s-prime)

        # for each non-terminal state s, do:
        for s in self.mdp.getStates():
            # not terminal
            if not self.mdp.isTerminal(s):
                v = float('-inf')
                for a in self.mdp.getPossibleActions(s):
                    # find absolute value of the diiference between current value of s in self.values and the highest Q-value across all possible actions from s
                    v = max(v, self.computeQValueFromValues(s, a))
                diff = abs(self.values[s] - v)
                # push s into priortiy queue
                queue.push(s, -1*diff)

        for k in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            # if not terminal, update value
            if not self.mdp.isTerminal(s):
                v = float('-inf')
                for a in self.mdp.getPossibleActions(s):
                    v = max(v, self.computeQValueFromValues(s, a))
                self.values[s] = v

            # for each predecessor p of s:
            for p in predecessors[s]:
                # find absolute value of difference between current value of p in self.values and highest Q-value across all possible actions from p
                pVal = float('-inf')
                for a in self.mdp.getPossibleActions(p):
                    pVal = max(pVal, self.computeQValueFromValues(p, a))
                diff = abs(self.values[p] - pVal)
                # if diff > theta, push p into priority queue with priority -diff
                if diff > self.theta:
                    queue.update(p, -1*diff)

