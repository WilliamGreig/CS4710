# multiAgents.py
# --------------
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

        # first, calculate how much food on board -- the less the better
        tiles = 0
        amt_food = 0
        for food_row in newFood:
            for food in food_row:
                if food:
                    amt_food = amt_food + 1
                tiles = tiles + 1

        # distance to food
        food_list = newFood.asList()
        near_food = tiles*tiles
        for food_c in food_list:
            dist = (abs(newPos[0] - food_c[0])**2 + abs(newPos[1] - food_c[1])**2)**(1/2)
            if dist < near_food:
                near_food = dist


        # calculate Manhattan distance of all ghosts to pacman -- the larger the number, the better
        total_dist = []
        for g in newGhostStates:
            g_pos = g.getPosition()
            dist = (abs(newPos[0] - g_pos[0])**2 + abs(newPos[1] - g_pos[1])**2)**(1/2)
            total_dist.append(dist)


        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore() - amt_food/tiles - 3/( min(total_dist) + 1) + 2/(near_food)
        return score

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action = self.miniMax(gameState)
        return action

    def miniMax(self, state):
        # get the max value action from the minimum successor
        pacman = 0
        actions = state.getLegalActions(pacman)
        v = float('-inf')
        best_act = None
        depth = 0
        next_agent = (pacman + 1) % state.getNumAgents()
        for a in actions:
            score = self.min_value(state.generateSuccessor(pacman, a), next_agent, depth)
            if score > v:
                v = score
                best_act = a
        return best_act

    def max_value(self, state, agentIndex, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(agentIndex)
        next_agent = (agentIndex + 1) % state.getNumAgents()
        v = float('-inf')
        for a in actions:
            v = max(v, self.min_value(state.generateSuccessor(agentIndex, a), next_agent, depth))
        return v


    def min_value(self, state, agentIndex, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(agentIndex)
        v = float('inf')
        next_agent = (agentIndex + 1) % state.getNumAgents()
        for a in actions:
            # if the next agent is not pacman, don't increment depth and get minimum value (adversary)
            if next_agent != 0:
                v = min(v, self.min_value(state.generateSuccessor(agentIndex, a), next_agent, depth))
            else:
                # otherwise, it is pacman and use max value
                v = min(v, self.max_value(state.generateSuccessor(agentIndex, a), next_agent, depth + 1))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # okay alpha_beta is basically minimax but you stop looking at branches/trees if they are too small/large depending on what has already been found for the above nodes

        action = self.alpha_beta(gameState)
        return action


    def alpha_beta(self, state):
        # similar to minimax but now include alpha beta
        pacman = 0
        actions = state.getLegalActions(pacman)
        best_act = None
        depth = 0
        next_agent = (pacman + 1) % state.getNumAgents()
        alpha = float('-inf')
        beta = float('inf')
        for a in actions:
            v = self.min_value(state.generateSuccessor(pacman, a), next_agent, depth, alpha, beta)
            if v > alpha:
                alpha = v
                best_act = a
        return best_act

    def max_value(self, state, agentIndex, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(agentIndex)
        next_agent = (agentIndex + 1) % state.getNumAgents()
        v = float('-inf')
        for a in actions:
            v = max(v, self.min_value(state.generateSuccessor(agentIndex, a), next_agent, depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v


    def min_value(self, state, agentIndex, depth, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(agentIndex)
        v = float('inf')
        next_agent = (agentIndex + 1) % state.getNumAgents()
        for a in actions:
            # if the next agent is not pacman, don't increment depth and get minimum value (adversary)
            if next_agent != 0:
                v = min(v, self.min_value(state.generateSuccessor(agentIndex, a), next_agent, depth, alpha, beta))
            else:
                # otherwise, it is pacman and use max value
                v = min(v, self.max_value(state.generateSuccessor(agentIndex, a), next_agent, depth + 1, alpha, beta))
            # i'm not sure why v <= alpha doesn't work...
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

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
        action = self.expectimax(gameState)
        return action


    def expectimax(self, state):
        # get the max value action from the minimum successor
        pacman = 0
        actions = state.getLegalActions(pacman)
        v = float('-inf')
        best_act = None
        depth = 0
        next_agent = (pacman + 1) % state.getNumAgents()
        for a in actions:
            score = self.min_value(state.generateSuccessor(pacman, a), next_agent, depth)
            if score > v:
                v = score
                best_act = a
        return best_act

    def max_value(self, state, agentIndex, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(agentIndex)
        next_agent = (agentIndex + 1) % state.getNumAgents()
        v = float('-inf')
        for a in actions:
            v = max(v, self.min_value(state.generateSuccessor(agentIndex, a), next_agent, depth))
        return v


    def min_value(self, state, agentIndex, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        actions = state.getLegalActions(agentIndex)
        v = 0
        next_agent = (agentIndex + 1) % state.getNumAgents()
        # the probability is uniform -- so all actions are the same (ie) -- 1/num. of actions
        probability = 1/len(actions)
        for a in actions:
            # if the next agent is not pacman, don't increment depth and get minimum value (adversary)
            if next_agent != 0:
                v = v + self.min_value(state.generateSuccessor(agentIndex, a), next_agent, depth) * probability
            else:
                # otherwise, it is pacman and use max value
                v = v + self.max_value(state.generateSuccessor(agentIndex, a), next_agent, depth + 1) * probability
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My mega-pacman, who I will call Jerry, will be scored using a few features (actually, this is the same evaluation function that I used above but it works, so okay!):
        + current score -- this is a plus for Jerry because the higher the score, the better he is doing (also, we want to make him move faster and higher score means less time wasted)
        + near_food -- we are adding the inverse of manhattan distance between Jerry and the nearest food; the idea is that the closer he is to food, the better Jerry is performing (as he can eat it!)
        - number of remaining food -- we subtract the amount of food still remaining on the board as an incentive to go and eat more food; I found that dividing it by the number of tiles (total squares on the grid) was most effective in getting a better score
        - ghost distance -- we subtract the inverse of the manhattan distance between Jerry and the nearest ghost; if ghosts are far, this is good for Jerry. if ghosts are near, he needs to be careful and therefore his score is lower
    """
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # first, calculate how much food on board -- the less the better
    tiles = 0
    amt_food = 0
    for food_row in newFood:
        for food in food_row:
            if food:
                amt_food = amt_food + 1
            tiles = tiles + 1

    # distance to food
    food_list = newFood.asList()
    near_food = tiles*tiles
    for food_c in food_list:
        dist = (abs(newPos[0] - food_c[0])**2 + abs(newPos[1] - food_c[1])**2)**(1/2)
        if dist < near_food:
            near_food = dist


    # calculate Manhattan distance of all ghosts to pacman -- the larger the number, the better
    total_dist = []
    for g in newGhostStates:
        g_pos = g.getPosition()
        dist = (abs(newPos[0] - g_pos[0])**2 + abs(newPos[1] - g_pos[1])**2)**(1/2)
        total_dist.append(dist)


    "*** YOUR CODE HERE ***"
    # i played around with the factors until i got all the point -- i have no idea about the particular numbers/factors that were best for Jerry's performance
    score = successorGameState.getScore() - amt_food/tiles - 5/( min(total_dist) + 1) + 12/(near_food)
    return score


# Abbreviation
better = betterEvaluationFunction
