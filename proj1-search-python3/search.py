# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def getMoveList(game_states):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    move_list = []
    for state in game_states:
        if state[1] == s:
            move_list.append(s)
        if state[1] == w:
            move_list.append(w)
        if state[1] == e:
            move_list.append(e)
        if state[1] == n:
            move_list.append(n)
    return move_list


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    visited = {}
    stack = util.Stack()
    path = util.Stack()
    result = dfs_rec(problem, problem.getStartState(), stack, path, visited)
    return result

def dfs_rec(problem, state, stack, path, visited):
    stack.push(state)
    if problem.isGoalState(state):
        return path.list
    visited[state] = True
    for successors in problem.getSuccessors(state):
        if successors[0] not in visited:
            path.push(successors[1])
            result = dfs_rec(problem, successors[0], stack, path, visited)
            if result != False:
                return path.list
                # return stack
    path.pop()
    stack.pop()
    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    queue = util.Queue()
    queue.push( (problem.getStartState(), [], 0) )
    # while queue has fringe nodes
    while not queue.isEmpty():
        # pop off queue
        state = queue.pop()
        # if goal, target reached
        if problem.isGoalState(state[0]):
            # return actions path, as opposed to the state atm
            return state[1]
        # if state not visited yet, visit and explore child nodes
        if state[0] not in visited:
            visited[state[0]] = True
            for successors in problem.getSuccessors(state[0]):
                # push new state onto queue: new_state, action path, and cumulative cost
                action_list = state[1].copy()
                action_list.append(successors[1])
                queue.push( (successors[0], action_list, state[2] + successors[2]) )
    return False


# haha -- copy and paste BFS but change to PriorityQueue to queue least cost first
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    queue = util.PriorityQueue()
    queue.push( (problem.getStartState(), [], 0), 0)
    # while queue has fringe nodes
    while not queue.isEmpty():
        # pop off queue
        state = queue.pop()
        # if goal, target reached
        if problem.isGoalState(state[0]):
            # return actions path, as opposed to the state atm
            return state[1]
        # if state not visited yet, visit and explore child nodes
        if state[0] not in visited:
            visited[state[0]] = True
            for successors in problem.getSuccessors(state[0]):
                # push new state onto queue: new_state, action path, and cumulative cost
                action_list = state[1].copy()
                action_list.append(successors[1])
                queue.push( (successors[0], action_list, state[2] + successors[2]), state[2] + successors[2])
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited = {}
    queue = util.PriorityQueue()
    queue.push( (problem.getStartState(), [], 0), 0)
    # while queue has fringe nodes
    while not queue.isEmpty():
        # pop off queue
        state = queue.pop()
        # if goal, target reached
        if problem.isGoalState(state[0]):
            # return actions path, as opposed to the state atm
            return state[1]
        # if state not visited yet, visit and explore child nodes
        if state[0] not in visited:
            visited[state[0]] = True
            for successors in problem.getSuccessors(state[0]):
                # push new state onto queue: new_state, action path, and cumulative cost
                action_list = state[1].copy()
                action_list.append(successors[1])
                queue.push( (successors[0], action_list, state[2] + successors[2]), state[2] + successors[2] + heuristic(successors[0], problem))
    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
