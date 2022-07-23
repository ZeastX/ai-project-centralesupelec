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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    actions=[]
    current_nodes=util.Stack()
    explored = {}
    S = problem.getStartState()
    while not (problem.isGoalState(S)):
        if not(S in explored):
            explored[S]=problem.getSuccessors(S)
            explored[S].reverse()
        succ = explored[S]
        stemp = S
        for i in range(len(succ)):
            if not(succ[i][0] in explored):
                stemp = succ[i][0]
                act = succ[i][1]
                break
        if stemp == S:
            S = current_nodes.pop()
            actions.pop(-1)
        else:
            current_nodes.push(item=S)
            S = stemp
            actions.append(act)
    return actions
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actions=[]
    nodes=util.Queue()
    explored = {}
    parent={}
    S = problem.getStartState()
    start = S
    while not (problem.isGoalState(S)):
        if not(S in explored):
            explored[S]=problem.getSuccessors(S)
            explored[S].reverse()
            succ = explored[S]
            for elt in succ :
                if not(elt[0] in explored):  
                    nodes.push(elt[0])
                if not(elt[0] in parent):    
                    parent[elt[0]] = S,elt[1]
        S = nodes.pop()
    while not(S == start):
        actions.insert(0,parent[S][1])
        S = parent[S][0]
    return actions

def uniformCostSearch(problem):
    actions=[]
    nodes=util.PriorityQueue()
    explored = {}
    parent={}
    S = problem.getStartState()
    start = S
    cost = {S:0}
    while not (problem.isGoalState(S)):
        if not(S in explored):
            explored[S]=problem.getSuccessors(S)
            explored[S].reverse()
            succ = explored[S]
            for elt in succ :
                if not(elt[0] in parent):
                    parent[elt[0]] = S,elt[1]
                if not(elt[0] in cost):
                    cost[elt[0]] = cost[S] + elt[2]
                elif cost[elt[0]] >= cost[S] + elt[2]:
                    cost[elt[0]] = cost[S] + elt[2]
                    parent[elt[0]] = S,elt[1]
                if not(elt[0] in explored):  
                    nodes.update(elt[0],cost[elt[0]])
        S = nodes.pop()
    while not(S == start):
        actions.insert(0,parent[S][1])
        S = parent[S][0]
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    actions=[]
    nodes=util.PriorityQueue()
    explored = {}
    parent={}
    S = problem.getStartState()
    start = S
    cost = {S:(0,heuristic(S,problem))}
    while not (problem.isGoalState(S)):
        if not(S in explored):
            explored[S]=problem.getSuccessors(S)
            explored[S].reverse()
            succ = explored[S]
            for elt in succ :
                if not(elt[0] in parent):
                    parent[elt[0]] = S,elt[1]
                if not(elt[0] in cost):
                    cost[elt[0]] = (cost[S][0] + elt[2], heuristic(elt[0],problem))
                elif cost[elt[0]][0]+cost[elt[0]][1] > cost[S][0] + elt[2] + heuristic(elt[0],problem):
                    cost[elt[0]] = (cost[S][0] + elt[2], heuristic(elt[0],problem))
                    parent[elt[0]] = S,elt[1]
                nodes.update(elt[0],cost[elt[0]][0]+cost[elt[0]][1])
        S = nodes.pop()
    while not(S == start):
        actions.insert(0,parent[S][1])
        S = parent[S][0]
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
