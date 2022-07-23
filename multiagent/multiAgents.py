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
from itertools import product

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        oldFood = currentGameState.getFood()
        oldFood = oldFood.asList()
        s=0
        ghostPos = []
        for ghost in newGhostStates:
            ghostPos.append(ghost.getPosition())
        newFood = newFood.asList()
        m=10**6
        for elt in oldFood:
            if manhattanDistance(newPos,elt) < m:
                m=manhattanDistance(newPos,elt)
                food = elt
        for i in range(len(ghostPos)):
            if manhattanDistance(food,ghostPos[i]) > manhattanDistance(newPos,food) and manhattanDistance(newPos,ghostPos[i]) > 4 :
                if newScaredTimes[i] == 0:    
                    s+= 15-m
                else:
                    s+= 10 - manhattanDistance(newPos,ghostPos[i]) - m
            elif manhattanDistance(newPos,ghostPos[i]) > 3 :
                s+=10-m
            else:
                if newScaredTimes[i] == 0:
                    s+= 5 + manhattanDistance(newPos,ghostPos[i]) - m/2
                else:
                    s+= 10 - manhattanDistance(newPos,ghostPos[i]) - m
        if len(oldFood)<len(newFood):
            s=s+50
        print(successorGameState.getScore() + s*5)
        if not(successorGameState.isWin()):
            return successorGameState.getScore() + s*5
        else:
            return 10**6

def scoreEvaluationFunction(currentGameState: GameState):
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
    
    
    def getAction(self, gameState: GameState):
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
        def minval(gamestate: GameState,i,index):
            L = gamestate.getLegalActions(index)
            L1=[10**6]
            v=10**6
            n = gamestate.getNumAgents()
            if gamestate.isLose():
                return self.evaluationFunction(gamestate)
            if gamestate.isWin():
                return self.evaluationFunction(gamestate)
            if i == self.depth:
                return self.evaluationFunction(gamestate)
            else:
                for elt in L:
                    succ = gamestate.generateSuccessor(index,elt)
                    if index < n-1:
                        v=min(v,minval(succ,i,index+1))
                    else : 
                        v = min(v,maxval(succ,i+1,0))    
            return v
        def maxval(gamestate: GameState,i,index):
            L = gamestate.getLegalActions(0)
            v=-10**6
            if i == self.depth:
                return self.evaluationFunction(gamestate)
            if gamestate.isWin():
                return self.evaluationFunction(gamestate)
            if gamestate.isLose():
                return self.evaluationFunction(gamestate)
            else:    
                for elt in L:
                    succ = gamestate.generateSuccessor(0,elt)
                    v = max(v,minval(succ,i,1))
            return v
        L = gameState.getLegalActions(0)
        m=-10**6
        a=L[0]
        for elt in L:
            succ = gameState.generateSuccessor(0,elt)
            value = minval(succ,0,1)
            if m<=value:
                m=value
                a=elt
        return a
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        
        def minval(gamestate: GameState,i,index,alpha,beta):
            L = gamestate.getLegalActions(index)
            v=float('+inf')
            n = gamestate.getNumAgents()
            if gamestate.isLose():
                return self.evaluationFunction(gamestate)
            if gamestate.isWin():
                return self.evaluationFunction(gamestate)
            if i == self.depth:
                return self.evaluationFunction(gamestate)
            else:
                for elt in L:
                    succ = gamestate.generateSuccessor(index,elt)
                    beta = min(v,beta)
                    if index < n-1:
                        v=min(v,minval(succ,i,index+1,alpha,beta)) 
                        
                        if v < alpha:
                            return v
                        beta = min(v,beta)
                        
                        
                    else : 
                        v = min(v,maxval(succ,i+1,0,alpha,beta))
                        
                        if v < alpha:
                            return v
                        beta = min(v,beta)
                            
            return v
        def maxval(gamestate: GameState,i,index,alpha,beta):
            L = gamestate.getLegalActions(0)
            v=float('-inf')
            if i == self.depth:
                return self.evaluationFunction(gamestate)
            if gamestate.isWin():
                return self.evaluationFunction(gamestate)
            if gamestate.isLose():
                return self.evaluationFunction(gamestate)
            else:    
                for elt in L:
                    succ = gamestate.generateSuccessor(0,elt)
                    v = max(v,minval(succ,i,1,alpha,beta))
                    alpha = max(v,alpha)
                    if v > beta:
                        return v
                    
            return v
        L = gameState.getLegalActions(0)
        m=float('-inf')
        alpha = float('-inf')
        a=L[0]
        for elt in L:
            succ = gameState.generateSuccessor(0,elt)
            value = minval(succ,0,1,alpha,float('+inf'))
            alpha = max(value,alpha)
            if m<value:
                m=value
                a=elt
        return a

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def minval(gamestate: GameState,i,index):
            L = gamestate.getLegalActions(index)
            v=0
            n = gamestate.getNumAgents()
            if gamestate.isLose():
                return self.evaluationFunction(gamestate)
            if gamestate.isWin():
                return self.evaluationFunction(gamestate)
            if i == self.depth:
                return self.evaluationFunction(gamestate)
            else:
                for elt in L:
                    succ = gamestate.generateSuccessor(index,elt)
                    if index < n-1:
                        v=v+minval(succ,i,index+1)
                    else : 
                        v =v +maxval(succ,i+1,0)  
            return v
        def maxval(gamestate: GameState,i,index):
            L = gamestate.getLegalActions(0)
            v=float('-inf')
            if i == self.depth:
                return self.evaluationFunction(gamestate)
            if gamestate.isWin():
                return self.evaluationFunction(gamestate)
            elif gamestate.isLose():
                return self.evaluationFunction(gamestate)
            else:    
                for elt in L:
                    succ = gamestate.generateSuccessor(0,elt)
                    v = max(v,minval(succ,i,1))
            return v
        L = gameState.getLegalActions(0)
        m=-10**6
        a=L[0]
        for elt in L:
            succ = gameState.generateSuccessor(0,elt)
            value = minval(succ,0,1)
            if m<value:
                m=value
                a=elt
        return a

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentGhost = currentGameState.getGhostStates()
    currentPos = currentGameState.getPacmanPosition()
    s=0
    for ghost in currentGhost:
        s+=manhattanDistance(ghost.getPosition(),currentPos)
    return currentGameState.getScore()+s



    

# Abbreviation
better = betterEvaluationFunction
