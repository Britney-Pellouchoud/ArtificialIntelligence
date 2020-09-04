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
from game import Directions
from game import Actions


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



def treeSearch(problem, fringe):
    node = problem.getStartState()
    fringe.push(node);
    foundpath = False

    paths = {} # child : parent
    closed = []
    while not foundpath:
        if fringe.isEmpty():
            return False
        currnode = fringe.pop()

        if not isinstance(currnode, tuple):
            position = currnode

        else:
            position = currnode[0]

        if (currnode != node) :
            if problem.isGoalState(position):
                return findpath(paths, currnode)

        if position not in closed:
            closed.append(position)
            
            if type(currnode) == tuple and len(currnode) != 3:
                for n in problem.getSuccessors(currnode):
                    fringe.push(n)
                    if n not in paths.values():
                        paths[n] = currnode
            else:
                for n in problem.getSuccessors(currnode[0]):
                    fringe.push(n)
                    if n not in paths.values():
                        paths[n] = currnode

    return paths;



def findpath(dic, leaf):
    pathway = [leaf[1]]
    while leaf in dic:
        if len(dic[leaf]) == 3:
            intner = dic[leaf]
            inner = dic[leaf][1]
            pathway.append(inner)


        leaf = dic[leaf]
    
    return pathway[::-1]
    
    




def depthFirstSearch(problem):
   
    s = util.Stack()
    return treeSearch(problem, s)

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
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    q = util.Queue()
    return treeSearch(problem, q)
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    dic = {} # childnode : parentnode
    closed = set() # list of nodes visited
    initialstate = (problem.getStartState(), None, 0)
    fringe = util.PriorityQueue() # based on cost of previous actions 
    fringe.push(initialstate, 0) 
    while True:
        if fringe.isEmpty():
            return None # no possible path
        parentnode = fringe.pop() 
        position = parentnode[0][0] # (x, y)
        #if not isinstance(parentnode, tuple):
        #    position = parentnode
#
 #       else:
  #          position = parentnode[0]
        if problem.isGoalState(parentnode[0]): 
            path = newfindpath(dic, parentnode)
            return path
        if parentnode[0] not in closed:
            closed.add(parentnode[0])
            
            for child in problem.getSuccessors(parentnode[0]):
                #addtodic = True
                #if parentnode in dic and dic[parentnode] == child:   
                #    addtodic = False 
                #if addtodic:
                
                node = child[0], child[1], child[2] + parentnode[2]
                dic[node] = parentnode
                #lstofaction = getbackdir(dic, child)
                #cost = problem.getCostOfActions(lstofaction)
                fringe.push(node, child[2] + parentnode[2])


            #else:
            #    for child in problem.getSuccessors(parentnode):
            #        if child[0] not in dic.values():
            #            dic[child] = parentnode
            #        
                    
                   # fringe.push(child, 0)

    return dic
    #fringe.push()



def getbackdir(dictnry, node):
    dire = []
    while node in dictnry:
        dire.append(node[1])
        node = dictnry[node]
    return dire[::-1]

def newfindpath(dic, leaf):
    #leaf = (1,1)
    pathway = [leaf[1]]
    
    while leaf[1] != None: 
        
        pathway.append(str(dic[leaf][1]))
        leaf = dic[leaf] 
    pathway = pathway[:-1]

    return pathway[::-1]

    """pathway = [leaf[1]]
    while leaf in dic:
        if len(dic[leaf]) == 3:
            intner = dic[leaf]
            inner = dic[leaf][1]
            pathway.append(inner)


        leaf = dic[leaf]
    
    return pathway[::-1]
    """





    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    dic = {} # childnode : parentnode
    closed = set() # list of nodes visited
    initialstate = (problem.getStartState(), None, 0)
    fringe = util.PriorityQueue() # based on cost of previous actions 
    fringe.push(initialstate, 0) 
    while True:
        if fringe.isEmpty():
            return None 
        parentnode = fringe.pop() 
        position = parentnode[0][0] 
        
        if problem.isGoalState(parentnode[0]): 
            path = newfindpath(dic, parentnode)
            return path
        if parentnode[0] not in closed:
            closed.add(parentnode[0])
            
            for child in problem.getSuccessors(parentnode[0]):
                #frontcost = heuristic(child)
                node = child[0], child[1], child[2] + parentnode[2] 
                dic[node] = parentnode
                frontcost = heuristic(child[0], problem)
                
                fringe.push(node, child[2] + parentnode[2] + frontcost)

    return dic
    








    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch