# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
import time

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    sol_nodes = []
    sol_actions = []

    start = problem.getStartState()
    if problem.isGoalState(start):
        return sol_actions

    successors = problem.getSuccessors(start)
    frontier = successors # the frontier holds the (node,action,cost) that still need to be expanded
    frontier_nodes = [] # frontier of (x,y) nodes
    for i in range(len(successors)):
        frontier_nodes.append(successors[i][0]) 

    explored = [start] #nodes explored
    if len(successors) > 1:
        branchpos = [0] #the positions where the graph branches
    else:
        branchpos = []
        
    while frontier != []:
        (node,action,cost) = frontier.pop()
        frontier_nodes.pop()
        sol_nodes.append(node)
        sol_actions.append(action)
        explored.append(node)
        
        if problem.isGoalState(node):
            return sol_actions
        
        successors = problem.getSuccessors(node)
        
        numsuc=0 # how many successor nodes that we haven't already seen are there?
        for suc in successors:
            (new_node,new_action,new_cost) = suc
            #add any node that is not in the frontier nor in the explored set to frontier
            if not (new_node in frontier_nodes or new_node in explored):
                frontier.append(suc)
                frontier_nodes.append(new_node)
                numsuc+=1
                
        if numsuc > 1: 
            for i in range(numsuc-1):
                branchpos.append(len(sol_nodes)) #add the position in sol_nodes of the branch as many times as numsuc 

        if numsuc == 0: #if there are no successors, remove from the solution nodes all nodes after the branching
            br=branchpos.pop()
            sol_nodes=sol_nodes[:br]
            sol_actions=sol_actions[:br]
            
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    sol_actions = []

    start = problem.getStartState()
    if problem.isGoalState(start):
        return sol_actions

    parent = dict() #parent dictionary parent(node) = parent_node, action from parent to child

    successors = problem.getSuccessors(start)
    
    frontier = successors # the frontier holds the (node,action,cost) that still need to be expanded
    frontier_nodes = [] # frontier of (x,y) nodes

    for i in range(len(successors)):
        frontier_nodes.append(successors[i][0])
        parent[successors[i][0]] = (start,successors[i][1])

    explored = [start] #nodes explored

    while frontier != []:
        (node,action,cost) = frontier.pop(0)
        frontier_nodes.pop(0)
        
        explored.append(node)

        if problem.isGoalState(node):
            this_node = node
            while this_node != start:
                sol_actions.append(parent[this_node][1])
                this_node = parent[this_node][0]
            sol_actions.reverse()
            return sol_actions
        
        successors = problem.getSuccessors(node)

        #add any node that is not in the frontier nor in the explored set to frontier
        for suc in successors:
            (new_node,new_action,new_cost) = suc
            if not (new_node in frontier_nodes or new_node in explored):
                parent[new_node] = (node, new_action)
                frontier.append(suc)
                frontier_nodes.append(new_node)
                
    util.raiseNotDefined()

def uniformCostSearch(problem):
	explored = []
	frontier = util.PriorityQueue()
	frontier_nodes = []
	parent = dict()
	node_cost = dict()
	ignore_list = []
	node = problem.getStartState()
	start = node
	if problem.isGoalState(node):
		return []
	frontier.push((node, 88, 0), 0)
	node_cost[node] = 0
	frontier_nodes.append(node)
	while not frontier.isEmpty():
		node = frontier.pop()
		if (node[0], node[2]) in ignore_list:
			continue
		pos = frontier_nodes.index(node[0])
		frontier_nodes.pop(pos)
		if problem.isGoalState(node[0]):
			solution = []
			while node[0] != start:
				solution.append(parent[node[0]][1])
				node = parent[node[0]][0]
			solution.reverse()
			return solution
		explored.append(node[0])
		for suc in problem.getSuccessors(node[0]):
			if suc[0] in explored:
				continue
			temp_cost = suc[2] + node[2]
			if not suc[0] in frontier_nodes:
				frontier.push((suc[0], suc[1], suc[2] + node[2]), suc[2] + node[2] + 0)
				node_cost[suc[0]] = suc[2] + node[2]
				frontier_nodes.append(suc[0])
				parent[suc[0]] = (node, suc[1])
			elif temp_cost < node_cost[suc[0]]:
				ignore_list.append((suc[0], node_cost[suc[0]]))
				frontier.push((suc[0], suc[1], suc[2] + node[2]), suc[2] + node[2] + 0)
				node_cost[suc[0]] = suc[2] + node[2]
				frontier_nodes.append(suc[0])
				parent[suc[0]] = (node, suc[1])
			else:
				continue
				
	print "Failure"
	return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
	explored = []
	frontier = util.PriorityQueue()
	frontier_nodes = []
	parent = dict()
	node_cost = dict()
	ignore_list = []
	node = problem.getStartState()
	start = node
	if problem.isGoalState(node):
		return []
	frontier.push((node, 88, 0), 0)
	node_cost[node] = 0
	frontier_nodes.append(node)
	while not frontier.isEmpty():
		node = frontier.pop()
		if (node[0], node[2]) in ignore_list:
			continue
		pos = frontier_nodes.index(node[0])
		frontier_nodes.pop(pos)
		if problem.isGoalState(node[0]):
			solution = []
			while node[0] != start:
				solution.append(parent[node[0]][1])
				node = parent[node[0]][0]
			solution.reverse()
			return solution
		explored.append(node[0])
		for suc in problem.getSuccessors(node[0]):
			if suc[0] in explored:
				continue
			temp_cost = suc[2] + node[2]
			if not suc[0] in frontier_nodes:
				frontier.push((suc[0], suc[1], suc[2] + node[2]), suc[2] + node[2] + heuristic(suc[0], problem))
				node_cost[suc[0]] = suc[2] + node[2]
				frontier_nodes.append(suc[0])
				parent[suc[0]] = (node, suc[1])
			elif temp_cost < node_cost[suc[0]]:
				ignore_list.append((suc[0], node_cost[suc[0]]))
				frontier.push((suc[0], suc[1], suc[2] + node[2]), suc[2] + node[2] + heuristic(suc[0], problem))
				node_cost[suc[0]] = suc[2] + node[2]
				frontier_nodes.append(suc[0])
				parent[suc[0]] = (node, suc[1])
			else:
				continue
				
	print "Failure"
	return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
