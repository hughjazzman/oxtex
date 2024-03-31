import numpy as np
from itertools import product

MAX_ACTIONS = 3

class Node:
    def __init__(self, graphs: list[GraphS], parent=None, vertex_cut_list: list[int]=None, 
                 cliffCount: int=None, scalar: complex=None, depth:int=None, max_actions:int=None):
        self.graphs = graphs  # The current state of the ZX-diagram
        self.parent = parent  # Parent node
        self.vertex_cut_list = vertex_cut_list  # Vertices cut to reach this node
        self.children = []  # List of child nodes
        self.score = 0.0  # Number of terms
        self.visits = 0  # Number of visits during simulations
        self.actions_done = set()
        self.max_actions = max_actions or MAX_ACTIONS
        self.scalar = scalar or complex(0)
        self.cliffCount = cliffCount or 0
        self.depth = depth or 0
        self.untried_actions = None
        self.best_action = None
        self.all_actions = None

    def __len__(self):
        return len(self.graphs)

    def is_fully_expanded(self):
        # Assuming a function that returns all possible cuts
        if self.is_terminal(): return True
        untried_actions = self.get_untried_actions()
        if not untried_actions: 
            return len(self.children) >= 1
        return len(self.children) >= self.max_actions
    
    def is_terminal(self):
        return not self.graphs

    def best_child(self, c_param=1.4):
        # Using UCB
        choices_weights = [
            (child.score / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]
    
    def get_all_actions(self):
        if self.all_actions is not None: return self.all_actions
        vList = []
        bList = []
        for g in self.graphs:
            g = g.copy()
            
            vBest,tier,vweights_max,vweights,vtiers, possV = compWeightsM(g,False)
            
            orderedV = sorted([(w, v) for v, w in enumerate(vweights_max) if w > 0], reverse=True)
            possV = [v for w, v in orderedV]
            vList.append(tuple(possV[:self.max_vertices]))
            bList.append(vBest)
        
        self.best_action = tuple(bList)
        self.all_actions = set(product(*vList))
        self.untried_actions = self.all_actions.copy()
        return self.all_actions
    
    def get_untried_actions(self):
        if self.untried_actions is not None: return self.untried_actions
        return self.get_all_actions()
    
    def get_best_action(self):
        if self.best_action is not None: return self.best_action
        self.get_all_actions()
        return self.best_action
  