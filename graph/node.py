from sotw.agents.agent import Agent

import numpy as np

class Node(Agent):

    belief          = None
    evidence        = int
    interactions    = int
    since_change    = int
    region          = None
    region_boundary = None

    def __init__(self, belief, position : int = None, agents : int = None, states : int = None):

        super()