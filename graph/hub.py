from sotw.agents.agent import Agent

import numpy as np

class Hub(Agent):

    belief          = None
    evidence        = int
    interactions    = int
    since_change    = int
    region          = None
    region_boundary = None

    def __init__(self, belief, position : int = None, agents : int = None, states : int = None):

        super()