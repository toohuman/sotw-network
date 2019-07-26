from sotw.agents.agent import Agent

import numpy as np

class Hub(Agent):

    def __init__(self, belief, position : int = None, agents : int = None, states : int = None):

        super()