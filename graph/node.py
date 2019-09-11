from sotw.agents.agent import Agent

import numpy as np

class Node(Agent):

    identity = None

    def __init__(self, belief, position : int = None, agents : int = None, states : int = None):

        super().__init__(belief)

    def set_identity(self, identity):
        self.identity = identity

    def get_identity(self, identity):
        return self.identity

    def move():
        pass