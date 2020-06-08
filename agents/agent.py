import numpy as np

class Agent:
    """ Thee-valued agent. """

    identity        = 0

    def __init__(self, belief):

        self.belief = belief
        self.evidence = 0
        self.interactions = 0
        self.since_change = 0
        self.identity = Agent.identity
        Agent.identity += 1


    def steady_state(self, threshold):
        """ Check if agent has reached a steady state. """

        return True if self.since_change >= threshold else False


    @staticmethod
    def consensus(belief1, belief2):
        """
        A renormalised sum of the two belief matrices
        """

        # Combine the belief matrices and then clip them to be in
        # the set of possible values: {-1,0,1}
        new_belief = np.clip(np.add(belief1, belief2), -1, 1)

        return new_belief


    def evidential_updating(self, true_state, noise_value, random_instance):
        """
        Update the agent's belief based on the evidence they received.
        Increment the evidence counter.
        """

        evidence = self.random_evidence(
            true_state,
            noise_value,
            random_instance
        )

        new_belief = self.consensus(self.belief, evidence)

        # Track the number of iterations.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = new_belief
        self.evidence += 1


    def update_belief(self, new_belief):
        """
        Update the agent's belief based on a combination of two beliefs:
        their previous belief and another agent's belief.
        Increment the interaction counter.
        """

        # Track the number of iterations.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = new_belief
        self.interactions += 1


    @staticmethod
    def ignorant_belief(states):
        """ Returns a belief of total ignorance/complete uncertainty. """

        return np.full(states, 0, int)


    def random_evidence(self, true_state, noise_value, random_instance):
        """
        Generate a random piece of evidence from the set of states about
        which the agent is uncertain.
        """

        evidence = np.full(len(self.belief), 0, int)
        unknowns = np.argwhere(self.belief == 0)

        if len(unknowns) > 0:
            choice = random_instance.choice(unknowns)

            evidence[choice] = true_state[choice] if random_instance.random() > noise_value else true_state[choice] * -1

        return evidence


class VoterAgent(Agent):
    """
    An agent adopting the stochastic voter model.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2, random_instance):
        """
        Stochastically select a truth value for each state under contention.
        """

        # Combine the belief matrices by flipping a coin for any states on
        # which the two beliefs disagree, and adopting the rest.
        new_belief = np.array([
            belief1[i] if belief1[i] & belief2[i] else random_instance.randint(0,2)
            for i in range(len(belief1))
        ])

        return new_belief


    def evidential_updating(self, true_state, noise_value, random_instance):
        """
        Update the agent's belief based on the evidence they received.
        Increment the evidence counter.
        """

        evidence = self.random_evidence(
            true_state,
            noise_value,
            random_instance
        )

        new_belief = self.belief
        new_belief[evidence[0]] = evidence[1]

        # Track the number of iterations.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = belief
        self.evidence += 1


    @staticmethod
    def ignorant_belief(states, random_instance):
        """ Returns a belief of total ignorance/complete uncertainty. """

        return np.random.randint(2, size=states)


    def random_evidence(self, true_state, noise_value, random_instance):
        """
        Generate a random piece of evidence from the set of states about
        which the agent is uncertain.
        """

        choice = random_instance.randint(len(true_state))

        evidence = (
            choice,
            true_state[choice] if random_instance.random() > noise_value
            else 1 - true_state[choice]
        )

        print(evidence)

        return evidence


class ProbabilisticAgent(Agent):
    """
    An agent adopting a probabilistic model of belief representations
    and consensus formation.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2):
        """
        Bayesian updating.
        """

        # Combine the belief matrices by flipping a coin for any states on
        # which the two beliefs disagree, and adopting the rest.
        new_belief = np.array([
            (belief1[i] * belief2[i]) /
            (belief1[i] * belief2[i] + (1.0 - belief1[i]) * (1.0 - belief2[i]))
            for i in range(len(belief1))
        ])

        return new_belief


    def evidential_updating(self, true_state, noise_value, random_instance):
        """
        Update the agent's belief based on the evidence they received.
        Increment the evidence counter.
        """

        evidence = self.random_evidence(
            true_state,
            noise_value,
            random_instance
        )

        print("Evidence:", evidence)
        print("Current belief:", self.belief)

        new_belief = self.consensus(self.belief, evidence)

        print("New belief:", new_belief)

        # Track the number of iterations.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = new_belief
        self.evidence += 1


    @staticmethod
    def ignorant_belief(states):
        """ Returns a belief of total ignorance/complete uncertainty. """

        return np.full(states, 0.5, float)


    def random_evidence(self, true_state, noise_value, random_instance):
        """
        Generate a random piece of evidence from the set of states about
        which the agent is uncertain.
        """

        evidence = np.full(len(self.belief), 0.5, float)
        unknowns = np.argwhere((self.belief != 0.) & (self.belief != 1.))
        print(unknowns)
        if len(unknowns) > 0:
            choice = random_instance.choice(unknowns)
            print(choice)

            # choice = random_instance.randrange(len(true_state))

            evidence[choice] = true_state[choice] if random_instance.random() > noise_value else 1.0 - true_state[choice]

        return evidence