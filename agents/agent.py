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

        # Track the number of iterations that the agent's belief has
        # remained unchanged.
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
            belief1[i] if belief1[i] == belief2[i] else random_instance.randint(0,1)
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

        # print()
        # print("Current belief:", self.belief)
        # print("Evidence:", evidence)

        new_belief = self.belief
        new_belief[evidence[0]] = evidence[1]
        # print("New belief:", new_belief)

        # Track the number of iterations that the agent's belief has
        # remained unchanged.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = new_belief
        self.evidence += 1


    @staticmethod
    def ignorant_belief(states, random_instance):
        """ Returns a belief of total ignorance/complete uncertainty. """

        return np.array([random_instance.randint(0,1) for x in range(states)])


    def random_evidence(self, true_state, noise_value, random_instance):
        """
        Select a random state and provide a piece of evidence subject to
        some noise_value.
        """

        choice = random_instance.randrange(len(true_state))

        evidence = (
            choice,
            true_state[choice] if random_instance.random() > noise_value
            else 1 - true_state[choice]
        )

        return evidence


class StochasticAgent(Agent):
    """
    An agent which randomly chooses between a conservative operator and the standard
    three-valued consensus operator.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2, random_instance, random=True):
        """
        With a 50:50 chance, prefer a conservative (maximally uncertain) belief over a
        typical consensus position.
        """

        if random and random_instance.randint(0,1):
            # Combine the belief matrices by flipping a coin for any states on
            # which the two beliefs disagree, and adopting the rest.
            new_belief = np.array([
                belief1[i] if belief1[i] == belief2[i] else 0
                for i in range(len(belief1))
            ])
        else:
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

        new_belief = self.consensus(self.belief, evidence, random_instance, random=False)

        # Track the number of iterations.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = new_belief
        self.evidence += 1


class ProbabilisticAgent(Agent):
    """
    An agent adopting a probabilistic model of belief representation
    and consensus formation.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2):
        """
        Probabilistic updating using the product operator. This combines two (possibly conflicting)
        probability distributions into a single probability distribution.
        Bayesian updating of each independent variable based on evidence.
        """

        # Combine the belief matrices using the product operator from (Lee et al. 2018)
        new_belief = np.array([
            (belief1[i] * belief2[i]) /
            (belief1[i] * belief2[i] + (1.0 - belief1[i]) * (1.0 - belief2[i]))
            for i in range(len(belief1))
        ])

        invalid_belief = np.isnan(np.sum(new_belief))

        if not invalid_belief:
            return new_belief
        else:
            return None


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
        if evidence is not None:
            new_belief[evidence[0]] = evidence[1]

        # Track the number of iterations that the agent's belief has
        # remained unchanged.
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

        evidence = None
        unknowns = np.argwhere((self.belief != 0.) & (self.belief != 1.))

        if len(unknowns) > 0:

            alpha = 0.1
            choice = random_instance.choice(unknowns)
            if true_state[choice] == 0:
                alpha = (1.0 - alpha)

            if random_instance.random() > noise_value:
                value = ((1.0 - alpha) * self.belief[choice])\
                    /((1.0 - alpha) * self.belief[choice] + alpha * (1.0 - self.belief[choice]))
            else:
                value = (alpha * self.belief[choice])\
                    / (alpha * self.belief[choice] + (1.0 - alpha) * (1.0 - self.belief[choice]))
            evidence = (choice, value)
        return evidence


class DampenedAgent(ProbabilisticAgent):
    """
    An agent adopting a probabilistic model of belief representations
    and consensus formation.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2):
        """
        A dampened variant of the probabilistic .
        In this variant, the variable lambda is used to prevent agents from
        reaching absolute certainty.
        """

        # Combine the belief matrices by flipping a coin for any states on
        # which the two beliefs disagree, and adopting the rest.
        new_belief = np.array([
            (belief1[i] * belief2[i]) /
            (belief1[i] * belief2[i] + (1.0 - belief1[i]) * (1.0 - belief2[i]))
            for i in range(len(belief1))
        ])

        # print("Before:", new_belief)

        # Jonathan's preferred lambda value
        var_lambda = 0.01
        new_belief = np.array([
            (var_lambda * 0.5) + ((1 - var_lambda) * belief)
            for belief in new_belief
        ])

        # print("After:", new_belief)

        invalid_belief = np.isnan(np.sum(new_belief))

        if not invalid_belief:
            return new_belief
        else:
            return None


class AverageAgent(ProbabilisticAgent):
    """
    An agent adopting a probabilistic model of belief representations
    and consensus formation.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2):
        """
        A dampened variant of Bayesian updating.
        In this variant, the variable lambda is used to prevent agents from
        reaching absolute certainty.
        """

        # Combine the belief matrices by flipping a coin for any states on
        # which the two beliefs disagree, and adopting the rest.
        new_belief = np.array([
            (belief1[i] + belief2[i]) / 2.0 for i in range(len(belief1))
        ])

        invalid_belief = np.isnan(np.sum(new_belief))

        if not invalid_belief:
            return new_belief
        else:
            return None


class ErrorCorrectingAgent(Agent):
    """
    This agent adopts a three-valued logic for uncertain belief representation, but instead of
    learning information from other agents, it seeks only to become less certain about conflicting
    propositions. Hence, this agent only fuses their beliefs in the pursuit of removing errors.
    """

    def __init__(self, belief):
        super().__init__(belief)


    @staticmethod
    def consensus(belief1, belief2):
        """
        Upon conflict, become uncertain about that specific proposition. Do not learn about
        uncertain propositions from other agents.

        This version is asymmetric, and so we will assume belief 1 is the focal agent.
        """

        # import inspect
        # current_frame = inspect.currentframe()
        # caller_frame = inspect.getouterframes(current_frame, 1)
        # print("Called by:", caller_frame[1][3])

        print("Beliefs:", belief1, belief2)

        new_belief = np.array([
            0 if truth_values[0] != truth_values[1]
            and truth_values[0] != 0 and truth_values[1] != 0
            else truth_values[0]
            for truth_values in zip(belief1, belief2)
        ])

        print("New belief:", new_belief)

        return new_belief


    def evidential_updating(self, true_state, noise_value, random_instance):
        """
        Update the agent's belief based on the evidence they received.
        Increment the evidence counter.

        Call the Agent class's consensus operator for proper updating based
        on evidence.
        """

        evidence = self.random_evidence(
            true_state,
            noise_value,
            random_instance
        )

        new_belief = super().consensus(self.belief, evidence)

        # Track the number of iterations.
        if np.array_equal(self.belief, new_belief):
            self.since_change += 1
        else:
            self.since_change = 0

        self.belief = new_belief
        self.evidence += 1