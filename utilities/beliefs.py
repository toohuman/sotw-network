import random

import numpy as np

def ignorant_belief_generator(states):
    """ Returns an empty belief matrix to denote complete uncertainty. """

    return np.full((states, states), 0, int)


def random_evidence(belief, region, true_state, noise_value, random_instance):
    """ Generate a random piece of evidence. """

    evidence = np.full((len(belief), len(belief)), 0, int)

    # If agents have not be partitioned into regions.
    if region is None:
        unknowns = np.array([[x, y] for x in range(len(belief)) for y in range(len(belief))])

    else:

        unknowns = np.array([[x, y] for x in range(region[0][0], region[1][0] + 1) for y in range(region[0][1], region[1][1] + 1)])

    if len(unknowns) > 0:
        choice = random_instance.choice(unknowns)

        evidence[choice[0]][choice[1]] = true_state[choice[0]][choice[1]] if random_instance.random() > noise_value else true_state[choice[0]][choice[1]] * -1

    return evidence