# This file contains the functions for calculating results from the agents'
# preferences, as well as writing the results to a file.

import numpy as np

def loss(belief, true_state, normalised = True):
    """
    Named after the loss function in machine learning, this function calculates
    the sum of the differences of two matrices, scaled by 0.5 so that we count
    half-values for having no preference between two elements.
    The idea is that we compare the agent's belief with the true state of
    the world and a return value of 0 indicates no differences (100% similarity)
    or "zero loss" between the true value and the agents' belief.

    The normalisation option uses the worst possible preference (in relation to
    the true state of the world) to normalise the values in [0, 1].
    """

    # Sum all of the inconsistencies between the true state of the world and the
    # agent's belief.
    # Divide answer by 2 as truth values are -1 (false), 0 (uncertain), 1 (true).
    differences = np.sum(abs(np.subtract(belief, true_state)))/2.0

    # If normalising the result (default) then divide the sum of the differences
    # by the length of the maximum number of pairs of relations.
    if normalised:
        return differences / len(true_state)

    return differences


def entropy(distribution, num_of_agents):
    """
    Population entropy where an uncertain belief about some proposition gives 0
    information.

    This is calculated as the normalised sum of each propositional variable in a
    belief.
    """

    normalised_distribution = np.array(distribution)/num_of_agents

    return (1/len(distribution) * - np.sum([x * np.log2(x) for x in normalised_distribution]))


def write_to_file(directory, file_name, params, data, max, array_data = False):
    """
    Write the results arrays to a file. The array_data argument allows us to write
    nested (array) data for recording the agents' averaged preferences for each state
    while reusing the same function for writing single value averages, e.g., loss.
    """

    with open(directory + file_name + '_' + '_'.join(params) + '.csv', 'w') as file:
        for i, test_data in enumerate(data):
            for j, results_data in enumerate(test_data):
                if array_data:
                    file.write('[')
                    for k, sub_data in enumerate(results_data):
                        file.write('{:.4f}'.format(sub_data))
                        if k != len(results_data) - 1:
                            file.write(',')
                    file.write(']')
                else:
                    file.write('{:.4f}'.format(results_data))
                # Determine whether the line ends here
                if j != len(test_data) - 1:
                    file.write(',')
                else:
                    file.write('\n')
            if i > max:
                break

