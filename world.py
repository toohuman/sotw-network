import argparse
import importlib
import random
import sys

import numpy as np

# Add the directory paths so that the submodules can also find the relative modules.
sys.path.append("sotw")
sys.path.append("CascadingFailure")

# from agents.agent import Agent
from graph.node import Node
from graph.hub import Hub
from utilities import operators
from utilities import beliefs
from utilities import results

tests = 100
iteration_limit = 5000
steady_state_threshold = 100

mode = "symmetric" # ["symmetric" | "asymmetric"]
evidence_only = False
demo_mode = False

# List of evidence rates
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
# Set a single evidence rate to begin with, in case we don't test the whole list
# and only want to experiment with a preset evidence rate.
evidence_rate = 10/100
# List of noise values: 0 would mean that evidence is always accurate.
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_value = 0.2 # None

# Set the initialisation function for agent beliefs - option to add additional
# initialisation functions later.
init_beliefs = beliefs.ignorant_belief_generator

def initialisation(
        num_of_nodes, num_of_hubs, states, agents: [], edges: [],
        partition: bool, random_instance
):
    """
    This initialisation function runs before any other part of the code. Starting with
    the creation of agents and the initialisation of relevant variables.
    """

    if partition:
        # TODO: Implement ...
        sys.exit(0)
    else:
        agents += [Node(init_beliefs(states)) for x in range(num_of_nodes)]
        agents += [Hub(init_beliefs(states)) for x in range(num_of_hubs)]

        edges += [(j, (j * num_of_nodes) + i) for j in range(num_of_hubs) for i in range(num_of_nodes)]

        print(edges)

    return

def main_loop(agents: [], edges: [], states: int, true_state: [], exploration: str, mode: str, random_instance):
    """
    The main loop performs various actions in sequence until certain conditions are
    met, or the maximum number of iterations is reached.
    """

    # For each agent, provided that the agent is to receive evidence this iteration
    # according to the current evidence rate, have the agent perform evidential
    # updating.
    reached_convergence = True
    for agent in agents:

        if random_instance.random() <= evidence_rate:

            # Currently, just testing with random evidence.
            evidence = beliefs.random_evidence(
                agent.belief,
                agent.region,
                true_state,
                exploration,
                noise_value,
                random_instance
            )

            agent.evidential_updating(operators.combine(agent.belief, evidence))

        reached_convergence &= agent.steady_state(steady_state_threshold)

    if reached_convergence:
        return False
    elif evidence_only:
        return True

    # Agents then combine at random

    # Symmetric
    if mode == "symmetric":
        agent1 = agents[random_instance.randint(0,len(agents) - 1)]
        agent2 = agent1
        while agent2 == agent1:
            agent2 = agents[random_instance.randint(0,len(agents) - 1)]

        new_belief = operators.combine(agent1.belief, agent2.belief)

        # Symmetric, so both agents adopt the combination belief.
        agent1.update_belief(new_belief)
        agent2.update_belief(new_belief)

    # Asymmetric
    # if mode == "asymmetric":
    #   ...

    return True


def main():
    """
    Main function for simulation experiments. Allows us to initiate start-up
    separately from main loop, and to extract results from the main loop at
    request. For example, the main_loop() will return FALSE when agents have
    fully converged according to no. of interactions unchanged. Alternatively,
    data can be processed for each iteration, or each test.
    """

    # Parse the arguments of the program, e.g., agents, states, random init.
    parser = argparse.ArgumentParser(description="Distributed\
    decision-making in a multi-agent environment in which agents must reach a consensus about the true state of the world.")
    # Nodes: Number of standard nodes (e.g., submarines).
    parser.add_argument("nodes", type=int)
    # Hubs: Number of hub nodes (e.g., ships) to allocate - the k in k-means partitioning.
    parser.add_argument("hubs", type=int)
    parser.add_argument("states", type=int)
    parser.add_argument("-r", "--random", type=bool, help="Random seeding of the RNG.")
    parser.add_argument("-p", "--partition", action="store_true", help="Partition agents into separate regions.")
    parser.add_argument("-e", "--explore", action="store_true", help="Explore previously visited states, even if the agent holds a belief about them.")
    parser.add_argument("-er", "--explore-randomly", action="store_true", help="Explore states uniformly at random.")
    arguments = parser.parse_args()

    exploration = None

    if arguments.explore and arguments.explore_randomly:
        sys.exit("Cannot have multiple exploration methods active.")
    elif arguments.explore:
        exploration = "revisit"
    elif arguments.explore_randomly:
        exploration = "random"

    # Create an instance of a RNG that is either seeded for consistency of simulation
    # results, or create using a random seed for further testing.
    random_instance = random.Random()
    random_instance.seed(128) if arguments.random == None else random_instance.seed()

    # Output variables
    directory = "../results/test_results/sotw/"
    file_name_params = []

    global evidence_rate
    global noise_value

    print("Evidence rate:", evidence_rate)
    print("Noise value:", noise_value)

    loss_results = [
        [ 0.0 for y in range(tests) ] for z in range(iteration_limit + 1)
    ]
    loss_results = np.array(loss_results)

    # Repeat the initialisation and loop for the number of simulation runs required
    max_iteration = 0
    for test in range(tests):

        # True state of the world
        true_state = np.array(
            [
                [random_instance.choice([-1,1]) for x in range(arguments.states)]
                for y in range(arguments.states)
            ]
        )

        agents = list()
        edges = list()

        # Initialise the agents and the environment.
        # If we are to partition the space, we need to assign agents regions of the
        # total grid space.
        initialisation(
            arguments.nodes,
            arguments.hubs,
            arguments.states,
            agents,
            edges,
            arguments.partition,
            random_instance
        )

        # Pre-loop results based on agent initialisation.
        for agent in agents:
            loss_results[0][test] += results.loss(agent.belief, true_state)

        # Main loop of the experiments. Starts at 1 because we have recorded the agents'
        # initial state above, at the "0th" index.
        for iteration in range(1, iteration_limit + 1):
            print("Test #" + str(test) + " - Iteration #" + str(iteration) + "  ", end="\r")

            max_iteration = iteration if iteration > max_iteration else max_iteration
            # While not converged, continue to run the main loop.
            if main_loop(agents, arguments.states, true_state, exploration, mode, random_instance):
                for agent in agents:
                    loss_results[iteration][test] += results.loss(agent.belief, true_state)

            # If the simulation has converged, end the test.
            else:
                # print("Converged: ", iteration)
                for agent in agents:
                    loss_results[iteration][test] += results.loss(agent.belief, true_state)
                for iter in range(iteration + 1, iteration_limit + 1):
                    loss_results[iter][test] = np.copy(loss_results[iteration][test])
                # Simulation has converged, so break main loop.
                break
    print()

    # Post-loop results processing (normalisation).
    loss_results /= len(agents)

    # Recording of results. First, add parameters in sequence.
    file_name_params.append("{}_agents".format(arguments.agents))
    file_name_params.append("{}_states".format(arguments.states))
    file_name_params.append("{:.3f}_er".format(evidence_rate))
    if noise_value is not None:
        file_name_params.append("{:.3f}_nv".format(noise_value))
    if arguments.partition:
        file_name_params.append("part")
    if arguments.explore:
        file_name_params.append("xplr")
    if arguments.explore_randomly:
        file_name_params.append("xplr_rand")

    results.write_to_file(
        directory,
        "loss",
        file_name_params,
        loss_results,
        max_iteration
    )

    # if demo_mode:
        # Output plots while running simulations, but do not record the results.
    # else:
        # Record the results but skip the plotting.


if __name__ == "__main__":

    test_set = "standard" # "standard" | "evidence" | "noise" | "both"

    if test_set == "standard":

        # Profiling setup.
        # import cProfile, pstats, io
        # pr = cProfile.Profile()
        # pr.enable()
        # END

        main()

        # Profile post-processing.
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        # END

    elif test_set == "evidence":

        for er in evidence_rates:
            evidence_rate = er
            main()

    elif test_set == "noise":

        for nv in noise_values:
            noise_value = nv
            main()

    elif test_set == "both":

        for er in evidence_rates:
            evidence_rate = er

            for nv in noise_values:
                noise_value = nv
                main()