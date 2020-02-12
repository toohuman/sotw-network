import argparse
import copy
import importlib
import networkx as nx
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
iteration_limit = 10_000
steady_state_threshold = 100

mode = "symmetric" # ["symmetric" | "asymmetric"]
evidence_only = False
# demo_mode should be used to visualise performance live during simulation run
demo_mode = False

evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_rate = 10/100
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_value = 0.2
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_value = 0.0

# Set the initialisation function for agent beliefs - option to add additional
# initialisation functions later.
init_beliefs = beliefs.ignorant_belief

# TODO:
# 1. Remove separate agents(nodes) and edges lists, using only network instead.

def initialisation(
    num_of_nodes, num_of_hubs, states, agents: [], edges: [], network,
    connectivity, random_instance
):
    """
    This initialisation function runs before any other part of the code. Starting with
    the creation of agents and the initialisation of relevant variables.
    """

    identity = 0
    # If there should be a separation of hubs from nodes, then use the standard implementation
    # until an alternative networkx implementation has been written.
    if num_of_hubs > 0:

        # Add the "node" agents to the list of agents first.
        agents += [Node(init_beliefs(states)) for x in range(num_of_nodes * num_of_hubs)]
        # Then add the "hub" agents.
        agents += [Hub(init_beliefs(states)) for x in range(num_of_hubs)]
        # Set agent's identity.
        for a in range(len(agents)):
            agents[a].set_identity = identity
            identity += 1

        # Finally, generate a list of edges in which each hub is connected to all of its
        # respective nodes and each hub is also connected to every other hub.
        edges += [((num_of_hubs * num_of_nodes) + i, (i * num_of_nodes) + j) for i in range(num_of_hubs) for j in range(num_of_nodes)]
        edges += [((num_of_hubs * num_of_nodes) + i, (num_of_hubs * num_of_nodes) + j) for i in range(num_of_hubs) for j in range(i + 1, num_of_hubs)]

        # A complete graph using networkx:
        # network = nx.complete_graph(agents)
    else:
        # When there are no hubs, implement random graphs with a connectivity parameter k

        agents += [Node(init_beliefs(states)) for x in range(num_of_nodes)]
        edges  += nx.gnp_random_graph(len(agents), connectivity, random_instance).edges
        network.update(edges, agents)

    return

def main_loop(
    agents: [], edges: [], states: int, true_state: [], mode: str, random_instance
):
    """
    The main loop performs various actions in sequence until certain conditions are
    met, or the maximum number of iterations is reached.
    """

    # For each agent, provided that the agent is to receive evidence this iteration
    # according to the current evidence rate, have the agent perform evidential
    # updating.
    reached_convergence = True
    for agent in agents:

        # Hubs do not receive direct evidence, only updating their beliefs based on
        # information from other nodes.
        # if isinstance(agent, Node) and random_instance.random() <= evidence_rate:
        if random_instance.random() <= evidence_rate:

            # Generate a random piece of evidence, selecting from the set of unknown states.
            evidence = beliefs.random_evidence(
                agent.belief,
                agent.region,
                true_state,
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

        try:
            chosen_nodes = random_instance.choice(edges)
        except IndexError:
            return True

        agent1, agent2 = agents[chosen_nodes[0]], agents[chosen_nodes[1]]

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
    parser = argparse.ArgumentParser(description="Distributed decision-making\
        in a multi-agent environment in which agents must reach a consensus\
            about the true state of the world.")
    parser.add_argument("states", type=int)
    # Nodes: Number of standard nodes (e.g., submarines).
    parser.add_argument("nodes", type=int)
    # Hubs: Number of hub nodes (e.g., ships) to allocate - the k in k-means partitioning.
    parser.add_argument("--hubs", type=int, default=0,
        help="Hubs are the pass-through nodes for other nodes.\
        They do not receive direct evidence.")
    parser.add_argument("-c", "--connectivity", type=float, help="Connectivity of the random graph in [0,1],\
        e.g., probability of an edge between any two nodes.")
    parser.add_argument("-r", "--random", action="store_true", help="Random seeding of the RNG.")
    arguments = parser.parse_args()

    if connectivity_value is not None:
        arguments.connectivity = connectivity_value

    if arguments.hubs == 0 and arguments.connectivity is None:
        print("Usage error: Connectivity must be specified for node-only graph.")
        sys.exit(0)

    # Create an instance of a RNG that is either seeded for consistency of simulation
    # results, or create using a random seed for further testing.
    random_instance = random.Random()
    random_instance.seed(128) if arguments.random == False else random_instance.seed()

    # Output variables
    directory = "../results/test_results/sotw-network/"
    file_name_params = []

    print("Connectivity:", arguments.connectivity)
    print("Evidence rate:", evidence_rate)
    print("Noise value:", noise_value)

    # Structure should be [mean, std_dev, min, max]
    global_loss_results = [
        [ [0.0 for x in range(4)] for y in range(tests) ] for z in range(iteration_limit + 1)
    ]
    global_loss_results = np.array(global_loss_results)
    # node_loss_results = np.copy(global_loss_results)
    # hub_loss_results = np.copy(global_loss_results)
    steady_state_results = [
        [ 0.0 for y in range(arguments.nodes) ] for z in range(tests)
    ]
    steady_state_results = np.array(steady_state_results)

    # Repeat the initialisation and loop for the number of simulation runs required
    max_iteration = 0
    for test in range(tests):

        # True state of the world
        true_state = np.array([random_instance.choice([-1,1]) for x in range(arguments.states)])

        agents = list()
        edges = list()
        network = nx.Graph()

        # Initialise the agents and the environment.
        # If we are to partition the space, we need to assign agents regions of the
        # total grid space.
        initialisation(
            arguments.nodes,
            arguments.hubs,
            arguments.states,
            agents,
            edges,
            network,
            arguments.connectivity,
            random_instance
        )

        # Reusable vector for loss values of population
        loss_values = np.array([0.0 for x in range(arguments.nodes)])

        # Pre-loop results based on agent initialisation.
        for a, agent in enumerate(agents):
            loss_values[a] = results.loss(agent.belief, true_state)

        global_loss_results[0][test] = [
            np.average(loss_values),
            np.std(loss_values),
            np.min(loss_values),
            np.max(loss_values)
        ]

        # Main loop of the experiments. Starts at 1 because we have recorded the agents'
        # initial state above, at the "0th" index.
        for iteration in range(1, iteration_limit + 1):
            print("Test #{} - Iteration #{}    ".format(test, iteration), end="\r")

            max_iteration = iteration if iteration > max_iteration else max_iteration
            # While not converged, continue to run the main loop.
            if main_loop(agents, edges, arguments.states, true_state, mode, random_instance):
                for a, agent in enumerate(agents):
                    loss = results.loss(agent.belief, true_state)
                    loss_values[a] = loss
                    # if isinstance(agent, Node):
                    #     node_loss_results[iteration][test] += results.loss(agent.belief, true_state)
                    # elif isinstance(agent, Hub):
                    #     hub_loss_results[iteration][test] += results.loss(agent.belief, true_state)
                    if iteration == iteration_limit:
                        steady_state_results[test][a] = loss

                global_loss_results[0][test] = [
                    np.average(loss_values),
                    np.std(loss_values),
                    np.min(loss_values),
                    np.max(loss_values)
                ]

            # If the simulation has converged, end the test.
            else:
                # print("Converged: ", iteration)
                for a, agent in enumerate(agents):
                    loss = results.loss(agent.belief, true_state)
                    loss_values[a] = loss
                    # global_loss_results[iteration][test] += loss
                    # if isinstance(agent, Node):
                    #     node_loss_results[iteration][test] += results.loss(agent.belief, true_state)
                    # elif isinstance(agent, Hub):
                    #     hub_loss_results[iteration][test] += results.loss(agent.belief, true_state)
                    steady_state_results[test][a] = loss

                global_loss_results[0][test] = [
                    np.average(loss_values),
                    np.std(loss_values),
                    np.min(loss_values),
                    np.max(loss_values)
                ]

                for iter in range(iteration + 1, iteration_limit + 1):
                    global_loss_results[iter][test] = np.copy(global_loss_results[iteration][test])
                    # node_loss_results[iter][test] = np.copy(node_loss_results[iteration][test])
                    # hub_loss_results[iter][test] = np.copy(hub_loss_results[iteration][test])
                # Simulation has converged, so break main loop.
                break
    print()

    # Recording of results. First, add parameters in sequence.
    
    # Networkx params
    file_name_params.append("{}_states".format(arguments.states))
    file_name_params.append("{}_nodes".format(arguments.nodes))
    if arguments.hubs != 0:
        file_name_params.append("{}_hubs".format(arguments.hubs))
    if arguments.connectivity is not None:
        file_name_params.append("{}_con".format(arguments.connectivity))
    file_name_params.append("{:.3f}_er".format(evidence_rate))
    if noise_value is not None:
        file_name_params.append("{:.3f}_nv".format(noise_value))

    results.write_to_file(
        directory,
        "loss",
        file_name_params,
        global_loss_results,
        max_iteration
    )
    results.write_to_file(
        directory,
        "steady_state_loss",
        file_name_params,
        steady_state_results,
        tests
    )

    # TODO: Implement hub/node separation results using networkx.

    # results.write_to_file(
    #     directory,
    #     "node_loss",
    #     file_name_params,
    #     node_loss_results,
    #     max_iteration
    # )

    # results.write_to_file(
    #     directory,
    #     "hub_loss",
    #     file_name_params,
    #     hub_loss_results,
    #     max_iteration
    # )


if __name__ == "__main__":

    test_set = "enc" # "standard" | "evidence" | "noise" | "en" | "enc"

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

    elif test_set == "en":

        for er in evidence_rates:
            evidence_rate = er

            for nv in noise_values:
                noise_value = nv
                main()

    elif test_set == "enc":

        for con in connectivity_values:
            connectivity_value = con

            for er in evidence_rates:
                evidence_rate = er

                for nv in noise_values:
                    noise_value = nv
                    main()