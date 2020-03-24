import argparse
import lzma
import networkx as nx
import pickle
import random
import sys

import numpy as np

# Add the directory paths so that the submodules can also find the relative modules.
sys.path.append("sotw")

# from agents.agent import Agent
from graph.node import Node
from graph.hub import Hub
from sotw.agents.agent import Agent
from utilities import operators
from utilities import beliefs
from utilities import results

tests = 100
iteration_limit = 10_000
steady_state_threshold = 100
trajectory_populations = [10, 50, 100]

# Set the graph type
graph_type = "ER"         # Erdos-Reyni: random
# graph_type = "WS"         # Watts-Strogatz: small-world
# graph_type = "Star"       # Star (hub-and-spoke) topology
# graph_type = "Ring"       # Ring topology
# graph_type = "Line"       # Line topology
# graph_type = "PATH:0"     # Manually construct pathological cases, e.g., star, ring.

mode = "symmetric" # ["symmetric" | "asymmetric"]
evidence_only = False
# demo_mode should be used to visualise performance live during simulation run
demo_mode = False

evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_rate = None
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_value = None
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_value = 1.0
knn_values = [2, 4, 6, 8, 10, 20, 50]
k_nearest_neighbours = None

# Set the initialisation function for agent beliefs - option to add additional
# initialisation functions later.
init_beliefs = beliefs.ignorant_belief

# TODO:
# 1. Remove separate agents(nodes) and edges lists, using only network instead.

def initialisation(
    num_of_agents, num_of_hubs, states, network, connectivity, knn, random_instance
):
    """
    This initialisation function
    print(steady_state_results)runs before any other part of the code. Starting with
    the creation of agents and the initialisation of relevant variables.
    """

    identity = 0
    agents = list()
    edges = list()

    # If there should be a separation of hubs from nodes, then use the standard implementation
    # until an alternative networkx implementation has been written.
    if num_of_hubs > 0:

        # Add the "node" agents to the list of agents first.
        agents += [Node(init_beliefs(states)) for x in range(num_of_agents * num_of_hubs)]
        # Then add the "hub" agents.
        agents += [Hub(init_beliefs(states)) for x in range(num_of_hubs)]
        # Set agent's identity.
        for a in range(len(agents)):
            agents[a].set_identity = identity
            identity += 1

        # Finally, generate a list of edges in which each hub is connected to all of its
        # respective nodes and each hub is also connected to every other hub.
        edges += [((num_of_hubs * num_of_agents) + i, (i * num_of_agents) + j) for i in range(num_of_hubs) for j in range(num_of_agents)]
        edges += [((num_of_hubs * num_of_agents) + i, (num_of_hubs * num_of_agents) + j) for i in range(num_of_hubs) for j in range(i + 1, num_of_hubs)]

        # A complete graph using networkx:
        # network = nx.complete_graph(agents)
    else:
        agents += [Agent(init_beliefs(states)) for x in range(num_of_agents)]

        # Produce a random graph (Erdos-Renyi) with a connectivity parameter p
        if graph_type == "ER":
            edges  += nx.gnp_random_graph(len(agents), connectivity, random_instance).edges
        # Produce a random small-world graph (Watts-Strogatz) with k nearest neighbours
        # and a connectivity parameter p
        elif graph_type == "WS":
            edges  += nx.watts_strogatz_graph(len(agents), knn, connectivity, random_instance).edges
        elif graph_type == "Star":
            hub = random_instance.choice(range(len(agents)))
            edges += [(hub, x) for x in range(len(agents)) if x != hub]
        elif graph_type == "Ring":
            edges += [(x, x+1) for x in range(len(agents) - 1)]
            edges += [(len(agents)-1, 0)]
        elif graph_type == "Line":
            edges += [(x, x+1) for x in range(len(agents) - 1)]

    # For Python 3.6+, dictionaries maintain a consistent order, so node/agent order
    # should be maintained.
    edges = map(lambda x: (agents[x[0]], agents[x[1]]), edges)
    network.update(edges, agents)

    return

def main_loop(
    states: int, network, true_state: [], mode: str, random_instance,
    entropy_data, error_data
):
    """
    The main loop performs various actions in sequence until certain conditions are
    met, or the maximum number of iterations is reached.
    """

    # Format: before, after evidence, after consensus.
    entropy_distributions = [0 for x in range(states)]
    entropy_diffs = [0 for x in range(states)]
    prior_belief = None

    # For each agent, provided that the agent is to receive evidence this iteration
    # according to the current evidence rate, have the agent perform evidential
    # updating.
    reached_convergence = True
    for agent in network.nodes:
        # Generate prior distribution
        for s, state in enumerate(agent.belief):
            entropy_distributions[s] += state

        # Hubs do not receive direct evidence, only updating their beliefs based on
        # information from other nodes.
        # if isinstance(agent, Node) and random_instance.random() <= evidence_rate:
        if random_instance.random() <= evidence_rate:

            prior_belief = agent.belief
            print(prior_belief)

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
            chosen_nodes = random_instance.choice(list(network.edges()))
        except IndexError:
            return True

        agent1, agent2 = chosen_nodes[0], chosen_nodes[1]

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
    parser.add_argument("agents", type=int)
    # Hubs: Number of hub nodes (e.g., ships) to allocate - the k in k-means partitioning.
    parser.add_argument("--hubs", type=int, default=0,
        help="Hubs are the pass-through nodes for other nodes.\
        They do not receive direct evidence.")
    parser.add_argument("-c", "--connectivity", type=float, help="Connectivity of the random graph in [0,1],\
        e.g., probability of an edge between any two nodes.")
    parser.add_argument("-k", "--knn", type=int, help="k nearest neighbours to which each node is connected.")
    parser.add_argument("-r", "--random", action="store_true", help="Random seeding of the RNG.")
    arguments = parser.parse_args()

    if arguments.connectivity is None and connectivity_value is not None:
        arguments.connectivity = connectivity_value
    if arguments.knn is None and k_nearest_neighbours is not None:
        arguments.knn = k_nearest_neighbours
        if arguments.knn > arguments.agents:
            return

    if arguments.hubs == 0 and arguments.connectivity is None and graph_type in ["ER", "WS"]:
        print("Usage error: Connectivity must be specified for node-only graph.")
        sys.exit(0)

    # Create an instance of a RNG that is either seeded for consistency of simulation
    # results, or create using a random seed for further testing.
    random_instance = random.Random()
    random_instance.seed(128) if arguments.random == False else random_instance.seed()

    # Output variables
    directory = "../results/test_results/sotw-network/"
    file_name_params = []

    param_strings = list()
    param_strings += ["States: {}".format(arguments.states)]
    param_strings += ["Agents: {}".format(arguments.agents)]
    param_strings += ["Connectivity: {} | {}".format(arguments.connectivity, graph_type)]
    if graph_type == "WS":
        param_strings += ["k: {}".format(arguments.knn)]
    param_strings += ["Evidence rate: {}".format(evidence_rate)]
    param_strings += ["Noise value: {}".format(noise_value)]
    print("\t".join(param_strings))

    # Structure is [mean, std_dev, min, max]
    loss_results = np.array([
        [[0.0 for x in range(4)] for y in range(tests)] for z in range(iteration_limit + 1)
    ])
    # Structure is [combined_info_gain, evidence_info_gain, consensus_info_gain]
    entropy_results = np.array([
        [[0.0 for x in range(3)] for y in range(tests)] for z in range(iteration_limit)
    ])
    # Structure is [combined_error, evidence_error, consensus_error]
    error_results = np.array([
        [[0.0 for x in range(3)] for y in range(tests)] for z in range(iteration_limit)
    ])
    steady_state_results = np.array([
        [0.0 for x in range(arguments.agents)] for y in range(tests)
    ])

    # Repeat the initialisation and loop for the number of simulation runs required
    max_iteration = 0
    for test in range(tests):

        # True state of the world
        true_state = np.array([random_instance.choice([-1,1]) for x in range(arguments.states)])

        network = nx.Graph()

        # Initialise the agents and the environment.
        # If we are to partition the space, we need to assign agents regions of the
        # total grid space.
        initialisation(
            arguments.agents,
            arguments.hubs,
            arguments.states,
            network,
            arguments.connectivity,
            arguments.knn,
            random_instance
        )

        # Reusable vector for loss values of population
        loss_values = np.array([0.0 for x in range(arguments.agents)])

        # Pre-loop results based on agent initialisation.
        for a, agent in enumerate(network.nodes):
            loss_values[a] = results.loss(agent.belief, true_state)

        loss_results[0][test] = [
            np.average(loss_values),
            np.std(loss_values),
            np.min(loss_values),
            np.max(loss_values)
        ]

        entropy_data = [0.0 for x in range(3)]
        error_data = [0.0 for x in range(3)]

        # Main loop of the experiments. Starts at 1 because we have recorded the agents'
        # initial state above, at the "0th" index.
        for iteration in range(1, iteration_limit + 1):
            print("Test #{} - Iteration #{}    ".format(test, iteration), end="\r")

            max_iteration = iteration if iteration > max_iteration else max_iteration
            # While not converged, continue to run the main loop.
            if main_loop(
                arguments.states, network, true_state, mode, random_instance,
                entropy_data, error_data
            ):
                for a, agent in enumerate(network.nodes):
                    loss = results.loss(agent.belief, true_state)
                    loss_values[a] = loss
                    if iteration == iteration_limit:
                        steady_state_results[test][a] = loss

                loss_results[iteration][test] = [
                    np.average(loss_values),
                    np.std(loss_values),
                    np.min(loss_values),
                    np.max(loss_values)
                ]

            # If the simulation has converged, end the test.
            else:
                for a, agent in enumerate(network.nodes):
                    loss = results.loss(agent.belief, true_state)
                    loss_values[a] = loss
                    # loss_results[iteration][test] += loss
                    steady_state_results[test][a] = loss

                loss_results[iteration][test] = [
                    np.average(loss_values),
                    np.std(loss_values),
                    np.min(loss_values),
                    np.max(loss_values)
                ]

                for iter in range(iteration + 1, iteration_limit + 1):
                    loss_results[iter][test] = np.copy(loss_results[iteration][test])
                # Simulation has converged, so break main loop.
                break

        # Reset the static identity for the Agent class.
        Agent.identity = 0

    print()

    # Recording of results. First, add parameters in sequence.

    file_name_params.append("{}s".format(arguments.states))
    file_name_params.append("{}a".format(arguments.agents))
    if arguments.hubs != 0:
        file_name_params.append("{}h".format(arguments.hubs))

    if graph_type == "ER":
        if arguments.connectivity is not None:
            file_name_params.append("{:.2f}con".format(arguments.connectivity))
    elif graph_type == "WS":
        if arguments.connectivity is not None and arguments.knn is not None:
            file_name_params.append("{}k".format(arguments.knn))
            file_name_params.append("{:.2f}con".format(arguments.connectivity))
    elif graph_type in ["Star", "Ring", "Line"]:
        file_name_params.append("{}".format(graph_type))

    file_name_params.append("{:.2f}er".format(evidence_rate))
    if noise_value is not None:
        file_name_params.append("{:.2f}nv".format(noise_value))

    # Write loss results to pickle file
    if arguments.agents in trajectory_populations:
        with lzma.open(directory + "loss" + '_' + '_'.join(file_name_params) + '.pkl.xz', 'wb') as file:
            pickle.dump(loss_results, file)

    # results.write_to_file(
    #     directory,
    #     "steady_state_loss",
    #     file_name_params,
    #     steady_state_results,
    #     tests
    # )

    with lzma.open(directory + "steady_state_loss" + '_' + '_'.join(file_name_params) + '.pkl.xz', 'wb') as file:
        pickle.dump(steady_state_results, file)

if __name__ == "__main__":

    test_set = "en"

    # "standard" | "evidence" | "noise" | "en" | "ce" | "cen" | "kce"

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

    elif test_set == "ce":

        for con in connectivity_values:
            connectivity_value = con

            for er in evidence_rates:
                evidence_rate = er
                main()

    elif test_set == "cen":

        for con in connectivity_values:
            connectivity_value = con

            for er in evidence_rates:
                evidence_rate = er

                for nv in noise_values:
                    noise_value = nv
                    main()

    elif test_set == "kce":

        for knn in knn_values:
            k_nearest_neighbours = knn

            for con in connectivity_values:
                connectivity_value = con

                for er in evidence_rates:
                    evidence_rate = er
                    main()