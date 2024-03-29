import argparse
import lzma
import networkx as nx
import pickle
import random
import sys

import numpy as np
from numpy.lib.function_base import _update_dim_sizes
from numpy.lib.stride_tricks import broadcast_arrays

# from agents.agent import Agent
from agents.agent import *
from utilities import results
from utilities import topologies

tests = 100
iteration_limit = 10_000
steady_state_threshold = 100
trajectory_populations = [10, 50, 100]

# Set the graph type
# Erdos-Reyni: random | Watts-Strogatz: small-world.
random_graphs = ["ER", "WS", "BA"]
# What we are calling "pathological" cases.
specialist_graphs = ["line", "star"]
clique_graphs = [
    "connected_star", "complete_star",
    "caveman", "complete_caveman"
]
graph_type = "ER"

evidence_only = False
update_type = "Asymmetric"    # Asymmetric

fusion_rates = [1, 5, 10, 20, 30, 40, 50]   # Number of pairs of agents to be selected for belief fusion
fusion_rate = 1
fusion_probs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]   # The probability with which each agent shall listen and update its belief
fusion_prob = 1.0
evidence_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
evidence_rate = 1.0
noise_values = [0.3, 0.4, 0.5] # [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
noise_value = 0.0
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_value = 1.0
knn_values = [2, 4, 6, 8, 10, 20, 50]
k_nearest_neighbours = None
m_values = [1, 2, 3, 4, 5, 10, 30, 50]
m_value = 1
clique_size = 10

# Set the type of agent: three-valued, voter or probabilistic
# (Three-valued) Agent | VoterAgent | StochasticAgent
# ProbabilisticAgent | DampenedAgent | AverageAgent
# CautiousAdventurousAgent
agent_type = Agent
# Set the initialisation function for agent beliefs - option to add additional
# initialisation functions later.
init_beliefs = agent_type.ignorant_belief

# TODO:
# 1. Remove separate agents(nodes) and edges lists, using only networkx instead.

def initialisation(
    num_of_agents, states, network, connectivity, knn, m, random_instance
):
    """
    This initialisation function
    print(steady_state_results)runs before any other part of the code. Starting with
    the creation of agents and the initialisation of relevant variables.
    """

    topology = topologies.Topologies()

    identity = 0
    agents = list()
    edges = list()

    if agent_type.__name__ == "VoterAgent":
        agents += [agent_type(init_beliefs(states, random_instance)) for x in range(num_of_agents)]
    # if agent_type.__name__ == "Agent" or agent_type.__name__ == "ProbabilisticAgent":
    else:
        agents += [agent_type(init_beliefs(states)) for x in range(num_of_agents)]

    # Produce a random graph (Erdos-Renyi) with a connectivity parameter p
    if graph_type == "ER":
        edges += nx.gnp_random_graph(len(agents), connectivity, random_instance).edges
    # Produce a random small-world graph (Watts-Strogatz) with k nearest neighbours
    # and a connectivity parameter p
    elif graph_type == "WS":
        edges += nx.watts_strogatz_graph(len(agents), knn, connectivity, random_instance).edges
    elif graph_type == "BA":
        edges += nx.barabasi_albert_graph(len(agents), m, random_instance).edges
    else:
        try:
            edges += getattr(topology, graph_type)(len(agents), clique_size, random_instance)
        except AttributeError:
            sys.exit("Topology does not match a corresponding topology generator function.")

    # For Python 3.6+, dictionaries maintain a consistent order, so node/agent order
    # should be maintained.
    edges = map(lambda x: (agents[x[0]], agents[x[1]]), edges)
    network.update(edges, agents)

    return

def main_loop(
    states: int, network, true_state: list(), random_instance,
    entropy_data, error_data
):
    """
    The main loop performs various actions in sequence until certain conditions are
    met, or the maximum number of iterations is reached.
    """

    # Format: before, after evidence, after consensus.
    entropy_distributions = [0 for x in range(states)]
    entropy_diffs = [0 for x in range(states)]

    # For each agent, provided that the agent is to receive evidence this iteration
    # according to the current evidence rate, have the agent perform evidential updating.
    reached_convergence = True
    for agent in network.nodes:
        # Generate prior distribution
        for s, state in enumerate(agent.belief):
            entropy_distributions[s] += state

        if random_instance.random() <= evidence_rate:

            # Have the agent update their belief with some piece of evidence.
            agent.evidential_updating(
                true_state,
                noise_value,
                random_instance
            )

        reached_convergence &= agent.steady_state(steady_state_threshold)

    if reached_convergence:
        return False
    elif evidence_only:
        return True

    # Agents then combine at random

    if update_type == "Symmetric":

        network_copy = network.copy()

        if fusion_rate is not None:
            num_of_edges = int(network.number_of_nodes() * (fusion_rate/100))
        else:
            num_of_edges = 1

        for i in range(num_of_edges):
            try:
                agent1, agent2 = random_instance.choice(list(network_copy.edges))
            except IndexError:
                return True

            if agent_type.__name__ in ["VoterAgent", "StochasticAgent", "CautiousAdventurousAgent"]:
                new_belief = agent_type.consensus(
                    agent1.belief, agent2.belief, random_instance
                )
            elif agent_type.__name__ in ["ErrorCorrectingAgent"]:
                new_belief = None
                agent1.update_belief(agent_type.consensus(agent1.belief, agent2.belief))
                agent2.update_belief(agent_type.consensus(agent2.belief, agent1.belief))
            else:
                new_belief = agent_type.consensus(agent1.belief, agent2.belief)

            if new_belief is not None:
                # Symmetric, so both agents adopt the combination belief.
                agent1.update_belief(new_belief)
                agent2.update_belief(new_belief)

            network_copy.remove_node(agent1)
            network_copy.remove_node(agent2)

    elif update_type == "Asymmetric":
        for agent in network.nodes:
            if random_instance.random() < fusion_prob:
                try:
                    broadcasting_agents = list(nx.node_connected_component(network, agent))
                    agent.obtained_belief = random_instance.choice(broadcasting_agents).belief
                except IndexError:
                    continue
            else:
                agent.obtained_belief = None
        for agent in network.nodes:
            if agent.obtained_belief is not None:
                agent.update_belief(agent_type.consensus(agent.belief, agent.obtained_belief))

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
    parser.add_argument("agents", type=int)
    parser.add_argument("-c", "--connectivity", type=float, help="Connectivity of the random graph in [0,1],\
        e.g., probability of an edge between any two nodes.")
    parser.add_argument("-k", "--knn", type=int, help="k nearest neighbours to which each node is connected.")
    parser.add_argument("-m", "--m", type=int, help="Number of edges to attach from a new node to existing nodes.")
    parser.add_argument("-r", "--random", action="store_true", help="Random seeding of the RNG.")
    arguments = parser.parse_args()

    if arguments.connectivity is None and connectivity_value is not None:
        arguments.connectivity = connectivity_value
    if arguments.knn is None and k_nearest_neighbours is not None:
        arguments.knn = k_nearest_neighbours
        if arguments.knn > arguments.agents:
            return
    if arguments.m is None and m_value is not None:
        arguments.m = m_value
        if arguments.m > arguments.agents:
            return

    if arguments.connectivity is None and graph_type in random_graphs:
        print("Usage error: Connectivity must be specified for node-only graph.")
        sys.exit(0)

    # Create an instance of a RNG that is either seeded for consistency of simulation
    # results, or create using a random seed for further testing.
    random_instance = random.Random()
    random_instance.seed(128) if arguments.random == False else random_instance.seed()

    # Output variables
    directory = "../results/test_results/sotw-network-temp/{}/".format(agent_type.__name__.lower())
    # directory = "../results/test_results/sotw-network/"
    file_name_params = []

    param_strings = list()
    param_strings += ["States: {}".format(arguments.states)]
    param_strings += ["Agents: {} - {}".format(arguments.agents, agent_type.__name__)]
    param_strings += ["Connectivity: {} | {}".format(arguments.connectivity, graph_type)]
    if graph_type == "WS":
        param_strings += ["k: {}".format(arguments.knn)]
    elif graph_type == "BA":
        param_strings += ["m: {}".format(arguments.m)]
    if update_type == "Symmetric":
        param_strings += ["Fusion rate: {}".format(fusion_rate)]
    elif update_type == "Asymmetric":
        param_strings += ["Fusion prob: {}".format(fusion_prob)]
    param_strings += ["Evidence rate: {}".format(evidence_rate)]
    param_strings += ["Noise value: {}".format(noise_value)]
    print("    ".join(param_strings))

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
        if agent_type.__name__ in ["Agent", "ErrorCorrectingAgent"]:
            true_state = np.array([random_instance.choice([-1,1]) for x in range(arguments.states)])
        # if agent_type.__name__ == "VoterAgent" or agent_type.__name__ == "ProbabilisticAgent":
        else:
            true_state = np.array([random_instance.choice([0,1]) for x in range(arguments.states)])

        network = nx.Graph()

        # Initialise the agents and the environment.
        # If we are to partition the space, we need to assign agents regions of the
        # total grid space.
        initialisation(
            arguments.agents,
            arguments.states,
            network,
            arguments.connectivity,
            arguments.knn,
            arguments.m,
            random_instance
        )

        # Reusable vector for loss values of population
        loss_values = np.array([0.0 for x in range(arguments.agents)])

        # Pre-loop results based on agent initialisation.
        for a, agent in enumerate(network.nodes):
            loss_values[a] = results.loss(agent_type, agent.belief, true_state)

        loss_results[0][test] = [
            np.average(loss_values),
            np.std(loss_values),
            np.min(loss_values),
            np.max(loss_values)
        ]

        entropy_data = [0.0 for x in range(3)]
        error_data = [0.0 for x in range(3)]

        # print(loss_results[0][test][0])

        # Main loop of the experiments. Starts at 1 because we have recorded the agents'
        # initial state above, at the "0th" index.
        for iteration in range(1, iteration_limit + 1):
            print("Test #{} - Iteration #{}    ".format(test + 1, iteration), end="\r")

            max_iteration = iteration if iteration > max_iteration else max_iteration
            # While not converged, continue to run the main loop.
            if main_loop(
                arguments.states, network, true_state, random_instance,
                entropy_data, error_data
            ):
                for a, agent in enumerate(network.nodes):
                    loss = results.loss(agent_type, agent.belief, true_state)
                    loss_values[a] = loss
                    if iteration == iteration_limit:
                        steady_state_results[test][a] = loss

                loss_results[iteration][test] = [
                    np.average(loss_values),
                    np.std(loss_values),
                    np.min(loss_values),
                    np.max(loss_values)
                ]
                # print(loss_results[iteration][test][0])

            # If the simulation has converged, end the test.
            else:
                for a, agent in enumerate(network.nodes):
                    loss = results.loss(agent_type, agent.belief, true_state)
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

        # print(np.average(steady_state_results[test]))

        # Reset the static identity for the Agent class.
        agent_type.identity = 0

    # Recording of results. First, add parameters in sequence.

    file_name_params.append("{}s".format(arguments.states))
    file_name_params.append("{}a".format(arguments.agents))

    if graph_type == "ER":
        if arguments.connectivity is not None:
            file_name_params.append("{:.2f}con".format(arguments.connectivity))
    elif graph_type == "WS":
        if arguments.connectivity is not None and arguments.knn is not None:
            file_name_params.append("{}k".format(arguments.knn))
            file_name_params.append("{:.2f}con".format(arguments.connectivity))
    elif graph_type == "BA":
        if arguments.m is not None:
            file_name_params.append("BA_{}m".format(arguments.m))
    elif graph_type in specialist_graphs + clique_graphs:
        file_name_params.append("{}".format(graph_type))
        if graph_type in clique_graphs:
            file_name_params.append("{}".format(clique_size))

    file_name_params.append("{:.3f}er".format(evidence_rate))
    if noise_value is not None:
        file_name_params.append("{:.2f}nv".format(noise_value))
    if update_type == "Symmetric":
        if fusion_rate is not None:
            file_name_params.append("{}fr".format(fusion_rate))
    elif update_type == "Asymmetric":
        if fusion_prob is not None:
            file_name_params.append("{:.3f}fp".format(fusion_prob))

    if evidence_only:
        file_name_params.append("eo")

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

    # "standard" | "evidence" | "noise" | "en" | "ce" | "cen" | "kce"
    test_set = "en"

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

    elif test_set == "me":

        for m in m_values:
            m_value = m

            for er in evidence_rates:
                evidence_rate = er
                main()