import argparse
import networkx as nx
import numpy as np
import pickle
import random

# Results variables
directory = "../results/test_results/sotw-network/"

agents_set = [10, 50, 100]
graph_types = ["ER", "WS", "Complete", "Star", "Ring", "Line"]
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
knn_values = [2, 4, 6, 8, 10, 20, 50]

for a, agents in enumerate(agents_set):
    for g, graph_type in enumerate(graph_types):
        for c, conn in enumerate(connectivity_values):
            for k, knn in enumerate(knn_values):
                if knn >= agents:
                    continue
                file_name_params = []
                file_name_params.append("{}".format(graph_type))
                file_name_params.append("{}a".format(agents))

                network = None

                if graph_type == "ER" or graph_type == "WS":
                    file_name_params.append("{:.2f}con".format(conn))

                if graph_type == "ER":
                    network = nx.gnp_random_graph(agents, conn)
                elif graph_type == "WS":
                    network = nx.watts_strogatz_graph(agents, knn, conn)
                    file_name_params.append("{}k".format(knn))
                elif graph_type == "Complete":
                    network = nx.complete_graph(agents)
                else:
                    network = nx.Graph()
                    edges = list()
                    if graph_type == "Star":
                        hub = random.choice(range(agents))
                        edges += [(hub, x) for x in range(agents) if x != hub]
                    elif graph_type == "Ring":
                        edges += [(x, x+1) for x in range(agents - 1)]
                        edges += [(agents-1, 0)]
                    elif graph_type == "Line":
                        edges += [(x, x+1) for x in range(agents - 1)]

                    network.add_edges_from(edges)

                network_stats = [["path length", "clustering coefficient"]]

                try:
                    path_length = nx.average_shortest_path_length(network)
                except nx.NetworkXError:
                    path_length = 0.0
                clustering_coefficient = nx.average_clustering(network)

                network_stats.append([path_length, clustering_coefficient])

                with open(directory + "network_stats" + '_' + '_'.join(file_name_params) + '.pkl', 'wb') as file:
                    pickle.dump(network_stats, file)

                if graph_type != "WS":
                    break
            if graph_type == "Complete":
                break