import lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns; sns.set(font_scale=1.3)

states_set = [100]
agents_set = list(range(10, 101, 10))
graph_types = ["ER", "WS", "Complete", "Star", "Ring", "Line"]
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
knn_values = [2, 4, 6, 8, 10, 20, 50]
line_labels = ["path length", "clustering coefficient"]

result_directory = "../../results/test_results/sotw-network/"

for g, graph_type in enumerate(graph_types):

    network_stats = []

    for a, agents in enumerate(agents_set):

        for c, conn in enumerate(connectivity_values):
            for k, knn in enumerate(knn_values):
                if knn >= agents:
                    continue
                if graph_type in ["ER", "WS"]:
                    print(
                        "{} Agents - {} Graph - {} conn - {} knn      "
                        .format(
                            agents, graph_type, conn, knn
                        ),
                        end="\r"
                    )
                else:
                    print(
                        "{} Agents - {} Graph                         "
                        .format(agents, graph_type), end="\r"
                    )
                file_name_params = []
                file_name_params.append("{}".format(graph_type))
                file_name_params.append("{}a".format(agents))

                if graph_type == "ER" or graph_type == "WS":
                    file_name_params.append("{:.2f}con".format(conn))
                    if graph_type == "WS":
                        file_name_params.append("{}k".format(knn))

                file_name = "network_stats" + '_' + '_'.join(file_name_params)

                # Values stored as: mean - std dev - min - max.
                try:
                    with open(result_directory + file_name + '.pkl', 'rb') as file:
                        temp = pickle.load(file)
                        network_stats.append(temp[1])

                except FileNotFoundError:
                    print("\n MISSSING: ", file_name)
                    continue

                if graph_type != "WS":
                    break
            if graph_type == "Complete":
                break

    if graph_type not in ["ER", "WS"]:

        file_name = "network_stats_{}.pdf".format(graph_type)
        sns.set_palette("rocket", 2)

        print(network_stats[0])

        results = [[network_stats[i][j][0] for i in range(len(agents_set))] for j in range(2)]
        std_dev = [[network_stats[i][j][1] for i in range(len(agents_set))] for j in range(2)]
        print(results)
        print(std_dev)
        print(graph_type)

        for i in range(2):
            ax = sns.lineplot(agents_set, results[i], linewidth = 2, label=line_labels[i])
            plt.fill_between(agents_set, np.subtract(results[i], std_dev[i]), np.add(results[i], std_dev[i]), alpha=.3)
        plt.xlabel("Agents")
        plt.ylabel("")

        # ax.get_legend().remove()

        # import pylab
        # fig_legend = pylab.figure(figsize=(1,2))
        # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(knn_strings))
        # fig_legend.show()
        # plt.show()

        # import time
        # time.sleep(10)

        plt.tight_layout()
        plt.show()
        # plt.savefig("../../results/graphs/sotw-network/loss_WS_{}_states_{}_agents_{:.2f}_er_{:.2f}_noise.pdf".format(states, agents, er, noise))
        plt.clf()

    elif:

        file_name = "network_stats_{}.pdf".format(graph_type)
        sns.set_palette("rocket", 2)

        print(network_stats[0])

        results = [[network_stats[i][j][0] for i in range(len(agents_set))] for j in range(2)]
        std_dev = [[network_stats[i][j][1] for i in range(len(agents_set))] for j in range(2)]
        print(results)
        print(std_dev)
        print(graph_type)

        for i in range(2):
            ax = sns.lineplot(agents_set, results[i], linewidth = 2, label=line_labels[i])
            plt.fill_between(agents_set, np.subtract(results[i], std_dev[i]), np.add(results[i], std_dev[i]), alpha=.3)
        plt.xlabel("Agents")
        plt.ylabel("")

        # ax.get_legend().remove()

        # import pylab
        # fig_legend = pylab.figure(figsize=(1,2))
        # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(knn_strings))
        # fig_legend.show()
        # plt.show()

        # import time
        # time.sleep(10)

        plt.tight_layout()
        plt.show()
        # plt.savefig("../../results/graphs/sotw-network/loss_WS_{}_states_{}_agents_{:.2f}_er_{:.2f}_noise.pdf".format(states, agents, er, noise))
        plt.clf()

    else:

        file_name = "network_stats_{}.pdf".format(graph_type)
        sns.set_palette("rocket", 2)

        print(network_stats[0])

        results = [[network_stats[i][j][0] for i in range(len(agents_set))] for j in range(2)]
        std_dev = [[network_stats[i][j][1] for i in range(len(agents_set))] for j in range(2)]
        print(results)
        print(std_dev)
        print(graph_type)

        for i in range(2):
            ax = sns.lineplot(agents_set, results[i], linewidth = 2, label=line_labels[i])
            plt.fill_between(agents_set, np.subtract(results[i], std_dev[i]), np.add(results[i], std_dev[i]), alpha=.3)
        plt.xlabel("Agents")
        plt.ylabel("")

        # ax.get_legend().remove()

        # import pylab
        # fig_legend = pylab.figure(figsize=(1,2))
        # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(knn_strings))
        # fig_legend.show()
        # plt.show()

        # import time
        # time.sleep(10)

        plt.tight_layout()
        plt.show()
        # plt.savefig("../../results/graphs/sotw-network/loss_WS_{}_states_{}_agents_{:.2f}_er_{:.2f}_noise.pdf".format(states, agents, er, noise))
        plt.clf()