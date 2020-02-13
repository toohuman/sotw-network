import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set()

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [10]
# hubs_set = [1, 2, 3, 4, 5, 10, 50]
# nodes_set = [1, 5, 10, 20, 30, 40, 50]
hubs_set = [50]
nodes_set = [1]
noise_values = [0/100, 5/100, 10/100, 20/100, 30/100, 40/100, 50/100]
er = 0.1

result_directory = "../../results/test_results/sotw-network/"

# iterations = [x for x in range(1000)]
iterations = [x for x in range(5000)]

hub_node_combo = list(itertools.product(hubs_set, nodes_set))

for noise in noise_values:

    noise_input_string = ""
    noise_output_string = ""

    if noise is not None:
        noise_input_string += "_{:.3f}_nv".format(noise)
        noise_output_string += "_{}_nv".format(noise)

    hub_loss_results = np.array([[[[0.0 for i in iterations] for z in hubs_set] for y in nodes_set] for x in states_set])
    node_loss_results = np.array([[[[0.0 for i in iterations] for z in hubs_set] for y in nodes_set] for x in states_set])
    combined_loss_results = np.array([[[[0.0 for i in iterations] for z in hubs_set] for y in nodes_set] for x in states_set])
    labels = [["" for x in hub_node_combo] for y in states_set]

    for i, states in enumerate(states_set):
        for j, nodes in enumerate(nodes_set):
            for k, hubs in enumerate(hubs_set):
                file_name_parts = ["loss", nodes * hubs, "nodes", hubs, "hubs", states, "states", "{:.3f}".format(er), "er", "{:.3f}".format(noise), "nv"]
                # hub_loss_10_nodes_1_hubs_10_states_0.100_er_0.200_nv
                file_ext = ".csv"

                labels[i][j] = "{} S : {} H : {} N".format(states, hubs, nodes)

                combined_file = "_".join(map(lambda x: str(x), file_name_parts)) + file_ext
                hub_file = "hub_" + "_".join(map(lambda x: str(x), file_name_parts)) + file_ext
                node_file = "node_" + "_".join(map(lambda x: str(x), file_name_parts)) + file_ext

                average_loss = 0.0

                try:
                    with open(result_directory + combined_file, "r") as file:
                        iteration = 0
                        for line in file:
                            if iteration == len(iterations):
                                break
                            average_loss = np.average([float(x) for x in line.strip().split(",")])

                            combined_loss_results[i][j][k][iteration] = average_loss
                            iteration += 1
                        for l in range(iteration, len(iterations)):
                            combined_loss_results[i][j][k][l] = combined_loss_results[i][j][k][iteration - 1]

                    with open(result_directory + hub_file, "r") as file:
                        iteration = 0
                        for line in file:
                            if iteration == len(iterations):
                                break
                            average_loss = np.average([float(x) for x in line.strip().split(",")])

                            hub_loss_results[i][j][k][iteration] = average_loss
                            iteration += 1
                        for l in range(iteration, len(iterations)):
                            hub_loss_results[i][j][k][l] = hub_loss_results[i][j][k][iteration - 1]

                    with open(result_directory + node_file, "r") as file:
                        iteration = 0
                        for line in file:
                            if iteration == len(iterations):
                                break
                            average_loss = np.average([float(x) for x in line.strip().split(",")])

                            node_loss_results[i][j][k][iteration] = average_loss
                            iteration += 1
                        for l in range(iteration, len(iterations)):
                            node_loss_results[i][j][k][l] = node_loss_results[i][j][k][iteration - 1]

                except FileNotFoundError:
                    # If no file, just skip it.
                    pass

    print(combined_loss_results)
    print(labels)
    cmap = sns.cm.rocket
    c = [cmap(x/len(states_set)) for x in range(0, len(states_set))]
    for i, states in enumerate(states_set):
        for j, nodes in enumerate(nodes_set):
            for k, hubs in enumerate(hubs_set):
                if combined_loss_results[i][j][k][0] == 0:
                    continue
                plt.axhline(noise, color="red", linestyle="dashed")
                ax = plt.plot(iterations, hub_loss_results[i][j][k], linestyle="dashed", linewidth = 2, label="Hub")
                ax = plt.plot(iterations, node_loss_results[i][j][k], linestyle="dashed", linewidth = 2, label="Node")
                ax = plt.plot(iterations, combined_loss_results[i][j][k], linewidth = 2, label="Combined")
                plt.xlabel("Iterations")
                plt.ylabel("Average Loss")
                plt.ylim(-0.01, 0.51)
                plt.title("{} hub(s), {} nodes, {} states".format(hubs, hubs * nodes, states**2))
                plt.legend()
                # plt.show()
                plt.savefig(
                    "../../results/graphs/sotw-network/{}_hubs_{}_nodes_{}_states_{}_er_{}_noise.pdf".format(
                        hubs, hubs * nodes, states, er, noise
                    )
                )
                plt.clf()