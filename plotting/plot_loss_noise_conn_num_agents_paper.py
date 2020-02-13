import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns; sns.set(font_scale=1)

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [100]
agents_set = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.3f}".format(x) for x in evidence_rates]
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_strings = ["{:.2f}".format(x) for x in connectivity_values]

result_directory = "../../results/test_results/sotw-network/"

for s, states in enumerate(states_set):
    for e, er in enumerate(reversed(evidence_rates)):
        for n, noise in enumerate(noise_values):

            loss_results = np.array([[0.0 for x in agents_set] for y in connectivity_values])

            skip = True

            for c, con in enumerate(connectivity_values):

                for a, agents in enumerate(agents_set):
                    file_name_parts = [
                        "steady_state_loss",
                        states, "states",
                        agents, "nodes",
                        "{}".format(con), "con",
                        "{:.3f}".format(er), "er",
                        "{:.3f}".format(noise), "nv"
                    ]
                    file_ext = ".csv"
                    file_name = "_".join(map(lambda x: str(x), file_name_parts)) + file_ext
                    # steady_state_loss_10_states_100_nodes_0.1_con_1.000_er_0.500_nv

                    file_contents = list()

                    try:
                        with open(result_directory + file_name, "r") as file:
                            for line in file:
                                file_contents.append([float(x) for x in line.strip().split(",")])

                    except FileNotFoundError:
                        print("MISSING: " + file_name)


                    loss_results[c][a] = np.average( [np.average(file_contents[x]) for x in range(len(file_contents))])

                    skip = False

                if skip:
                    continue

            print(loss_results)
            cmap = sns.cm.rocket
            for c, con in enumerate(connectivity_values):
                ax = sns.lineplot(agents_set, loss_results[c], linewidth = 2, label=connectivity_strings[c])
            plt.axhline(noise, color="red", linestyle="dotted", linewidth = 2)
            plt.xlabel("Agents")
            plt.ylabel("Average Error")
            if noise == 0:
                plt.ylim(-0.01, 0.01)
            else:
                plt.ylim(-0.01, noise + (noise * 0.1))
            # plt.title("Average loss | {} states, {} er, {} noise".format(states, er, noise))
            # plt.legend()
            # plt.show()
            plt.savefig("../../results/graphs/sotw-network/loss_{}_states_{:.2f}_er_{:.2f}_noise.pdf".format(states, er, noise, con))
            plt.clf()