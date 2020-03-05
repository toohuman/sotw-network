import lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns; sns.set()

PERC_LOWER = 10
PERC_UPPER = 90

agents_set = [100]
states_set = [100]
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.2f}".format(x) for x in evidence_rates]
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
knn_values = [2, 4, 6, 8, 10, 20, 50]
knn_strings = ["{}".format(x) for x in knn_values]
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_strings = ["{:.2f}".format(x) for x in connectivity_values]

result_directory = "../../results/test_results/sotw-network/"

for a, agents in enumerate(agents_set):
    for s, states in enumerate(states_set):
        for n, noise in enumerate(noise_values):
            for k, knn in enumerate(knn_values):

                heatmap_results = np.array([[0.0 for x in connectivity_values] for y in evidence_rates])

                skip = True

                for e, er in enumerate(reversed(evidence_rates)):
                    for c, con in enumerate(connectivity_values):
                        file_name_parts = [
                            "steady_state_loss",
                            "{}s".format(states),
                            "{}a".format(agents),
                            "{}k".format(knn),
                            "{:.2f}con".format(con),
                            "{:.2f}er".format(er),
                            "{:.2f}nv".format(noise)
                        ]
                        file_ext = ".pkl.xz"
                        file_name = "_".join(map(lambda x: str(x), file_name_parts)) + file_ext

                        steady_state_results = []
                        average_loss = 0.0

                        try:
                            with lzma.open(result_directory + file_name, "rb") as file:
                                data = pickle.load(file)

                            heatmap_results[e][c] = np.average([np.average(x) for x in data])

                            skip = False

                        except FileNotFoundError:
                            print("MISSING: " + file_name)
                            # Add obvious missing entry into final results array here
                            heatmap_results[e][c] = 1.0

                if skip:
                    continue

                print(heatmap_results)
                # cmap = sns.cm.rocket_r
                cmap = sns.cubehelix_palette(10, start=0.5, rot=-.75)
                ax = sns.heatmap(
                    heatmap_results,
                    # center=0,
                    cmap=cmap,
                    cbar_kws={"shrink": .75},
                    xticklabels=connectivity_strings,
                    yticklabels=list(reversed(evidence_strings)),
                    vmin=0, vmax=0.5,
                    linewidths=.5,
                    # annot=True,
                    # annot_kws={"size": 8},
                    # fmt=".2f",
                    square=True
                )
                # legend = plt.figure()
                # fig_ax =

                # plt.title("Average loss | {} states, {} agents, {} noise".format(agents, states, noise))
                ax.set(xlabel=r'Rewiring $p$', ylabel='Evidence rate')
                # plt.show()
                plt.tight_layout()
                plt.savefig("../../results/graphs/sotw-network/hm_loss_{}_states_{}_agents_{:.2f}_noise_{}_k_er_con.pdf".format(states, agents, noise, knn), bbox_inches="tight")
                plt.clf()