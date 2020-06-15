import lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns; sns.set(font_scale=1.3)

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [100]
agents_set = [10, 100]
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.2f}".format(x) for x in evidence_rates]
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
graph_types = [
    "ring", "line", "star",
    "star", "connected_star_10", "complete_star_10",
    "caveman_10", "complete_caveman_10"
]


result_directory = "../../results/test_results/sotw-network-temp/probabilisticagent/"

for a, agents in enumerate(agents_set):
    for s, states in enumerate(states_set):
        for g, graph in enumerate(graph_types):

            heatmap_results = np.array([[0.0 for x in noise_values] for y in evidence_rates])

            skip = True

            for e, er in enumerate(reversed(evidence_rates)):
                for n, noise in enumerate(noise_values):

                    file_name_parts = [
                        "steady_state_loss",
                        "{}s".format(states),
                        "{}a".format(agents),
                        "{}".format(graph),
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

                        heatmap_results[e][n] = np.average([np.average(x) for x in data])

                        skip = False

                    except FileNotFoundError:
                        print("MISSING: " + file_name)
                        # Add obvious missing entry into final results array here
                        heatmap_results[e][n] = 1.0

            if skip:
                continue

            print(heatmap_results)
            # cmap = sns.cm.rocket_r
            cmap = sns.cubehelix_palette(10, start=0.5, rot=-.75)
            ax = sns.heatmap(
                heatmap_results,
                # center=0,
                cmap=cmap,
                cbar=False,
                cbar_kws={"shrink": .75, "orientation": "horizontal"},
                xticklabels=noise_values,
                yticklabels=list(reversed(evidence_strings)),
                vmin=0, vmax=0.5,
                linewidths=.5,
                # annot=True,
                # annot_kws={"size": 8},
                # fmt=".2f",
                square=True
            )

            # plt.title("Average loss | {} states, {} agents, {} noise".format(agents, states, noise))

            # import pylab
            # fig_legend = pylab.figure(figsize=(1,2))
            # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left")
            # fig_legend.show()
            # plt.show()

            ax.set(xlabel=r'Noise $\epsilon$', ylabel='Evidence rate')
            # plt.show()
            plt.tight_layout()

            plt.savefig("{}hm_loss_{}_{}_states_{}_agents_noise_er.pdf".format(result_directory, graph.lower(), states, agents), bbox_inches="tight")

            # ax.remove()
            # plt.show()
            # import time
            # time.sleep(10)

            plt.clf()