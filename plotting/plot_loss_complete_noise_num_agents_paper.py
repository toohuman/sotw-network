import lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns; sns.set(font_scale=1.3)

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [100]
agents_set = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.2f}".format(x) for x in evidence_rates]
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
connectivity_value = 1.0
connectivity_string = "{:.2f}".format(connectivity_value)

result_directory = "../../results/test_results/sotw-network/"

for s, states in enumerate(states_set):
    for n, noise in enumerate(noise_values):

        results = np.array([[0.0 for x in agents_set] for y in evidence_rates])
        data = None

        skip = True

        for e, er in enumerate(evidence_rates):
            for a, agents in enumerate(agents_set):
                file_name_parts = [
                    "steady_state_loss",
                    "{}s".format(states),
                    "{}a".format(agents),
                    "{:.2f}con".format(connectivity_value),
                    "{:.2f}er".format(er),
                    "{:.2f}nv".format(noise)
                ]
                file_ext = ".pkl.xz"
                file_name = "_".join(map(lambda x: str(x), file_name_parts)) + file_ext

                try:
                    with lzma.open(result_directory + file_name, "rb") as file:
                        data = pickle.load(file)

                except FileNotFoundError:
                    print("MISSING: " + file_name)

                results[e][a] = np.average([np.average(x) for x in data])

                skip = False

            if skip:
                continue

        print(results)
        # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        # sns.set_palette(sns.color_palette(flatui))
        sns.set_palette("rocket", 8)
        for e, er in enumerate(evidence_rates):
            ax = sns.lineplot(agents_set, results[e], linewidth = 2, label=evidence_strings[e])
        plt.axhline(noise, color="red", linestyle="dotted", linewidth = 2)
        plt.xlabel("Agents")
        plt.ylabel("Average Error")
        if noise == 0:
            plt.ylim(-0.2, 0.2)
        else:
            plt.ylim(-0.01, noise + (noise * 0.1))
        # plt.title("Average loss | {} states, {} er, {} noise".format(states, er, noise))

        ax.get_legend().remove()

        # import pylab
        # fig_legend = pylab.figure(figsize=(1,2))
        # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(evidence_strings))
        # fig_legend.show()
        # plt.show()

        # import time
        # time.sleep(10)

        plt.tight_layout()
        plt.savefig("../../results/graphs/sotw-network/loss_complete_{}_states_{:.2f}_noise.pdf".format(states, noise))
        plt.clf()