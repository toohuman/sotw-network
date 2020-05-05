import lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns; sns.set(font_scale=1.3)

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [100]
agents_set = [10, 50, 100]
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.2f}".format(x) for x in evidence_rates]
noise_values = [0/100, 5/100, 10/100, 20/100, 30/100, 40/100, 50/100]
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
connectivity_strings = ["{:.2f}".format(x) for x in connectivity_values]

result_directory = "../../results/test_results/sotw-network/"

iterations = [x for x in range(10001)]
conn = 1.0

for s, states in enumerate(states_set):
    for n, noise in enumerate(noise_values):
        for a, agents in enumerate(agents_set):

            results = np.array([[0.0 for x in iterations] for y in evidence_rates])
            lowers = np.array([[0.0 for x in iterations] for y in evidence_rates])
            uppers = np.array([[0.0 for x in iterations] for y in evidence_rates])
            data = None

            for e, er in enumerate(evidence_rates):
                file_name_parts = [
                    "loss",
                    "{}s".format(states),
                    "{}a".format(agents),
                    "{:.2f}con".format(conn),
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

                for i, tests in enumerate(data):
                    sorted_data = sorted([x[0] for x in tests])
                    lowers[e][i] = sorted_data[PERC_LOWER - 1]
                    uppers[e][i] = sorted_data[PERC_UPPER - 1]
                    results[e][i] = np.average([x[0] for x in tests])

            # print(results)
            convergence_times = [0 for x in evidence_rates]
            max_iteration = 0
            iterations_maxed = False
            import math
            for e, er_results in enumerate(results):
                for i, iteration in enumerate(er_results):
                    if math.isclose(iteration, 0):
                        convergence_times[e] = i
                        if i > max_iteration:
                            max_iteration = i
                        break
                    elif i == len(er_results) - 1:
                        convergence_times[e] = -1
                        iterations_maxed = True

            max_iteration += 50 if not iterations_maxed else int(len(iterations)/2)

            print("{} states | {} agents | {:.2f} noise".format(states, agents, noise))
            for e, er in enumerate(evidence_rates):
                print("   [{:.2f} er]: {} t".format(er, convergence_times[e]))


            # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            # sns.set_palette(sns.color_palette(flatui))
            sns.set_palette("rocket", len(evidence_rates))
            for e, er in enumerate(evidence_rates):
                ax = sns.lineplot(iterations, results[e], linewidth = 2, color=sns.color_palette()[e], label=evidence_strings[e])
                plt.fill_between(iterations, lowers[e], uppers[e], facecolor=sns.color_palette()[e], edgecolor="none", alpha=0.3, antialiased=True)
            plt.xlabel(r'Time $t$')
            plt.ylabel("Average Error")
            plt.ylim(-0.01, 0.525)
            plt.xlim(0, 1400)
            # plt.xlim(0, 5000)
            # plt.title("Average loss | {} states, {} er, {} noise".format(states, er, noise))

            ax.get_legend().remove()

            # import pylab
            # fig_legend = pylab.figure(figsize=(1,2))
            # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(connectivity_strings))
            # fig_legend.show()
            # plt.show()

            # import time
            # time.sleep(10)

            plt.tight_layout()
            plt.savefig("../../results/graphs/sotw-network/loss_trajectory_{}_states_{}_agents_{:.2f}_noise.pdf".format(states, agents, noise))
            plt.clf()