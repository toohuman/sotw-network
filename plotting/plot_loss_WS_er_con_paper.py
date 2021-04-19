import lzma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import seaborn as sns; sns.set(font_scale=1.3)

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [100]
agents_set = [100]
evidence_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.3f}".format(x) for x in evidence_rates]
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
knn_values = [2, 4, 6, 8, 10, 20, 50, 99]
# knn_strings = ["{}".format(x) for x in knn_values]
knn_strings = ["2", "", "", "", "10", "20", "50", "99"]
connectivity_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
# connectivity_strings = ["{:.2f}".format(x) for x in connectivity_values]
connectivity_strings = ["0.0", "", "", "", "0.1", "0.5", "1.0"]

result_directory = "../../results/test_results/sotw-network/"

for s, states in enumerate(states_set):
    for n, noise in enumerate(noise_values):
        for a, agents in enumerate(agents_set):
            for k, knn in enumerate(knn_values):

                results = np.array([[0.0 for x in connectivity_values] for y in evidence_rates])
                lowers = np.array([[0.0 for x in connectivity_values] for y in evidence_rates])
                uppers = np.array([[0.0 for x in connectivity_values] for y in evidence_rates])
                data = None

                for e, er in enumerate(evidence_rates):
                    for c, con in enumerate(connectivity_values):
                        file_name_parts = [
                            "steady_state_loss",
                            "{}s".format(states),
                            "{}a".format(agents),
                            "{}k".format(knn),
                            "{:.2f}con".format(con),
                            "{:.3f}er".format(er),
                            "{:.2f}nv".format(noise)
                        ]
                        if knn == 99:
                            file_name_parts = [
                                "steady_state_loss",
                                "{}s".format(states),
                                "{}a".format(agents),
                                "{:.2f}con".format(1.0),
                                "{:.3f}er".format(er),
                                "{:.2f}nv".format(noise)
                            ]
                        file_ext = ".pkl.xz"
                        file_name = "_".join(map(lambda x: str(x), file_name_parts)) + file_ext

                        # print(file_name)

                        file_contents = list()

                        try:
                            with lzma.open(result_directory + file_name, "rb") as file:
                                data = pickle.load(file)

                        except FileNotFoundError:
                            print("MISSING: " + file_name)
                            continue

                        data = sorted([np.average(x) for x in data])
                        lowers[e][c] = data[PERC_LOWER - 1]
                        uppers[e][c] = data[PERC_UPPER - 1]
                        results[e][c] = np.average(data)
                        # print(lowers[e][k], results[e][k], uppers[e][k])

                print("Average Error: {} states | {} knn | {:.2f} noise".format(states, knn, noise))
                for e, er in enumerate(evidence_rates):
                    print("   [{:.2f} er]:  ".format(er), end="")
                    for c, con in enumerate(connectivity_values):
                        print("[{} con]: {:.3f}".format(con, results[e][c]), end=" ")
                    print("")

                # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
                # sns.set_palette(sns.color_palette(flatui))
                sns.set_palette("flare", len(evidence_rates))
                for e, er in enumerate(evidence_rates):
                    # ax = sns.lineplot(x = knn_values, y = results[e], linewidth = 2, label=evidence_strings[e])
                    ax = sns.lineplot(x = connectivity_values, y = results[e], linewidth = 2)
                    plt.fill_between(connectivity_values, lowers[e], uppers[e], facecolor=sns.color_palette()[e], edgecolor="none", alpha=0.3, antialiased=True)
                plt.axhline(noise, color="red", linestyle="dotted", linewidth = 2)
                plt.xlabel(r'Rewiring probability $p$')
                plt.ylabel("Average error")
                plt.xticks(connectivity_values, connectivity_strings)
                if noise == 0:
                    plt.ylim(-0.2, 0.2)
                else:
                    plt.ylim(-0.01, noise + (noise * 0.1))
                # plt.title("Average loss | {} states, {} er, {} noise".format(states, er, noise))

                # import pylab
                # ax.remove()
                # fig_legend = pylab.figure(figsize=(1,2))
                # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(knn_strings))
                # plt.savefig('er_legend.pdf',bbox_inches='tight')
                # plt.show()

                # import time
                # time.sleep(10)

                plt.tight_layout()
                plt.savefig("../../results/graphs/sotw-network/loss_WS_{}_states_{}_agents_{}_k_{:.2f}_noise_shaded.pdf".format(states, agents, knn, noise))
                plt.clf()