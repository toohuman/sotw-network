import lzma
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pickle
import seaborn as sns; sns.set(font_scale=1.5)

PERC_LOWER = 10
PERC_UPPER = 90

states_set = [100]
agents_set = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
evidence_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
evidence_strings = ["{:.3f}".format(x) for x in evidence_rates]
noise_values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
connectivity_value = 1.0

agent_types = ["Agent", "StochasticAgent"]

result_directory_base = "../../results/test_results/sotw-network-temp/"

for s, states in enumerate(states_set):
    for e, er in enumerate(evidence_rates):
        for n, noise in enumerate(noise_values):

            results = np.array([[0.0 for x in agents_set] for y in agent_types])
            lowers = np.array([[0.0 for x in agents_set] for y in agent_types])
            uppers = np.array([[0.0 for x in agents_set] for y in agent_types])
            data = None

            skip = True

            for t, agent_type in enumerate([x.lower() for x in agent_types]):
                for a, agents in enumerate(agents_set):
                    result_directory = result_directory_base + agent_type + "/"
                    if t == 0:
                        file_name_parts = [
                            "steady_state_loss",
                            "{}s".format(states),
                            "{}a".format(agents),
                            "star",
                            "{:.2f}er".format(er),
                            "{:.2f}nv".format(noise)
                        ]
                    else:
                        file_name_parts = [
                            "steady_state_loss",
                            "{}s".format(states),
                            "{}a".format(agents),
                            "star",
                            "{:.3f}er".format(er),
                            "{:.2f}nv".format(noise)
                        ]
                    file_ext = ".pkl.xz"
                    file_name = "_".join(map(lambda x: str(x), file_name_parts)) + file_ext

                    try:
                        with lzma.open(result_directory + file_name, "rb") as file:
                            data = pickle.load(file)

                    except FileNotFoundError:
                        print("MISSING: {} {}".format(agent_type,file_name))

                    data = sorted([np.average(x) for x in data])
                    lowers[t][a] = data[PERC_LOWER - 1]
                    uppers[t][a] = data[PERC_UPPER - 1]
                    results[t][a] = np.average(data)

                    skip = False

                if skip:
                    continue

            print("Average Error: {} states | {:.3f} er | {:.2f} noise".format(states, er, noise))
            for t, agent_type in enumerate([x.lower() for x in agent_types]):
                print("   [{}]:  ".format(agent_type), end="")
                for a, agents in enumerate(agents_set):
                    print("[{}a]: {:.3f}".format(agents, results[t][a]), end=" ")
                print("")

            for _ in range(2):
                # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
                # sns.set_palette(sns.color_palette(flatui))
                sns.set_palette("rocket", len(evidence_rates))
                for t, agent_type in enumerate([x.lower() for x in agent_types]):
                    ax = sns.lineplot(x=agents_set, y=results[t], linewidth = 2, color=sns.color_palette()[t], label=agent_types[t])
                    plt.fill_between(x=agents_set, y1=lowers[t], y2=uppers[t], facecolor=sns.color_palette()[t], edgecolor="none", alpha=0.3, antialiased=True)
                plt.axhline(noise, color="red", linestyle="dotted", linewidth = 2)
                plt.xlabel("Agents")
                plt.ylabel("Average Error")

                # plt.title("Average loss | {} states, {} er, {} noise".format(states, er, noise))

                # ax.get_legend().remove()

                # import pylab
                # fig_legend = pylab.figure(figsize=(1,2))
                # pylab.figlegend(*ax.get_legend_handles_labels(), loc="upper left", ncol=len(evidence_strings))
                # fig_legend.show()
                # plt.show()

                # import time
                # time.sleep(10)

                plt.tight_layout()
                # Complete graph
                if connectivity_value == 1.0:
                    plt.savefig("../../results/graphs/sotw-network-temp/error_star_{}_states_{:.3f}_er_{:.2f}_noise.pdf".format(states, er, noise))
                # Evidence-only graph
                elif connectivity_value == 0.0:
                    plt.savefig("../../results/graphs/sotw-network-temp/error_star_ev_only_{}_states_{:.3f}_er_{:.2f}_noise.pdf".format(states, er, noise))
                plt.clf()