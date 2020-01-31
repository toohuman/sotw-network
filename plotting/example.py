def run_AAMAS_load_data(plot_suffix):
    root_folder = '/home/aetek/Documents/TBPhase/Code/MARL/Results/comparison_results/AAMAS/'
    # plot_suffix = 'Compare_A1'
    pickle_file = root_folder + plot_suffix + '.pkl'
    pickle_data = pickle.load(open(pickle_file, "rb"))
    steps = pickle_data['steps']
    means = pickle_data['means']
    maxs = pickle_data['maxs']
    mins = pickle_data['mins']
    stds = pickle_data['stds']
    labels = pickle_data['labels']
    total_reward = pickle_data['total_reward']
    cum_rewards = pickle_data['cum_rewards']
    scores = pickle_data['scores']
    all_scores = pickle_data['all_scores']
    noTests = len(labels)
    maxscores = [[np.max(score) for score in tests] for tests in scores]
    fig = plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 14})
    clrs = sb.color_palette("husl", 5)
    with sb.axes_style("darkgrid"):
        epochs = list(range(steps))
        for ti in range(noTests):
            plt.fill_between(epochs, means[ti] - stds[ti], means[ti] + stds[ti], alpha=0.2, facecolor=clrs[ti])
        for ti in range(noTests):
            plt.plot(epochs, means[ti], label=labels[ti], c=clrs[ti])
    plt.ylabel('Surveillance Score $\mathcal{V}(H_t)$')
    plt.xlabel('Steps')
    plt.legend(loc='lower center', ncol=3)
    plt.xlim([0, steps])
    plt.tight_layout()
    # ax.set_yscale('log')
    # plt.show()
    plt.savefig(root_folder + plot_suffix + '_means.png')
    plt.clf()
    fig = plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 14})
    clrs = sb.color_palette("husl", 5)
    with sb.axes_style("darkgrid"):
        epochs = list(range(steps))
        # for ti in range(noTests):
        #     plt.fill_between(epochs, maxs[ti], mins[ti], alpha=0.2, facecolor=clrs[ti])
        # for ti in range(noTests):
        #     plt.plot(epochs, means[ti], label=labels[ti], c=clrs[ti])
        for ti in range(noTests):
            plt.plot(epochs, mins[ti], label=labels[ti], c=clrs[ti])
    plt.ylabel('Surveillance Score $\mathcal{V}(H_t)$')
    plt.xlabel('Steps')
    plt.legend(loc='lower center', ncol=3)
    plt.xlim([0, steps])
    plt.tight_layout()
    # ax.set_yscale('log')
    # plt.show()
    plt.savefig(root_folder + plot_suffix + '_min.png')
    sb.set(style="whitegrid", palette="pastel", color_codes=True)
    df = pd.DataFrame(list(map(list, zip(*maxscores))))
    df.transpose()
    df.columns = labels
    fig = plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 14})
    ax = sb.violinplot(data=df, palette=clrs, scale='width')
    plt.ylabel('Surveillance Score $\mathcal{V}^*(H)$')
    plt.xlabel('Policy')
    plt.tight_layout()
    # plt.show()
    plt.savefig(root_folder + plot_suffix + '_max_score_violin.png')