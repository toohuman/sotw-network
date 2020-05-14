import random

class Topologies:
    """ A collection of functions to generate specific network topologies. """

    def line(self, agents: int, clique_size: int, random_instance):
        return [(x, x + 1) for x in range(agents - 1)]


    def ring(self, agents: int, clique_size: int, random_instance):
        return [(x, x + 1) for x in range(agents - 1)] + [(agents - 1, 0)]


    def star(self, agents: int, clique_size: int, random_instance):
        hub = random_instance.choice(range(agents))
        return [(hub, x) for x in range(agents) if x != hub]


    def connected_star(self, agents: int, clique_size: int, random_instance):
        """
        We place star networks around a ring where each star is
        connected to the ring via the central hub.
        """

        if agents % clique_size != 0:
            sys.exit("Number of agents is not divisible by {}: required for Connected Star network.".format(clique_size))

        edges = []
        agent_indices = [x for x in range(agents)]
        random_instance.shuffle(agent_indices)

        hubs = [agent_indices.pop() for x in range(int(agents / clique_size))]

        # First, connect the hubs together in a ring.
        edges += [(hubs[x], hubs[x + 1]) for x in range(len(hubs) - 1)]
        edges += [(hubs[-1], hubs[0])]

        # Then, create the star graphs attached to each hub node.
        pick = int(len(agent_indices) / len(hubs))
        for hub in hubs:
            chosen_nodes = [agent_indices.pop() for x in range(pick)]
            edges += [(hub, x) for x in chosen_nodes if x != hub]

        return edges


    def complete_star(self, agents: int, clique_size: int, random_instance):
        """
        We place star networks around a ring where each hub is a node on the ring.
        We then connect each hub in the ring to every other hub, forming a complete
        graph for which each node in the complete graph is the hub of a miniature
        star network.
        """

        if agents % clique_size != 0:
            sys.exit("Number of agents is not divisible by {}: required for Connected Star network.".format(clique_size))

        edges = []
        agent_indices = [x for x in range(agents)]
        random_instance.shuffle(agent_indices)

        hubs = [agent_indices.pop() for x in range(int(agents / clique_size))]

        # First, connect the hubs together in a totally connected network.
        edges += [(x, y) for i, x in enumerate(hubs[:-1]) for y in hubs[i+1:]]

        # Then, create the star graphs attached to each hub node.
        pick = int(len(agent_indices) / len(hubs))
        for hub in hubs:
            chosen_nodes = [agent_indices.pop() for x in range(pick)]
            edges += [(hub, x) for x in chosen_nodes if x != hub]

        return edges


    def caveman(self, agents: int, clique_size: int, random_instance):
        """
        We place small complete graphs around a ring where each complete graph is
        connected to the ring via a single node.
        """

        if agents % clique_size != 0:
            sys.exit("Number of agents is not divisible by {}: required for Caveman network.".format(clique_size))

        edges = []
        agent_indices = [x for x in range(agents)]
        random_instance.shuffle(agent_indices)

        hubs = [agent_indices.pop() for x in range(int(agents / clique_size))]

        # First, connect the hubs together in a ring.
        edges += [(hubs[x], hubs[x + 1]) for x in range(len(hubs) - 1)]
        edges += [(hubs[-1], hubs[0])]

        # Then, create the complete graphs attached to each hub node.
        pick = int(len(agent_indices) / len(hubs))
        for hub in hubs:
            chosen_nodes = [hub]
            chosen_nodes += [agent_indices.pop() for x in range(pick)]
            edges += [(x, y) for i, x in enumerate(chosen_nodes[:-1]) for y in chosen_nodes[i+1:]]

        return edges


    def complete_caveman(self, agents: int, clique_size: int, random_instance):
        """
        We place small complete graphs around a ring where each complete graph is
        connected to the ring via a single node. Each node on the main ring is then
        connected to every other node on the ring, forming a complete graph at the
        core of the network.
        """

        if agents % clique_size != 0:
            sys.exit("Number of agents is not divisible by {}: required for Caveman network.".format(clique_size))

        edges = []
        agent_indices = [x for x in range(agents)]
        random_instance.shuffle(agent_indices)

        hubs = [agent_indices.pop() for x in range(int(agents / clique_size))]

        # First, connect the hubs together in a totally connected network.
        edges += [(x, y) for i, x in enumerate(hubs[:-1]) for y in hubs[i+1:]]

        # Then, create the complete graphs attached to each hub node.
        pick = int(len(agent_indices) / len(hubs))
        for hub in hubs:
            chosen_nodes = [hub]
            chosen_nodes += [agent_indices.pop() for x in range(pick)]
            edges += [(x, y) for i, x in enumerate(chosen_nodes[:-1]) for y in chosen_nodes[i+1:]]

        return edges