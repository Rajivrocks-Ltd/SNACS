from collections import deque, defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Exercise: 1.6 - Dependency Graph
def is_dependent(input_graph, a, b):
    """
    Determine if there is a path from function a to function b in the function call graph.

    :param input_graph: A directed graph represented as an adjacency list (dict of sets).
    :param a: The starting function (node).
    :param b: The target function (node).
    :return: True if a unit test for function a depends on function b, False otherwise.
    """

    # Initialize a set to keep track of visited nodes to avoid cycles.
    visited = set()

    # Define a recursive depth-first search (DFS) function.
    def dfs(node):
        # If we reach node b, return True.
        if node == b:
            return True

        # Mark the current node as visited.
        visited.add(node)

        # Explore all neighbors of the current node.
        for neighbor in input_graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):  # If any path leads to b, return True.
                    return True

        # If no path to b is found, return False.
        return False

    # Start the DFS from node a.
    return dfs(a)


# Exercise: 1.7 - Count Open Wedges
def count_open_wedges(input_graph):
    """
    Count the number of open wedges in the undirected graph.

    :param graph: The undirected graph represented as an adjacency list (dict of sets).
    :return: The total number of open wedges in the graph.
    """
    open_wedges = 0

    # Iterate over all nodes in the graph.
    for v in input_graph:
        neighbors = sorted(input_graph[v])  # Get the neighbors of node v.

        # Iterate over all pairs of neighbors of v.
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u = neighbors[i]
                w = neighbors[j]

                # Check if there is no edge between u and w.
                if w not in input_graph[u]:
                    # If there is no edge, we have an open wedge (u, v, w).
                    open_wedges += 1

    return open_wedges


# Exercise: 1.8 - Count Friendship Paradox
def count_friendship_paradox(input_graph):
    """
    Count the number of nodes for which the friendship paradox holds.

    :param graph: The undirected graph represented as an adjacency list (dict of sets).
    :return: The number of nodes satisfying the friendship paradox.
    """

    paradox_count = 0

    # Iterate over all nodes in the graph.
    for v in input_graph:
        # Get the degree of node v.
        degree_v = len(input_graph[v])

        # Skip nodes with no neighbors (degree 0), as the paradox doesn't apply.
        if degree_v == 0:
            continue

        # Compute the average degree of neighbors of v.
        neighbor_degrees_sum = 0
        for neighbor in input_graph[v]:
            neighbor_degrees_sum += len(input_graph[neighbor])

        avg_neighbor_degree = neighbor_degrees_sum / degree_v

        # Check if the friendship paradox holds.
        if degree_v < avg_neighbor_degree:
            paradox_count += 1

    return paradox_count


def compute_diameter(graph):
    """
    Compute the diameter of the graph using the concept of k-neighborhoods.

    :param graph: The undirected graph represented as an adjacency list (dict of sets).
    :return: The diameter of the graph.
    """

    def bfs_distance(v):
        """
        Perform a BFS from node v to compute the maximum distance (eccentricity) of v.
        """
        # Initialize a queue for BFS and a distance dictionary.
        queue = deque([v])
        distances = {v: 0}

        while queue:
            node = queue.popleft()
            current_distance = distances[node]

            # Explore all neighbors of the current node.
            for neighbor in graph[node]:
                if neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        # Return the maximum distance from node v.
        return max(distances.values())

    # Compute the eccentricity for each node and track the maximum eccentricity (diameter).
    diameter = 0
    for v in graph:
        eccentricity_v = bfs_distance(v)
        diameter = max(diameter, eccentricity_v)

    return diameter


def count_directed_links(file):
    """
    Count the number of valid directed links in a file. A valid link is a line containing
    exactly two numbers separated by a tab.

    :param file: A .tsv file containing the network data.
    :return int: The number of valid directed links in the file.
    """
    valid_links = 0

    with open(file, 'r') as f:
        for line in f:
            # Strip any extra whitespace and seperate on tabs
            parts = line.strip().split('\t')

            # Check if the line contains exactly two parts
            if len(parts) == 2:
                userA, userB = parts

                # Ensure both parts are numbers
                if userA.isdigit() and userB.isdigit():
                    valid_links += 1

    return valid_links

def user_network_count(file):
    """
    Count the number of unique users in the network.

    :param file: A .tsv file containing the network data.
    :return int: The number of unique users in the network.
    """
    users = set()

    # Open the file and read each line
    with open(file, 'r') as f:
        for link in f:
            line = link.strip().split('\t')

            # Check if the line contains exactly two parts
            if len(line) == 2:
                userA, userB = line
                users.add(userA)
                users.add(userB)

            # Check if line only has one part to it, and that part is a digit add it to the set
            elif line[0].isdigit():
                users.add(line[0])

    return len(users)

def calculate_degrees_per_node(file):
    """
    Calculate the indegree and outdegree for each node in the network.

    :param file: The path to the TSV file.
    :return: Two dictionaries: indegree and outdegree, mapping nodes to their degree values.
    """
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')

            if len(parts) == 2:
                userA, userB = parts

                if userA.isdigit() and userB.isdigit():
                    out_degree[userA] += 1
                    in_degree[userB] += 1

    return in_degree, out_degree

def plot_distribution(degree_dict, degree_type, network_size):
    """
    Plot the degree distribution using a bar plot #ToDo (change to histogram maybe?)

    :param degree_dict: A dictionary mapping nodes to their degree values.
    :param degree_type: A string indicating the type of degree ('Indegree' or 'Outdegree').
    """
    # Get the degree distribution (frequency of each degree)
    degree_count = defaultdict(int)

    # Count the number of nodes for each degree
    for node, degree in degree_dict.items():
        degree_count[degree] += 1

    # Sort the distribution by degree value
    degrees = sorted(degree_count.keys())
    counts = [degree_count[deg] for deg in degrees]

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, counts, color='blue')
    plt.title(f"{degree_type} Distribution - Network Size: {network_size}")
    plt.xlabel(f"{degree_type}")
    plt.ylabel("Number of Nodes")
    plt.yscale('log')  # Log scale for better visualisation purposes!
    plt.xscale('log')  # Log scale for better visualisation purposes!
    plt.show()


def analyze_components(file, network_size):
    """
    Analyze weakly and strongly connected components of the graph.

    :param file: The path to the TSV file.
    :return: Dictionary with component analysis.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Read the file and add edges to the graph
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(parts[0], parts[1])

    # 1. Number of weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(G))
    num_wcc = len(weakly_connected_components)

    # 2. Number of strongly connected components
    strongly_connected_components = list(nx.strongly_connected_components(G))
    num_scc = len(strongly_connected_components)

    # 3. Largest weakly connected component (by number of nodes and links)
    largest_wcc = max(weakly_connected_components, key=len)
    subgraph_wcc = G.subgraph(largest_wcc)  # Extract subgraph of largest WCC
    wcc_size_nodes = subgraph_wcc.number_of_nodes()
    wcc_size_edges = subgraph_wcc.number_of_edges()

    # 4. Largest strongly connected component (by number of nodes and links)
    largest_scc = max(strongly_connected_components, key=len)
    subgraph_scc = G.subgraph(largest_scc)  # Extract subgraph of largest SCC
    scc_size_nodes = subgraph_scc.number_of_nodes()
    scc_size_edges = subgraph_scc.number_of_edges()

    # Return the analysis results
    return {
        f"{network_size} - Number of WCC": num_wcc,
        f"{network_size} - Number of SCC": num_scc,
        f"{network_size} - Largest WCC (nodes)": wcc_size_nodes,
        f"{network_size} - Largest WCC (edges)": wcc_size_edges,
        f"{network_size} - Largest SCC (nodes)": scc_size_nodes,
        f"{network_size} - Largest SCC (edges)": scc_size_edges,
    }


def calculate_average_clustering(file):
    """
    Calculate the average clustering coefficient for a directed graph.

    :param file: The path to the TSV file.
    :return: The average clustering coefficient of the graph.
    """
    G = nx.DiGraph()

    # Read the file and add edges to the graph
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(parts[0], parts[1])

    # Compute the average clustering coefficient (directed)
    avg_clustering = nx.average_clustering(G, count_zeros=True)

    return avg_clustering


def compute_distance_distribution(file, num_samples=1000):
    """
    Compute the distance distribution for the largest weakly connected component (WCC)
    by sampling shortest paths from a subset of nodes using numpy's random sampling.

    :param file: The path to the TSV file.
    :param num_samples: Number of nodes to sample for BFS (default: 100).
    :return: A dictionary with distances as keys and their frequency as values.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Read the file and add edges to the graph
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(parts[0], parts[1])

        # Find the largest weakly connected component (WCC)
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        subgraph_wcc = G.subgraph(largest_wcc)  # Extract the subgraph of the largest WCC

        # Sample a subset of nodes from the largest WCC for BFS
        sampled_nodes = np.random.choice(list(subgraph_wcc.nodes), size=min(num_samples, len(subgraph_wcc)),
                                         replace=False)

        # Create a dictionary to store the distance distribution
        distance_distribution = defaultdict(int)

        # Perform BFS from the sampled nodes and count the distances
        for node in sampled_nodes:
            lengths = nx.single_source_shortest_path_length(subgraph_wcc, node)
            for target, distance in lengths.items():
                if node != target:  # Ignore self-loops (distance 0)
                    distance_distribution[distance] += 1

        return distance_distribution


def plot_distance_distribution(distance_distribution, network_size):
    distances = sorted(distance_distribution.keys())
    frequencies = [distance_distribution[dist] for dist in distances]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(distances, frequencies, color='blue', alpha=0.7)
    plt.title(f"Distance Distribution in the Largest WCC - Network Size: {network_size}")
    plt.xlabel("Shortest Path Distance")
    plt.ylabel("Number of Node Pairs")
    plt.yscale('log') # Log scale for better visualisation purposes!
    plt.grid(True, axis='y')
    plt.show()


def compute_average_distance_sampled(file, num_samples=100):
    """
    Compute the average distance for the largest weakly connected component (WCC)
    by sampling shortest paths from a subset of nodes.

    :param file: The path to the TSV file.
    :param num_samples: Number of nodes to sample for BFS (default: 100).
    :return: The average shortest path distance in the largest WCC.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Read the file and add edges to the graph
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                G.add_edge(parts[0], parts[1])

    # Find the largest weakly connected component (WCC)
    largest_wcc = max(nx.weakly_connected_components(G), key=len)
    subgraph_wcc = G.subgraph(largest_wcc)  # Extract the subgraph of the largest WCC

    # Sample a subset of nodes from the largest WCC for BFS
    sampled_nodes = np.random.choice(list(subgraph_wcc.nodes), size=min(num_samples, len(subgraph_wcc)), replace=False)

    # List to store all shortest path lengths
    all_distances = []

    # Perform BFS from the sampled nodes and store the distances
    for node in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(subgraph_wcc, node)
        for target, distance in lengths.items():
            if node != target:  # Ignore self-loops (distance 0)
                all_distances.append(distance)

    # Compute and return the average distance
    average_distance = np.mean(all_distances)

    return average_distance

graph_1_9 = {
    'a': {'b', 'c'},
    'b': {'a', 'c', 'd'},
    'c': {'a', 'b'},
    'd': {'b', 'e'},
    'e': {'d'}
}

graph_1_8 = {
    'a': {'b', 'c'},
    'b': {'a', 'c', 'd'},
    'c': {'a', 'b'},
    'd': {'b'}
}

graph_1_7 = {
    'a': {'b', 'c'},
    'b': {'a', 'c', 'd'},
    'c': {'a', 'b', 'e'},
    'd': {'b'}
}

graph_1_6 = {
    'a': {'c', 'd'},
    'b': {'e'},
    'c': {'f'},
    'd': {},
    'e': {'f'},
    'f': {}
}

if __name__ == '__main__':
    # Exercise 1
    # print(f"Exercise: 1.6 - {is_dependent(graph_1_6,'a', 'e')}")  # Outputs: True, because a -> c -> f.
    # print(f"Exercise: 1.7 - {count_open_wedges(graph_1_7)}")  # Outputs: 2, because (a, c, f) and (a, d, f) are open wedges.
    # print(f"Exercise: 1.8 - {count_friendship_paradox(graph_1_8)}")  # Outputs: 2, because the paradox holds for nodes 'a' and 'b'.
    # print(f"Exercise: 1.9 - {compute_diameter(graph_1_9)}")  # Outputs: 3, because the diameter of the graph is 3.
    #
    # # Exercise 2
    # print(f'Exercise: 2.1 - Medium = {count_directed_links("data/medium.tsv")}. Large = {count_directed_links("data/large.tsv")}')
    # print(f'Exercise 2.2 - Medium = {user_network_count("data/medium.tsv")}. Large = {user_network_count("data/large.tsv")}')

    # Exercise 2.3
    # in_degree, out_degree = calculate_degrees_per_node("data/medium.tsv")
    # plot_distribution(in_degree, "In-Degree", "Medium")
    # plot_distribution(out_degree, "Out-Degree", "Medium")
    #
    # in_degree, out_degree = calculate_degrees_per_node("data/large.tsv")
    # plot_distribution(in_degree, "In-Degree", "Large")
    # plot_distribution(out_degree, "Out-Degree", "large")

    # Exercise 2.4
    # results = analyze_components("data/medium.tsv", 'Medium')
    # # Display the results
    # for key, value in results.items():
    #     print(f"{key}: {value}")
    #
    # results = analyze_components("data/large.tsv", 'Large')
    # # Display the results
    # for key, value in results.items():
    #     print(f"{key}: {value}")

    # Exercise 2.5
    # avg_clustering_medium = calculate_average_clustering("data/medium.tsv")
    # avg_clustering_large = calculate_average_clustering("data/large.tsv")
    # print(f"Exercise 2.5 - Average Clustering Coefficient (Medium): {avg_clustering_medium}")
    # print(f"Exercise 2.5 - Average Clustering Coefficient (Large): {avg_clustering_large}")

    # Exercise 2.6
    # distance_distribution_medium = compute_distance_distribution("data/medium.tsv")
    # plot_distance_distribution(distance_distribution_medium, "Medium")
    # #
    # distance_distribution_large = compute_distance_distribution("data/large.tsv")
    # plot_distance_distribution(distance_distribution_large, "Large")

    # Exercise 2.7
    avg_distance_medium = compute_average_distance_sampled("data/medium.tsv")
    avg_distance_large = compute_average_distance_sampled("data/large.tsv")

    print(f"Exercise 2.7 - Average Distance (Medium): {avg_distance_medium}")
    print(f"Exercise 2.7 - Average Distance (Large): {avg_distance_large}")
