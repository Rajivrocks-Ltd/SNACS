import networkx as nx
from collections import defaultdict, Counter
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from scipy.stats import spearmanr
import json
import seaborn as sns
import community as community_louvain

# Exercise 1 - implementing BoundingDiameter algorithm
# =====================================================================================================================

def generate_graph_q1():
    # Create the graph
    G = nx.Graph()

    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U']
    G.add_nodes_from(nodes)

    # Define the edges from Figure 1 of the assignment
    edges = [
        ('A', 'C'), ('B', 'C'), ('B', 'E'), ('C', 'D'), ('C', 'F'), ('D', 'F'), ('E', 'G'), ('E', 'F'), ('F', 'H'),
        ('F', 'L'), ('F', 'J'), ('G', 'J'), ('I', 'J'), ('K', 'L'), ('L', 'M'), ('L', 'P'), ('L', 'N'), ('N', 'P'),
        ('P', 'Q'), ('P', 'R'), ('Q', 'R'), ('Q', 'T'), ('Q', 'S'), ('R', 'T'), ('S', 'U')
    ]
    G.add_edges_from(edges)

    # Calculate the eccentricity of all nodes
    # eccentricity_dict = nx.eccentricity(G)
    return G




# End of Exercise 1
# =====================================================================================================================

# Exercise 2 - museum data
# =====================================================================================================================
def shuffle_excel_rows(input_file_path, output_file_path='shuffled_file.xlsx'):
    """
    Reads an Excel file, shuffles its rows, and saves the shuffled data to a new Excel file.

    Parameters:
        input_file_path (str): The path to the input Excel file.
        output_file_path (str): The path to save the shuffled Excel file. Default is 'shuffled_file.xlsx'.
    """
    # Load the Excel file
    df = pd.read_excel(input_file_path)

    # Shuffle the rows
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    # Save the shuffled DataFrame to a new Excel file
    df_shuffled.to_excel(output_file_path, index=False)

    print(f"Rows have been shuffled and saved to {output_file_path}")

def construct_undirected_graph_from_adjacency_list(file_path):
    # Load the adjacency list data from the Excel file, with the room pairs in one column
    adjacency_data = pd.read_excel(file_path, header=None, names=["Rooms"])

    # Create an undirected graph
    graph = nx.Graph()

    # Add edges to the graph from each line in the file
    for _, row in adjacency_data.iterrows():
        # Split the comma-separated rooms
        room1, room2 = row["Rooms"].split(",")
        room1, room2 = room1.strip(), room2.strip()  # Remove any extra whitespace
        print(f"Adding edge between {room1} and {room2}")
        graph.add_edge(room1, room2)  # Add an undirected edge between Room1 and Room2

    return graph

def find_mst_and_hamiltonian_path(graph):
    # Step 1: Generate the Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(museum_graph)

    # Step 2: Perform a DFS traversal on the MST to approximate a Hamiltonian path
    start_node = list(mst.nodes)[0]  # Choose an arbitrary start node
    hamiltonian_path = list(nx.dfs_preorder_nodes(mst, source=start_node))

    # Display the path
    print("Approximate Hamiltonian Path to visit each room:", hamiltonian_path)


import networkx as nx


def transform_to_dual_graph(original_graph):
    """
    Transforms the original museum graph into a dual graph where each edge (doorway or staircase)
    becomes a node and each transition within a room becomes an edge.

    Parameters:
    - original_graph (nx.Graph): The original graph where nodes represent rooms and edges
      represent doorways/staircases.

    Returns:
    - dual_graph (nx.Graph): The transformed graph where nodes represent doorways/stairs
      and edges represent transitions through rooms.
    """
    # Initialize the dual graph
    dual_graph = nx.Graph()

    # Use a set to track unique edges in the original graph to avoid duplicates
    unique_edges = set()

    # Step 1: Add a node in dual graph for each unique edge in the original graph
    for edge in original_graph.edges():
        # Sort edge nodes to prevent (A, B) and (B, A) duplicates in undirected graph
        sorted_edge = tuple(sorted(edge))
        unique_edges.add(sorted_edge)

    # Add nodes to the dual graph based on unique edges
    for edge in unique_edges:
        dual_graph.add_node(edge)  # Each unique edge in original_graph is a node in dual_graph

    # Step 2: Add edges in dual graph to represent possible transitions within each room
    for room in original_graph.nodes():
        # Get all edges (doorways/stairs) connected to this room
        connected_edges = [tuple(sorted(e)) for e in original_graph.edges(room) if tuple(sorted(e)) in unique_edges]

        # Connect each pair of edges to simulate transitions through the room
        for i in range(len(connected_edges)):
            for j in range(i + 1, len(connected_edges)):
                dual_graph.add_edge(connected_edges[i], connected_edges[j])

    return dual_graph


def find_best_rooms_for_first_aid(graph):
    # Calculate eccentricity for each room
    eccentricity = nx.eccentricity(graph)

    # Find the three rooms with the smallest eccentricity values
    sorted_rooms = sorted(eccentricity, key=eccentricity.get)
    best_rooms = sorted_rooms[:3]  # The three most central rooms

    # Display the results
    for rank, room in enumerate(best_rooms, start=1):
        print(f"{rank}. Room: {room}, Eccentricity: {eccentricity[room]}")

    return best_rooms

# End of Exercise 2
# =====================================================================================================================


# Exercise 3.1 3.2 - Twitter data
# =====================================================================================================================
def construct_mention_graph(file_path):
    mentions = defaultdict(int)
    mention_pattern = re.compile(r'@(\w+)')  # Regex pattern for @mentions

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = line.strip().split('\t')
            if len(fields) != 3:
                continue

            timestamp, user_x, tweet = fields
            user_x = user_x.lower()  # Normalize sender to lowercase

            # Extract mentions and filter out emails
            raw_mentions = mention_pattern.findall(tweet.lower())
            unique_mentions = set(raw_mentions)  # Keep only unique mentions in each tweet

            for user_y in unique_mentions:
                # Check the filtering conditions:
                # - Not a self-mention
                # - Not an email (no '.' in the username)
                # - Username length is between 5 and 15 characters
                if user_y == user_x or '.' in user_y or not (5 <= len(user_y) <= 15):
                    continue

                # Increment the unique mention count
                mentions[(user_x, user_y)] += 1

    # Create the graph with the cleaned mentions data
    mention_graph = nx.DiGraph()
    for (user_x, user_y), weight in mentions.items():
        mention_graph.add_edge(user_x, user_y, weight=weight)

    return mention_graph


def create_clean_tsv(output_file, mention_graph):
    # Step 1: Write to the .tsv file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['user_x', 'user_y', 'weight'])  # Header row

        # Step 2: Write each edge from the mention graph with its weight
        for user_x, user_y, data in mention_graph.edges(data=True):
            writer.writerow([user_x, user_y, data['weight']])

    print(f"TSV file created successfully at {output_file}")


# 3. Method to Calculate Basic Graph Statistics
def calculate_basic_stats(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = nx.density(graph)

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Density of the graph: {density}")

    return num_nodes, num_edges, density


# 4. Method to Analyze Connected Components
def analyze_components(graph):
    strong_components = list(nx.strongly_connected_components(graph))
    weak_components = list(nx.weakly_connected_components(graph))

    num_strong = len(strong_components)
    num_weak = len(weak_components)
    largest_strong = max(len(c) for c in strong_components)
    largest_weak = max(len(c) for c in weak_components)

    print(f"Strongly connected components: {num_strong}")
    print(f"Largest strongly connected component size: {largest_strong}")
    print(f"Weakly connected components: {num_weak}")
    print(f"Largest weakly connected component size: {largest_weak}")

    return num_strong, largest_strong, num_weak, largest_weak


# 5. Method to Calculate Clustering Coefficient
def calculate_clustering_coefficient(graph):
    avg_clustering = nx.average_clustering(graph.to_undirected())
    print(f"Average clustering coefficient: {avg_clustering}")
    return avg_clustering


def compute_degree_distributions(graph):
    # Bulk fetch in-degree and out-degree as dictionaries
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())

    # Convert degree dictionaries to lists for easier handling in plotting
    in_degree_values = list(in_degrees.values())
    out_degree_values = list(out_degrees.values())

    return in_degree_values, out_degree_values

def plot_degree_distributions_optimized(in_degree_values, out_degree_values, network_size: str):
    plt.figure(figsize=(12, 5))

    # In-degree distribution
    plt.subplot(1, 2, 1)
    # Calculate unique values and their counts
    in_degrees, in_counts = np.unique(in_degree_values, return_counts=True)
    plt.scatter(in_degrees, in_counts, color='blue', alpha=0.7)
    plt.title(f"In-degree Distribution - Network size: {network_size}")
    plt.xlabel("In-degree")
    plt.xscale('log')
    plt.ylabel("Frequency")
    plt.yscale('log')

    # Out-degree distribution
    plt.subplot(1, 2, 2)
    # Calculate unique values and their counts
    out_degrees, out_counts = np.unique(out_degree_values, return_counts=True)
    plt.scatter(out_degrees, out_counts, color='green', alpha=0.7)
    plt.title(f"Out-degree Distribution - Network size: {network_size}")
    plt.xlabel("Out-degree")
    plt.xscale('log')
    plt.ylabel("Frequency")
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


def load_graph_from_tsv(tsv_file):
    # Initialize an empty directed graph
    graph = nx.DiGraph()

    # Open and read the .tsv file
    with open(tsv_file, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header row if it exists
        for line in file:
            # Split each line by tab
            user_x, user_y, weight = line.strip().split('\t')
            weight = int(weight)  # Convert weight to integer

            # Add edge with weight to the graph
            graph.add_edge(user_x, user_y, weight=weight)

    return graph


def calculate_giant_component(graph):
    # Find the largest weakly connected component and return its subgraph as undirected
    largest_weak_component = max(nx.weakly_connected_components(graph), key=len)
    giant_component = graph.subgraph(largest_weak_component).to_undirected()
    return giant_component


def approximate_average_distance(graph, num_samples=10000):
    # Calculate the giant component in undirected form
    giant_component = calculate_giant_component(graph)
    nodes = list(giant_component.nodes())

    # Sample random pairs of nodes
    sampled_pairs = [(random.choice(nodes), random.choice(nodes)) for _ in range(num_samples)]

    # Calculate shortest paths for each sampled pair and store the lengths
    path_lengths = []
    for u, v in sampled_pairs:
        if u != v:  # Ensure we aren't sampling the same node as both u and v
            try:
                length = nx.shortest_path_length(giant_component, source=u, target=v)
                path_lengths.append(length)
            except nx.NetworkXNoPath:
                continue  # Skip pairs with no path

    # Use numpy to calculate the mean of path lengths as an approximation of the average distance
    avg_distance = np.mean(path_lengths)
    print(f"Approximate average distance in the giant component: {avg_distance}")

    return path_lengths, avg_distance


def plot_distance_distribution(path_lengths, network_size: str):
    # Plot the distribution of shortest path lengths
    plt.figure(figsize=(10, 6))
    plt.hist(path_lengths, bins=30, color='purple', alpha=0.7)
    plt.title(f"Distance Distribution in the Giant Component (Sampled) - Network size: {network_size}")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

def plot_similarity_matrix(csv_file):
    # Load the similarity matrix from the CSV file
    similarity_matrix = pd.read_csv(csv_file, index_col=0)

    # Plot the similarity matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", cbar=True, square=True, fmt=".3f",
                annot_kws={"size": 12})
    plt.title("Spearman Correlation Matrix of Centrality Rankings")
    plt.show()

# End of Exercise 3.1, 3.2
# =====================================================================================================================

# Exercise 3.3 - Top 20 users based on three different centrality measures
# =====================================================================================================================

def calculate_centrality_measures(graph):
    # 1. Degree Centrality (in-degree for directed graphs)
    degree_centrality = nx.in_degree_centrality(graph)
    print("Degree Centrality Calculated")

    # 2. Closeness Centrality (for directed graphs)
    closeness_centrality = nx.closeness_centrality(graph)
    print("Closeness Centrality Calculated")

    # 3. Betweenness Centrality (for directed graphs)
    betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
    print("Betweenness Centrality Calculated")

    return degree_centrality, closeness_centrality, betweenness_centrality


def get_top_20_users(centrality_dict):
    # Sort by centrality values in descending order and select the top 20
    sorted_users = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    return [user for user, centrality in sorted_users]


def compare_rankings(rankings):
    # Convert rankings to DataFrame for easier comparison
    df = pd.DataFrame(rankings)

    # Calculate Spearman's rank correlation coefficient for each pair of rankings
    similarity_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for col1 in df.columns:
        for col2 in df.columns:
            # Calculate Spearman correlation
            similarity_matrix.loc[col1, col2] = spearmanr(df[col1], df[col2])[0]

    return similarity_matrix


def save_rankings_and_similarity(rankings, similarity_matrix, rankings_file='rankings.json',
                                 similarity_file='similarity_matrix.csv'):
    # Save rankings as JSON
    with open(rankings_file, 'w') as file:
        json.dump(rankings, file)
    print(f"Rankings saved to {rankings_file}")

    # Save similarity matrix as CSV
    similarity_matrix.to_csv(similarity_file)
    print(f"Similarity matrix saved to {similarity_file}")

def implement_question_3_3(graph):
    # Step 1: Calculate centrality measures
    degree_centrality, closeness_centrality, betweenness_centrality = calculate_centrality_measures(graph)

    # Step 2: Extract top 20 users by each centrality measure
    top_20_degree = get_top_20_users(degree_centrality)
    top_20_closeness = get_top_20_users(closeness_centrality)
    top_20_betweenness = get_top_20_users(betweenness_centrality)

    # Combine the rankings for comparison
    rankings = {
        "Degree Centrality": top_20_degree,
        "Closeness Centrality": top_20_closeness,
        "Betweenness Centrality": top_20_betweenness
    }

    # Step 3: Compare rankings using Spearman's rank correlation
    similarity_matrix = compare_rankings(rankings)
    print("Spearman Rank Correlation between Centrality Measures:\n", similarity_matrix)

    # Save results to disk
    save_rankings_and_similarity(rankings, similarity_matrix)

    return rankings, similarity_matrix


# Plot the top 20 users for each centrality measure
def plot_top_20_users_from_ranking(rankings_file):
    # Load the JSON file with pre-calculated rankings
    with open(rankings_file, 'r') as file:
        rankings = json.load(file)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Degree Centrality
    axes[0].bar(rankings["Degree Centrality"], range(20, 0, -1), color='blue')
    axes[0].set_title("Top 20 Users by Degree Centrality")
    axes[0].set_xticklabels(rankings["Degree Centrality"], rotation=90)
    axes[0].set_xlabel("User")
    axes[0].set_ylabel("Ranking Position")

    # Closeness Centrality
    axes[1].bar(rankings["Closeness Centrality"], range(20, 0, -1), color='green')
    axes[1].set_title("Top 20 Users by Closeness Centrality")
    axes[1].set_xticklabels(rankings["Closeness Centrality"], rotation=90)
    axes[1].set_xlabel("User")
    axes[1].set_ylabel("Ranking Position")

    # Betweenness Centrality
    axes[2].bar(rankings["Betweenness Centrality"], range(20, 0, -1), color='purple')
    axes[2].set_title("Top 20 Users by Betweenness Centrality")
    axes[2].set_xticklabels(rankings["Betweenness Centrality"], rotation=90)
    axes[2].set_xlabel("User")
    axes[2].set_ylabel("Ranking Position")

    plt.tight_layout()
    plt.show()

# End of Exercise 3.3
# =====================================================================================================================

# Start of Exercise 3.4
# =====================================================================================================================

def detect_communities(giant_component):
    # Apply the Louvain method for community detection
    partition = community_louvain.best_partition(giant_component)

    # Organize nodes by community
    communities = {}
    for node, community_id in partition.items():
        communities.setdefault(community_id, []).append(node)

    # Display community information
    print(f"Number of communities detected: {len(communities)}")
    i = 0
    for community_id, nodes in communities.items():
        print(f"Community {community_id}: Size {len(nodes)}, iters {i}")
        i += 1

    print("Done with community detection")
    return communities, partition


def create_and_plot_meta_graph(graph, communities, partition):
    # Create the meta-graph
    meta_graph = nx.Graph()

    # Add each community as a node
    for comm_id, nodes in communities.items():
        meta_graph.add_node(comm_id, size=len(nodes))

    # Add weighted edges between communities based on inter-community connections
    for src, dst in graph.edges():
        src_comm = partition[src]
        dst_comm = partition[dst]
        if src_comm != dst_comm:
            if meta_graph.has_edge(src_comm, dst_comm):
                meta_graph[src_comm][dst_comm]['weight'] += 1
            else:
                meta_graph.add_edge(src_comm, dst_comm, weight=1)

    # Draw the meta-graph with node sizes proportional to community sizes
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(meta_graph, k=0.5, seed=42)  # Positioning for better visualization
    sizes = [meta_graph.nodes[comm]['size'] * 10 for comm in meta_graph.nodes()]
    nx.draw(meta_graph, pos, node_size=sizes, with_labels=True, font_size=8, edge_color='gray')
    plt.title("Meta-Graph of Communities")
    plt.show()

# End of Exercise 3.4
# =====================================================================================================================

# start of Exercise 3.5
# =====================================================================================================================

def plot_weight_distribution_scatter(graph):
    # Extract weights of all edges
    edge_weights = [data['weight'] for _, _, data in graph.edges(data=True)]

    # Calculate the frequency of each unique edge weight
    weight_counts = Counter(edge_weights)
    weights = list(weight_counts.keys())
    frequencies = list(weight_counts.values())

    # Plot the distribution of edge weights as a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(weights, frequencies, color='blue', alpha=0.7)
    plt.title("Edge Weight Distribution in the Mention Graph")
    plt.xlabel("Edge Weight (Number of Mentions)")
    plt.ylabel("Frequency")
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()

# End of Exercise 3.5
# =====================================================================================================================

if __name__ == "__main__":
    # Exercise 1
    # ...
    # shuffle excel file rows
    # shuffle_excel_rows('data/museum_adjacency_list.xlsx', 'data/museum_adjacency_list_2.xlsx')

    # Exercise 2.2
    # print("Constructing the Rijksmuseum Floor Plan Layout Graph")
    # museum_graph = construct_undirected_graph_from_adjacency_list('data/museum_adjacency_list_2.xlsx')
    # print(f'Number of nodes: {museum_graph.number_of_nodes()}')
    # print(f'Number of edges: {museum_graph.number_of_edges()}')
    # plt.figure(figsize=(12, 12))
    # plt.title("Rijksmuseum Floor Plan Layout Graph")
    # nx.draw(museum_graph, with_labels=True, pos=nx.kamada_kawai_layout(museum_graph))
    # plt.show()

    # Exercise 2.4
    # print("Finding the Minimum Spanning Tree and Approximate Hamiltonian Path")
    # find_mst_and_hamiltonian_path(museum_graph)

    # Exercise 2.5
    # print("Transforming the Museum Graph into a Dual Graph")
    # dual_graph = transform_to_dual_graph(museum_graph)
    # print(f'Number of nodes in the dual graph: {dual_graph.number_of_nodes()}')
    # print(f'Number of edges in the dual graph: {dual_graph.number_of_edges()}')
    # plt.figure(figsize=(12, 12))
    # plt.title("Dual Graph of the Rijksmuseum Floor Plan Layout")
    # nx.draw(dual_graph, with_labels=True, pos=nx.kamada_kawai_layout(dual_graph))
    # plt.show()

    # Exercise 2.6
    # print("Finding the Best Rooms for First Aid Locations")
    # _ = find_best_rooms_for_first_aid(museum_graph)

    # Exercise 3.1
    # twitter_data_small = 'data/twitter-small.tsv'
    # twitter_data_large = 'data/twitter-larger.tsv'

    # mention_graph_small = construct_mention_graph(twitter_data_small)
    # create_clean_tsv('data/twitter-small-cleaned.tsv', mention_graph_small)
    # mention_graph_small = load_graph_from_tsv('data/twitter-small-cleaned.tsv')

    # mention_graph_larger = construct_mention_graph(twitter_data_large)
    # create_clean_tsv('data/twitter-larger-cleaned.tsv', mention_graph_larger)
    # mention_graph_larger = load_graph_from_tsv('data/twitter-larger-cleaned.tsv')

    # # Exercise 3.2
    # print("Basic Stats for Small Graph:")
    # calculate_basic_stats(mention_graph_small)
    # print("\nBasic Stats for Larger Graph:")
    # calculate_basic_stats(mention_graph_larger)
    #
    # print("\nConnected Components for Small Graph:")
    # analyze_components(mention_graph_small)
    # print("\nConnected Components for Larger Graph:")
    # analyze_components(mention_graph_larger)
    #
    # print("\nClustering Coefficient for Small Graph:")
    # calculate_clustering_coefficient(mention_graph_small)
    # print("\nClustering Coefficient for Larger Graph:")
    # calculate_clustering_coefficient(mention_graph_larger)

    # print("\nAverage distance Distributions for Small Graph:")
    # path_lengths, avg_distance = approximate_average_distance(mention_graph_small, num_samples=20000)
    # print(f'Average distance: {avg_distance}')
    # plot_distance_distribution(path_lengths, network_size='Small')
    #
    # print("\nAverage distance Distributions for Larger Graph:")
    # path_lengths, avg_distance = approximate_average_distance(mention_graph_larger, num_samples=100000)
    # print(f'Average distance: {avg_distance}')
    # plot_distance_distribution(path_lengths, network_size='Larger')

    # print("\nDegree Distributions for Small Graph:")
    # in_degree_values, out_degree_values = compute_degree_distributions(mention_graph_small)
    # plot_degree_distributions_optimized(in_degree_values, out_degree_values, 'Small')
    # print("\nDegree Distributions for Larger Graph:")
    # in_degree_values, out_degree_values = compute_degree_distributions(mention_graph_larger)
    # plot_degree_distributions_optimized(in_degree_values, out_degree_values, 'Larger')

    # # Exercise 3.3
    # print("\nTop 20 Users for Small Graph:")
    # rankings, similarity_matrix = implement_question_3_3(mention_graph_small)
    # print(rankings)
    # print("\nTop 20 Users for Larger Graph:")
    # rankings, similarity_matrix = implement_question_3_3(mention_graph_larger)
    # print(rankings)

    # plot_top_20_users_from_ranking('rankings.json')
    # plot_similarity_matrix('similarity_matrix.csv')

    # Exercise 3.4
    # print("\nCommunities for Small Graph:")
    # giant_component = calculate_giant_component(mention_graph_small)
    # communities, partition = detect_communities(giant_component)
    # create_and_plot_meta_graph(giant_component, communities, partition)

    # Exercise 3.5
    # print("\nWeight Distribution for Small Graph:")
    # plot_weight_distribution_scatter(mention_graph_small)


    # mention_graph_large = construct_mention_graph(twitter_data_large)
    # create_clean_tsv('data/twitter-small-cleaned.tsv', mention_graph_small)
    # create_clean_tsv('data/twitter-larger-cleaned.tsv', mention_graph_large)