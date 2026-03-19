import os

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def plot_algorithm_comparison(bfs_nodes, astar_nodes, bfs_time, astar_time):
    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib is not installed. Skipping graph generation.")
        print("To generate graphs, install it via: pip install matplotlib")
        return
        
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Nodes bar chart
    algs = ['BFS', 'A*']
    nodes = [bfs_nodes, astar_nodes]
    ax1.bar(algs, nodes, color=['#e11d48', '#059669'])
    ax1.set_title('Nodes Explored')
    ax1.set_ylabel('Nodes count')
    
    # Time bar chart
    times = [bfs_time, astar_time]
    ax2.bar(algs, times, color=['#e11d48', '#059669'])
    ax2.set_title('Time Taken (ms)')
    ax2.set_ylabel('Milliseconds')
    
    plt.tight_layout()
    plt.savefig('results/graphs.png')
    print("Graphs successfully generated and saved to results/graphs.png")
