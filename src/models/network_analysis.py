import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupplyChainNetwork:
    """Class for analyzing supply chain networks using graph theory."""
    
    def __init__(self, directed=True):
        """
        Initialize the supply chain network.
        
        Parameters:
        -----------
        directed : bool
            Whether the supply chain graph should be directed (default: True)
        """
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self.metrics = {}
        self.communities = None
        self.critical_nodes = None
    
    def build_network_from_dataframe(self, df, source_col, target_col, weight_col=None, attrs_cols=None):
        """
        Build a network from a pandas DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing network edge data
        source_col : str
            Column name for source nodes
        target_col : str
            Column name for target nodes
        weight_col : str, optional
            Column name for edge weights
        attrs_cols : list, optional
            List of column names for additional edge attributes
            
        Returns:
        --------
        self : SupplyChainNetwork
            Returns the instance
        """
        try:
            # Reset network
            if isinstance(self.graph, nx.DiGraph):
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.Graph()
                
            # Add all nodes first
            nodes = pd.concat([df[source_col], df[target_col]]).unique()
            self.graph.add_nodes_from(nodes)
            
            # Add edges with attributes
            for _, row in df.iterrows():
                edge_data = {}
                
                if weight_col:
                    edge_data['weight'] = row[weight_col]
                    
                if attrs_cols:
                    for col in attrs_cols:
                        edge_data[col] = row[col]
                        
                self.graph.add_edge(row[source_col], row[target_col], **edge_data)
                
            logger.info(f"Network built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self
            
        except Exception as e:
            logger.error(f"Error building network: {str(e)}")
            raise
    
    def calculate_centrality_metrics(self):
        """
        Calculate various centrality metrics for the network.
        
        Returns:
        --------
        metrics : dict
            Dictionary containing various centrality metrics
        """
        try:
            metrics = {
                'degree_centrality': nx.degree_centrality(self.graph),
                'betweenness_centrality': nx.betweenness_centrality(self.graph),
                'closeness_centrality': nx.closeness_centrality(self.graph)
            }
            
            # For directed graphs, add in and out degree centrality
            if isinstance(self.graph, nx.DiGraph):
                metrics['in_degree_centrality'] = nx.in_degree_centrality(self.graph)
                metrics['out_degree_centrality'] = nx.out_degree_centrality(self.graph)
            
            # Calculate eigenvector centrality for connected graphs
            if nx.is_connected(self.graph.to_undirected()):
                metrics['eigenvector_centrality'] = nx.eigenvector_centrality(self.graph, max_iter=1000)
            
            self.metrics = metrics
            logger.info("Calculated centrality metrics for the network")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating centrality metrics: {str(e)}")
            raise
    
    def identify_communities(self, resolution=1.0):
        """
        Identify communities in the network using Louvain algorithm.
        
        Parameters:
        -----------
        resolution : float
            Resolution parameter for the Louvain algorithm (default: 1.0)
            
        Returns:
        --------
        communities : dict
            Dictionary mapping nodes to community IDs
        """
        try:
            undirected_graph = self.graph.to_undirected()
            self.communities = community_louvain.best_partition(undirected_graph, resolution=resolution)
            
            # Count nodes in each community
            community_sizes = defaultdict(int)
            for node, community_id in self.communities.items():
                community_sizes[community_id] += 1
                
            logger.info(f"Identified {len(set(self.communities.values()))} communities in the network")
            logger.info(f"Community sizes: {dict(community_sizes)}")
            
            return self.communities
            
        except Exception as e:
            logger.error(f"Error identifying communities: {str(e)}")
            raise
    
    def identify_critical_nodes(self, top_n=10):
        """
        Identify critical nodes in the supply chain network.
        
        Parameters:
        -----------
        top_n : int
            Number of top critical nodes to identify (default: 10)
            
        Returns:
        --------
        critical_nodes : dict
            Dictionary containing different sets of critical nodes
        """
        try:
            if not self.metrics:
                self.calculate_centrality_metrics()
                
            # Get critical nodes based on different centrality measures
            critical_nodes = {}
            
            for metric_name, metric_values in self.metrics.items():
                sorted_nodes = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                critical_nodes[metric_name] = dict(sorted_nodes[:top_n])
                
            # Identify nodes that are critical based on multiple metrics
            all_critical_nodes = set()
            for nodes in critical_nodes.values():
                all_critical_nodes.update(nodes.keys())
                
            consensus_scores = {node: 0 for node in all_critical_nodes}
            for metric, nodes in critical_nodes.items():
                for node in nodes:
                    consensus_scores[node] += 1
                    
            critical_nodes['consensus_critical'] = {
                node: score/len(critical_nodes) 
                for node, score in sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
                if score > 1  # Node appears in at least 2 metrics
            }
            
            self.critical_nodes = critical_nodes
            logger.info(f"Identified {len(critical_nodes['consensus_critical'])} consensus critical nodes")
            
            return critical_nodes
            
        except Exception as e:
            logger.error(f"Error identifying critical nodes: {str(e)}")
            raise
    
    def calculate_supply_chain_resilience(self):
        """
        Calculate supply chain network resilience metrics.
        
        Returns:
        --------
        resilience_metrics : dict
            Dictionary containing resilience metrics
        """
        try:
            if not self.critical_nodes:
                self.identify_critical_nodes()
                
            # Create a copy of the graph for resilience analysis
            G = self.graph.copy()
            
            # Initial network properties
            initial_properties = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'connected_components': nx.number_weakly_connected_components(G) if isinstance(G, nx.DiGraph) 
                                       else nx.number_connected_components(G),
                'avg_path_length': nx.average_shortest_path_length(G) if nx.is_strongly_connected(G) 
                                   else 'Not applicable'
            }
            
            # Simulate removal of critical nodes
            consensus_critical_nodes = list(self.critical_nodes['consensus_critical'].keys())[:5]
            G.remove_nodes_from(consensus_critical_nodes)
            
            # Post-disruption network properties
            post_properties = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'connected_components': nx.number_weakly_connected_components(G) if isinstance(G, nx.DiGraph) 
                                       else nx.number_connected_components(G),
                'avg_path_length': nx.average_shortest_path_length(G) if (isinstance(G, nx.DiGraph) and nx.is_strongly_connected(G)) 
                                  or (not isinstance(G, nx.DiGraph) and nx.is_connected(G))
                                  else 'Not applicable'
            }
            
            # Calculate resilience metrics
            resilience_metrics = {
                'node_loss_percentage': (initial_properties['nodes'] - post_properties['nodes']) / initial_properties['nodes'] * 100,
                'edge_loss_percentage': (initial_properties['edges'] - post_properties['edges']) / initial_properties['edges'] * 100,
                'fragmentation_increase': post_properties['connected_components'] - initial_properties['connected_components'],
                'nodes_removed': consensus_critical_nodes
            }
            
            logger.info(f"Calculated resilience metrics: {resilience_metrics}")
            return resilience_metrics
            
        except Exception as e:
            logger.error(f"Error calculating resilience metrics: {str(e)}")
            raise
    
    def visualize_network(self, figsize=(12, 8), save_path=None, show_labels=True, node_size_factor=100, color_by_community=True):
        """
        Visualize the supply chain network.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save the network visualization
        show_labels : bool
            Whether to show node labels
        node_size_factor : int
            Factor to multiply node sizes by
        color_by_community : bool
            Whether to color nodes by community
            
        Returns:
        --------
        fig, ax : tuple
            Matplotlib figure and axis objects
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            plt.axis('off')
            
            # Set node colors by community if available
            if color_by_community and self.communities:
                # Get unique community IDs
                community_ids = set(self.communities.values())
                colors = plt.cm.rainbow(np.linspace(0, 1, len(community_ids)))
                color_map = dict(zip(community_ids, colors))
                
                # Map each node to a color
                node_colors = [color_map[self.communities[node]] for node in self.graph.nodes()]
            else:
                node_colors = 'skyblue'
                
            # Set node sizes based on degree centrality
            node_sizes = [self.graph.degree(node) * node_size_factor for node in self.graph.nodes()]
            
            # Draw the network
            pos = nx.spring_layout(self.graph, seed=42)
            nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=isinstance(self.graph, nx.DiGraph))
            
            if show_labels:
                nx.draw_networkx_labels(self.graph, pos, font_size=8, font_family='sans-serif')
                
            plt.title('Supply Chain Network Visualization')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Network visualization saved to {save_path}")
            
            return fig, ax
            
        except Exception as e:
            logger.error(f"Error visualizing network: {str(e)}")
            raise

def create_weighted_edge_list(from_df, edge_attr_cols=None):
    """
    Create a weighted edge list for network construction from a DataFrame.
    
    Parameters:
    -----------
    from_df : pandas.DataFrame
        DataFrame containing supply chain data
    edge_attr_cols : list, optional
        List of column names for additional edge attributes
        
    Returns:
    --------
    edge_list : pandas.DataFrame
        DataFrame containing formatted edge list
    """
    # Implementation would depend on the specific data structure
    pass 