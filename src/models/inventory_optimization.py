import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InventoryOptimizer:
    """A class for optimizing inventory levels in a supply chain"""
    
    def __init__(self):
        """Initialize the InventoryOptimizer with default parameters"""
        self.results = {}
        self.sensitivity_analysis = {}
        self.items_data = None
    
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load inventory item data into the optimizer
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing inventory item information with columns:
            - item_id: unique identifier for each item
            - demand_mean: average demand per period
            - demand_std: standard deviation of demand per period
            - lead_time_mean: average lead time for replenishment
            - lead_time_std: standard deviation of lead time
            - holding_cost: cost to hold one unit for one period
            - stockout_cost: cost of a stockout per unit
            - order_cost: fixed cost to place an order
        """
        required_columns = [
            'item_id', 'demand_mean', 'demand_std', 'lead_time_mean', 
            'lead_time_std', 'holding_cost', 'stockout_cost', 'order_cost'
        ]
        
        # Validate data
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate that numeric columns contain valid values
        numeric_cols = required_columns[1:]
        for col in numeric_cols:
            if not np.issubdtype(data[col].dtype, np.number):
                raise ValueError(f"Column '{col}' must contain numeric values")
            
            # Check for negative values in columns that should be non-negative
            if col not in ['demand_mean', 'demand_std'] and (data[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values")
        
        self.items_data = data.copy()
        logger.info(f"Loaded data for {len(data)} inventory items")
    
    def calculate_eoq(self, item_id: str = None) -> Dict:
        """
        Calculate Economic Order Quantity (EOQ) for items
        
        Parameters:
        -----------
        item_id : str, optional
            Specific item ID to calculate EOQ for. If None, calculates for all items.
            
        Returns:
        --------
        Dict
            Dictionary with item_ids as keys and EOQ values as values
        """
        if self.items_data is None:
            raise ValueError("No inventory data loaded. Call load_data() first.")
        
        # Filter data if item_id is provided
        if item_id is not None:
            data = self.items_data[self.items_data['item_id'] == item_id]
            if len(data) == 0:
                raise ValueError(f"Item ID '{item_id}' not found in data")
        else:
            data = self.items_data
        
        # Calculate EOQ for each item
        eoq_results = {}
        for _, row in data.iterrows():
            # EOQ formula: sqrt(2*D*K/h)
            # D = annual demand, K = order cost, h = holding cost
            annual_demand = row['demand_mean'] * 365  # Assuming demand_mean is daily
            order_cost = row['order_cost']
            holding_cost = row['holding_cost']
            
            eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
            eoq_results[row['item_id']] = eoq
        
        # Store results
        self.results['eoq'] = eoq_results
        
        logger.info(f"Calculated EOQ for {len(eoq_results)} items")
        return eoq_results
    
    def calculate_reorder_point(self, service_level: float = 0.95, item_id: str = None) -> Dict:
        """
        Calculate reorder points for inventory items
        
        Parameters:
        -----------
        service_level : float
            Desired service level (probability of not stocking out)
        item_id : str, optional
            Specific item ID to calculate for. If None, calculates for all items.
            
        Returns:
        --------
        Dict
            Dictionary with item_ids as keys and reorder point values as values
        """
        if self.items_data is None:
            raise ValueError("No inventory data loaded. Call load_data() first.")
        
        if not 0 < service_level < 1:
            raise ValueError("Service level must be between 0 and 1")
        
        # Filter data if item_id is provided
        if item_id is not None:
            data = self.items_data[self.items_data['item_id'] == item_id]
            if len(data) == 0:
                raise ValueError(f"Item ID '{item_id}' not found in data")
        else:
            data = self.items_data
        
        # Calculate reorder point for each item
        reorder_points = {}
        for _, row in data.iterrows():
            # Calculate demand during lead time
            ltm = row['lead_time_mean']
            lts = row['lead_time_std']
            dm = row['demand_mean']
            ds = row['demand_std']
            
            # Mean demand during lead time
            mean_ddlt = dm * ltm
            
            # Standard deviation of demand during lead time
            # Using formula that accounts for variability in both demand and lead time
            std_ddlt = np.sqrt((ltm * ds**2) + (dm**2 * lts**2))
            
            # Safety factor based on service level
            z = stats.norm.ppf(service_level)
            
            # Safety stock
            safety_stock = z * std_ddlt
            
            # Reorder point = mean demand during lead time + safety stock
            rop = mean_ddlt + safety_stock
            
            reorder_points[row['item_id']] = rop
        
        # Store results
        self.results['reorder_points'] = reorder_points
        
        logger.info(f"Calculated reorder points for {len(reorder_points)} items")
        return reorder_points
    
    def calculate_safety_stock(self, service_level: float = 0.95, item_id: str = None) -> Dict:
        """
        Calculate safety stock levels for inventory items
        
        Parameters:
        -----------
        service_level : float
            Desired service level (probability of not stocking out)
        item_id : str, optional
            Specific item ID to calculate for. If None, calculates for all items.
            
        Returns:
        --------
        Dict
            Dictionary with item_ids as keys and safety stock values as values
        """
        if self.items_data is None:
            raise ValueError("No inventory data loaded. Call load_data() first.")
        
        if not 0 < service_level < 1:
            raise ValueError("Service level must be between 0 and 1")
        
        # Filter data if item_id is provided
        if item_id is not None:
            data = self.items_data[self.items_data['item_id'] == item_id]
            if len(data) == 0:
                raise ValueError(f"Item ID '{item_id}' not found in data")
        else:
            data = self.items_data
        
        # Calculate safety stock for each item
        safety_stocks = {}
        for _, row in data.iterrows():
            # Calculate standard deviation of demand during lead time
            ltm = row['lead_time_mean']
            lts = row['lead_time_std']
            dm = row['demand_mean']
            ds = row['demand_std']
            
            # Standard deviation of demand during lead time
            std_ddlt = np.sqrt((ltm * ds**2) + (dm**2 * lts**2))
            
            # Safety factor based on service level
            z = stats.norm.ppf(service_level)
            
            # Safety stock
            safety_stock = z * std_ddlt
            
            safety_stocks[row['item_id']] = safety_stock
        
        # Store results
        self.results['safety_stocks'] = safety_stocks
        
        logger.info(f"Calculated safety stocks for {len(safety_stocks)} items")
        return safety_stocks
    
    def perform_abc_analysis(self, value_column: str = 'demand_mean', 
                            a_threshold: float = 0.8, 
                            b_threshold: float = 0.95) -> pd.DataFrame:
        """
        Perform ABC analysis on inventory items
        
        Parameters:
        -----------
        value_column : str
            Column to use for determining item value
        a_threshold : float
            Threshold for category A (e.g., 0.8 means top items accounting for 80% of total value)
        b_threshold : float
            Threshold for category B (e.g., 0.95 means items accounting for the next 15% of total value)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with original data plus ABC classification
        """
        if self.items_data is None:
            raise ValueError("No inventory data loaded. Call load_data() first.")
        
        if value_column not in self.items_data.columns:
            raise ValueError(f"Column '{value_column}' not found in data")
        
        if not (0 < a_threshold < b_threshold < 1):
            raise ValueError("Thresholds must satisfy 0 < a_threshold < b_threshold < 1")
        
        # Create a copy of the data
        result_df = self.items_data.copy()
        
        # Sort by value in descending order
        result_df = result_df.sort_values(by=value_column, ascending=False)
        
        # Calculate cumulative value and percentage
        total_value = result_df[value_column].sum()
        result_df['cumulative_value'] = result_df[value_column].cumsum()
        result_df['cumulative_percentage'] = result_df['cumulative_value'] / total_value
        
        # Assign ABC categories
        result_df['abc_category'] = 'C'
        result_df.loc[result_df['cumulative_percentage'] <= a_threshold, 'abc_category'] = 'A'
        result_df.loc[(result_df['cumulative_percentage'] > a_threshold) & 
                     (result_df['cumulative_percentage'] <= b_threshold), 'abc_category'] = 'B'
        
        # Count items in each category
        categories = result_df['abc_category'].value_counts()
        for category, count in categories.items():
            logger.info(f"Category {category}: {count} items")
        
        # Store results
        self.results['abc_analysis'] = result_df
        
        return result_df
    
    def calculate_total_costs(self, order_quantities: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate total inventory costs based on given order quantities
        
        Parameters:
        -----------
        order_quantities : Dict[str, float], optional
            Dictionary with item_ids as keys and order quantities as values.
            If None, uses calculated EOQ values if available.
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with item_ids as keys and total annual costs as values
        """
        if self.items_data is None:
            raise ValueError("No inventory data loaded. Call load_data() first.")
        
        # If no order quantities provided, use EOQ if available
        if order_quantities is None:
            if 'eoq' not in self.results:
                self.calculate_eoq()
            order_quantities = self.results['eoq']
        
        # Calculate total costs for each item
        total_costs = {}
        cost_components = {}
        
        for _, row in self.items_data.iterrows():
            item_id = row['item_id']
            
            if item_id not in order_quantities:
                logger.warning(f"No order quantity specified for item '{item_id}'. Skipping.")
                continue
            
            q = order_quantities[item_id]
            annual_demand = row['demand_mean'] * 365  # Assuming demand_mean is daily
            
            # Annual ordering cost = (annual demand / order quantity) * cost per order
            annual_ordering_cost = (annual_demand / q) * row['order_cost']
            
            # Annual holding cost = (average inventory) * holding cost per unit
            annual_holding_cost = (q / 2) * row['holding_cost']
            
            # Total annual cost
            total_annual_cost = annual_ordering_cost + annual_holding_cost
            
            total_costs[item_id] = total_annual_cost
            cost_components[item_id] = {
                'ordering_cost': annual_ordering_cost,
                'holding_cost': annual_holding_cost,
                'total_cost': total_annual_cost
            }
        
        # Store results
        self.results['total_costs'] = total_costs
        self.results['cost_components'] = cost_components
        
        logger.info(f"Calculated total costs for {len(total_costs)} items")
        return total_costs
    
    def run_sensitivity_analysis(self, item_id: str, 
                                parameter: str, 
                                range_percent: float = 0.3,
                                steps: int = 10) -> Dict:
        """
        Perform sensitivity analysis on a parameter for a specific item
        
        Parameters:
        -----------
        item_id : str
            Item ID to analyze
        parameter : str
            Parameter to vary (e.g., 'demand_mean', 'lead_time_mean', etc.)
        range_percent : float
            Percentage range to vary the parameter (e.g., 0.3 for ±30%)
        steps : int
            Number of steps to divide the range into
            
        Returns:
        --------
        Dict
            Dictionary with parameter values as keys and results as values
        """
        if self.items_data is None:
            raise ValueError("No inventory data loaded. Call load_data() first.")
        
        if item_id not in self.items_data['item_id'].values:
            raise ValueError(f"Item ID '{item_id}' not found in data")
        
        if parameter not in self.items_data.columns:
            raise ValueError(f"Parameter '{parameter}' not found in data")
        
        # Get the base value of the parameter
        item_data = self.items_data[self.items_data['item_id'] == item_id].iloc[0]
        base_value = item_data[parameter]
        
        # Create a range of values to test
        min_value = base_value * (1 - range_percent)
        max_value = base_value * (1 + range_percent)
        test_values = np.linspace(min_value, max_value, steps)
        
        # Initialize results
        results = {}
        
        # Make a copy of the original data
        original_data = self.items_data.copy()
        
        # For each test value
        for test_value in test_values:
            # Modify the parameter for this item
            self.items_data = original_data.copy()
            self.items_data.loc[self.items_data['item_id'] == item_id, parameter] = test_value
            
            # Calculate EOQ and costs
            self.calculate_eoq(item_id)
            eoq = self.results['eoq'][item_id]
            
            self.calculate_reorder_point(item_id=item_id)
            rop = self.results['reorder_points'][item_id]
            
            self.calculate_safety_stock(item_id=item_id)
            safety_stock = self.results['safety_stocks'][item_id]
            
            self.calculate_total_costs({item_id: eoq})
            total_cost = self.results['total_costs'][item_id]
            
            # Store results
            results[test_value] = {
                'eoq': eoq,
                'reorder_point': rop,
                'safety_stock': safety_stock,
                'total_cost': total_cost
            }
        
        # Restore original data
        self.items_data = original_data
        
        # Store sensitivity analysis results
        if item_id not in self.sensitivity_analysis:
            self.sensitivity_analysis[item_id] = {}
        self.sensitivity_analysis[item_id][parameter] = results
        
        logger.info(f"Performed sensitivity analysis on '{parameter}' for item '{item_id}'")
        return results
    
    def plot_sensitivity_analysis(self, item_id: str, parameter: str, metric: str = 'total_cost',
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the results of a sensitivity analysis
        
        Parameters:
        -----------
        item_id : str
            Item ID that was analyzed
        parameter : str
            Parameter that was varied
        metric : str
            Metric to plot (e.g., 'eoq', 'total_cost', etc.)
        figsize : Tuple[int, int]
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if item_id not in self.sensitivity_analysis or parameter not in self.sensitivity_analysis[item_id]:
            raise ValueError(f"No sensitivity analysis found for item '{item_id}' and parameter '{parameter}'")
        
        # Get sensitivity analysis results
        results = self.sensitivity_analysis[item_id][parameter]
        
        # Extract parameter values and metric values
        param_values = list(results.keys())
        metric_values = [results[p][metric] for p in param_values]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(param_values, metric_values, 'o-', linewidth=2)
        
        # Add labels and title
        ax.set_xlabel(parameter)
        ax.set_ylabel(metric)
        ax.set_title(f'Sensitivity of {metric} to {parameter} for item {item_id}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format axes
        ax.ticklabel_format(style='plain')
        
        return fig
    
    def optimize_multi_echelon_inventory(self, network_data: pd.DataFrame, 
                                       lead_times: Dict[Tuple[str, str], float],
                                       service_levels: Dict[str, float]) -> Dict:
        """
        Optimize inventory levels in a multi-echelon supply chain
        
        Parameters:
        -----------
        network_data : pd.DataFrame
            DataFrame with columns: 'source_node', 'target_node', 'item_id', 'demand_mean', 'demand_std'
        lead_times : Dict[Tuple[str, str], float]
            Dictionary with (source, target) tuples as keys and lead times as values
        service_levels : Dict[str, float]
            Dictionary with node IDs as keys and required service levels as values
            
        Returns:
        --------
        Dict
            Dictionary containing optimal inventory levels for each node
        """
        # Implementation would depend on specific multi-echelon optimization approach
        # This is a placeholder for future implementation
        logger.info("Multi-echelon inventory optimization not yet implemented")
        return {} 