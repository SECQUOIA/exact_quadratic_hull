import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def create_plots():
    # Get all Excel files in the current directory
    excel_files = glob.glob('*.xlsx')
    
    # Process data from all files
    all_data = []
    for file in excel_files:
        # Read the Excel file
        df = pd.read_excel(file)
        all_data.append(df)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data)
    else:
        print("No Excel files found!")
        return
    
    # Create full range plots (both connected and scatter)
    create_plot(combined_df, filter_reactors=False, scatter_only=False)
    create_plot(combined_df, filter_reactors=False, scatter_only=True)
    


def create_plot(df, filter_reactors=False, max_reactors=None, scatter_only=False):
    # Create a single figure
    plt.figure(figsize=(12, 8))
    
    # Define different markers
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    
    # Create fixed color mapping for strategies
    strategy_colors = {
        'gdp.bigm': 'blue',  
        'gdp.hull': 'brown',  
        'gdp.hull_exact': 'green',  
        'gdp.hull_reduced_y': 'purple',  
        'gdp.binary_multiplication': 'orange',  
    }
    
    # Keep track of styles we've used
    style_idx = 0
    
    # Get unique strategies
    strategies = df['strategy'].unique()
    strats = []
    if "gdp.bigm" in strategies:
        strats.append("gdp.bigm")
    if "gdp.hull" in strategies:
        strats.append("gdp.hull")
    if "gdp.hull_exact" in strategies:
        strats.append("gdp.hull_exact")
    if "gdp.binary_multiplication" in strategies:
        strats.append("gdp.binary_multiplication")


    strategies = strats
    
    # Plot data for each strategy
    for strategy in strategies:
        # Filter data for the current strategy
        strategy_data = df[df['strategy'] == strategy]
        
        # Apply reactor number filter if needed
        if filter_reactors and max_reactors is not None:
            strategy_data = strategy_data[strategy_data['num_reactors'] <= max_reactors]
        
        # Group by number of reactors and calculate mean and std
        grouped_data = strategy_data.groupby('num_reactors').agg({
            'solve_time_sec': ['mean']  # Only calculate mean, remove std
        }).reset_index()
        
        # Get color from fixed mapping and marker from rotation
        color = strategy_colors.get(strategy, '#808080')  # Default to gray if strategy not in mapping
        marker = markers[style_idx % len(markers)]
        style_idx += 1
        
        # Use strategy name directly as label
        display_name = strategy.replace("gdp.", "")
        display_name = display_name.replace("bigm", "BigM")
        display_name = display_name.replace("hull_exact", "Hull Exact")
        display_name = display_name.replace("hull", "Hull(ε-approx.)")
        display_name = display_name.replace("hull_reduced_y", "Hull Reduced Y")
        display_name = display_name.replace("binary_multiplication", "Binary Mult.")
        label = display_name
        
        if scatter_only:
            # Plot only scatter points
            plt.scatter(
                grouped_data['num_reactors'],
                grouped_data[('solve_time_sec', 'mean')],
                marker=marker,
                color=color,
                label=label,
                s=100  # Make points larger
            )
        else:
            # Plot with connecting lines
            plt.plot(
                grouped_data['num_reactors'],
                grouped_data[('solve_time_sec', 'mean')],
                marker=marker,
                color=color,
                linestyle='-',
                linewidth=6,
                label=label
            )
    
    # Add time limit line
    plt.axhline(y=3600, color='r', linestyle='--', alpha=0.7, label='Time Limit (3600s)', linewidth=4)
    
    # Customize the plot
    if filter_reactors and max_reactors is not None:
        title = f'Solve Time vs Number of Reactors (Up to {max_reactors})'
        output_filename = f'solve_time_vs_reactors_up_to_{max_reactors}'
    else:
        title = 'Solve Time vs Number of Reactors'
        output_filename = 'solve_time_vs_reactors_all_files'
    
    # Add scatter/line indicator to filename
    if scatter_only:
        output_filename += '_scatter'
    else:
        output_filename += '_lines'
    
    
    #plt.title(title, fontsize=40)
    plt.xlabel('Number of Reactors', fontsize=38)
    plt.ylabel('Solve Time (seconds)', fontsize=38)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Add both major and minor grid lines
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(loc='lower right', fontsize=27, framealpha=0.6)  # Place legend inside plot in bottom right corner
    
    # Adjust layout to accommodate the legend
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_filename + '.jpg', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    create_plots()
