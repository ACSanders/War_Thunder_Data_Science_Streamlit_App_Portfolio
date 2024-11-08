import streamlit as st
st.set_page_config(layout="wide")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta, norm, uniform
import matplotlib.pyplot as plt

title_col, logo_col = st.columns(2)

with title_col:
    st.write("""
            # War Thunder Science :boom:
            ##### Vehicle Performance and Game Statistics Using Bayesian A/B Testing, K-Means Clustering, and Data Science :chart_with_upwards_trend:
            """)
    st.write('Developed by **A. Sanders** - also known as, *DrKnoway*')

with logo_col:
    st.empty()  # Create an empty space
    st.image("knoway_eye.png", width=150)


st.header("Vehicle Trends")

# Load and cache the DataFrame
@st.cache  # You can upgrade Streamlit to use @st.cache_data for newer versions
def load_data():
    return pd.read_csv("full_data.csv")

# Load the data
data = load_data()

# Make a copy of the DataFrame
df_copy = data.copy()

# Ensure 'date' column is in datetime format for the entire DataFrame
df_copy['date'] = pd.to_datetime(df_copy['date'])

# Create columns for the dropdown menus
col1, col2, col3, col4 = st.columns(4)

# 1st dropdown for Vehicle Type
with col1:
    vehicle_types = df_copy['cls'].unique()
    selected_vehicle_types = st.selectbox("Vehicle Type", vehicle_types)

# Filter the DataFrame based on the selected Vehicle Type
filtered_by_type_df = df_copy[df_copy['cls'] == selected_vehicle_types]

# 2nd dropdown for Nation
with col2:
    nations = filtered_by_type_df['nation'].unique()
    selected_nations = st.multiselect("Nation", nations, default=[nations[0]] if nations.size > 0 else [])

# Filter the DataFrame based on selected Nation
filtered_by_nation_df = filtered_by_type_df[filtered_by_type_df['nation'].isin(selected_nations)]

# 3rd dropdown for BR
with col3:
    br_values = filtered_by_nation_df['rb_br'].unique()
    selected_br = st.multiselect("BR", sorted(br_values, reverse=True), default=[sorted(br_values, reverse=True)[0]] if br_values.size > 0 else [])

# Filter the DataFrame based on selected BR
filtered_by_br_df = filtered_by_nation_df[filtered_by_nation_df['rb_br'].isin(selected_br)]

# 4th dropdown for Metric
with col4:
    metrics = [
        'rb_win_rate',
        'rb_battles',  
        'rb_air_frags_per_death', 
        'rb_air_frags_per_battle', 
        'rb_ground_frags_per_death', 
        'rb_ground_frags_per_battle' 
    ]
    selected_metric = st.selectbox("Metric", metrics, index=0)  # Defaults to the first metric

# 5th dropdown for Vehicle Name (filtered based on Vehicle Type and Nation)
vehicle_names = filtered_by_br_df['name'].unique()
selected_vehicle_names = st.multiselect("Vehicle Name", vehicle_names, default=list(vehicle_names) if vehicle_names.size > 0 else [])

# Filter the DataFrame based on user selections
final_filtered_df = df_copy[
    (df_copy['cls'] == selected_vehicle_types) &
    (df_copy['nation'].isin(selected_nations)) &
    (df_copy['rb_br'].isin(selected_br)) &
    (df_copy['name'].isin(selected_vehicle_names))
]

# Sort the DataFrame by date and get unique dates, formatted as strings (MM/DD/YY)
unique_dates = sorted(df_copy['date'].dt.strftime('%m/%d/%y').unique())

# Add a date range select slider to Streamlit
date_range = st.select_slider(
    "Select date range:",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1]),
    key = 'date_range_slider'
)

# Convert selected start and end dates back to datetime for filtering
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter the final DataFrame based on the date range
final_filtered_df = final_filtered_df[
    (final_filtered_df['date'] >= start_date) & 
    (final_filtered_df['date'] <= end_date)
]

# Display the number of rows in the filtered DataFrame for debugging
st.write(f"Filtered DataFrame rows: {final_filtered_df.shape[0]}")  # Display row count

# Check if there are any rows to plot
if final_filtered_df.shape[0] > 0:
    # Create a line plot with Plotly Express for each vehicle
    fig = px.line(final_filtered_df, 
                  x='date', 
                  y=selected_metric, 
                  color='name',  # This will create separate lines for each vehicle
                  title='Performance Over Time', 
                  labels={'date': 'Date', selected_metric: selected_metric}, 
                  hover_data=['name', 'nation', 'rb_br'])
    
    fig.update_layout(template='plotly')

    # Customize further if needed
    fig.update_traces(line=dict(width=2), mode='lines+markers')  # Set line width and add markers
    fig.update_layout(title=dict(font=dict(size=24)), xaxis_title=dict(font=dict(size=18)), yaxis_title=dict(font=dict(size=18)))  # Update titles
    
    # Show the figure, fill the container width
    st.plotly_chart(fig, use_container_width=True)  # Add use_container_width=True to fill the container
else:
    st.write("No data available for the selected filters.")


st.subheader("Aggregated Win Rates by Nation")

# copy of our data
wr_df = data.copy()

# Ensure 'date' column is in datetime format
wr_df['date'] = pd.to_datetime(wr_df['date'])

# Get unique vehicle types from the 'cls' column
vehicle_types = wr_df['cls'].unique()

# Add a dropdown menu to select the vehicle type
selected_vehicle_type = st.selectbox(
    "Select Vehicle Type:",
    options=vehicle_types,
    index=0  # Optional: Sets the default selection to the first type
)

# Sort the DataFrame by date and get unique dates, formatted as strings (MM/DD/YY)
unique_dates = sorted(wr_df['date'].dt.strftime('%m/%d/%y').unique())

# Add a date range select slider to Streamlit
date_range = st.select_slider(
    "Select date range:",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1])
)

# Convert selected start and end dates back to datetime for filtering
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter the DataFrame based on the selected vehicle type and date range
filtered_df = wr_df[
    (wr_df['cls'] == selected_vehicle_type) &
    (wr_df['date'] >= start_date) & 
    (wr_df['date'] <= end_date)
]

# Continue with your processing for the filtered DataFrame
filtered_df['br_range'] = np.floor(filtered_df['rb_br']).astype(int)
agg_wr_df = filtered_df.groupby(['nation', 'br_range']).rb_win_rate.mean().reset_index()
agg_wr_pivot = agg_wr_df.pivot(index='nation', columns='br_range', values='rb_win_rate')

# Plot heatmap with custom text annotations for displaying rounded values
fig_wr_heatmap = px.imshow(
    agg_wr_pivot,
    color_continuous_scale='RdBu',
    labels=dict(x='BR Range', y='Nation', color='rb_win_rate')
)

# Display all x-axis ticks
fig_wr_heatmap.update_xaxes(tickmode='linear')

# Add annotations with rounded values
fig_wr_heatmap.update_traces(
    text=agg_wr_pivot.round(1).astype(str),  # Round values to 1 decimal and convert to string for display
    texttemplate="%{text}",  # Use the rounded values as text
    textfont=dict(size=10),  # Adjust font size if needed
    hoverinfo='text'  # Show these values in the hover tooltips
)

# Update layout for aesthetics if needed
fig_wr_heatmap.update_layout(template='plotly')

# Display the heatmap in Streamlit
st.plotly_chart(fig_wr_heatmap, use_container_width=True)


######################################################################################

#### k-means clustering

######################################################################################

st.header('k-Means Clustering')
st.subheader('Ranked Vehicle Performance Groups')
st.write("""k-Means clustering is performed on several engagement variables like *K/D*
            and *vehicles destroyed per battle*. Vehicles are automatically clustered into one
            of three performance groups: **high performers**, **moderate performers**, 
            and **low performers**.""")


# Function to plot a scatter plot with Plotly
def plot_scatter_plot(df, x_metric, y_metric, color_metric):
    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color=color_metric,
        hover_data=['name', 'cls', 'nation'],
        # title=f'Scatter Plot of {y_metric} vs {x_metric} Colored by {color_metric}'
        title = f'Scatter Plot of K/D vs Frags per Battle Colored by Performance Cluster'
    )
    fig.update_layout(
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        legend_title=color_metric
    )
    return fig

# Assuming df_copy has already been created earlier
key_metrics = ['rb_ground_frags_per_death', 
               'rb_ground_frags_per_battle', 
              ]       
               # 'rb_win_rate'] # removing win_rate as it was causing some issues with 

# Function to filter, create the 'br_range' column, group by 'name', and segment data for the last month
def filter_and_segment_data(df, key_metrics):
    recent_date = datetime.now() - timedelta(days=30)
    
    # Filter data for the last 2 months and drop rows with NaN in key metrics
    filtered_df = df[df['date'] >= recent_date].dropna(subset=key_metrics)
    
    # Remove rows where 'cls' is 'Fleet' or 'Aviation'
    filtered_df = filtered_df[~filtered_df['cls'].isin(['Fleet', 'Aviation'])]
    
    # Create the 'br_range' column based on 'rb_br'
    filtered_df['br_range'] = np.floor(filtered_df['rb_br']).astype(int)
    
    # Group by 'name' and aggregate key metrics by taking the mean
    aggregated_df = filtered_df.groupby('name', as_index=False).agg({
        'rb_ground_frags_per_death': 'mean',
        'rb_ground_frags_per_battle': 'mean',
        'rb_win_rate': 'mean',
        'br_range': 'first',  # Keep the first BR range per name if consistent
        'cls': 'first',       # Keep the first cls per name if consistent
        'nation': 'first'     # Keep the first nation per name if consistent
    })
    
    # Segment data by 'br_range'
    segmented_data = {br: aggregated_df[aggregated_df['br_range'] == br] for br in range(1, 14)}
    
    return segmented_data

def perform_kmeans_and_label(segmented_data, key_metrics):
    scaler = StandardScaler()
    results = {}

    for br, data in segmented_data.items():
        if not data.empty:
            # Standardize the data for clustering
            standardized_metrics = scaler.fit_transform(data[key_metrics])

            # Perform k-means clustering (3 clusters)
            kmeans = KMeans(n_clusters=3, random_state=42)
            data['cluster'] = kmeans.fit_predict(standardized_metrics)
            
            # Get the centroids (the average position of each cluster in the feature space)
            centroids = kmeans.cluster_centers_

            # Inverse transform centroids back to the original scale
            centroids_original_scale = scaler.inverse_transform(centroids)

            # Calculate the average value for each cluster's centroid across all key metrics
            avg_centroids = np.mean(centroids_original_scale, axis=1)

            # Assign labels based on the average values of the centroids
            label_map = {}
            for i, avg_value in enumerate(avg_centroids):
                if avg_value == max(avg_centroids):
                    label_map[i] = 'high performance'
                elif avg_value == min(avg_centroids):
                    label_map[i] = 'low performance'
                else:
                    label_map[i] = 'moderate performance'

            # Map clusters to performance labels
            data['performance_label'] = data['cluster'].map(label_map)

            results[br] = data

            # Debug: Print centroids and labels for verification
            print(f"BR {br}: Centroids (original scale)")
            for i, centroid in enumerate(centroids_original_scale):
                print(f"Cluster {i} - Centroid: {centroid}, Label: {label_map[i]}")

    return results

# Ensure the 'date' column is a datetime type
df_copy['date'] = pd.to_datetime(df_copy['date'])

# Filter and segment data
segmented_data = filter_and_segment_data(df_copy, key_metrics)

# Streamlit user interaction: Select BR range
selected_br = st.selectbox("Select BR range for clustering results", list(segmented_data.keys()))

# Debug: Check and display the type of `selected_br`
# st.write(f"Selected BR: {selected_br} (Type: {type(selected_br)})")

# Run clustering and display results
clustering_results = perform_kmeans_and_label(segmented_data, key_metrics)

# Debug: Print keys of `clustering_results` to ensure they are integers
print("Clustering results keys:", clustering_results.keys())

# Check if the selected BR data is available for plotting
selected_br_data = clustering_results.get(selected_br)

# Example usage for Streamlit: Display data and plot scatter plot
if selected_br_data is not None and not selected_br_data.empty:
    st.dataframe(selected_br_data)
    st.write("Clustering completed for BR:", selected_br, use_container_width=True)
    
    fig = plot_scatter_plot(
        selected_br_data,
        x_metric='rb_ground_frags_per_death',
        y_metric='rb_ground_frags_per_battle',
        color_metric='performance_label'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available for the selected BR.")

########################################################################################

### Bayesian Statistics ###

#########################################################################################

st.header("Bayesian A/B Testing")
st.subheader("Probability that One Vehicle (A) Has a Better K/D than Another Vehicle (B)")
st.write("Please select two vehicles to run a Bayesian test on *K/D*")

# function to run Bayesian testing on numeric or continuous data
def bayesian_ab_test_numeric(vehicle_one_series, vehicle_two_series, vehicle_one_name, vehicle_two_name, n_simulations=10000):
    # Calculate sample statistics
    test_mean, test_std = np.mean(vehicle_one_series), np.std(vehicle_one_series, ddof=1)
    control_mean, control_std = np.mean(vehicle_two_series), np.std(vehicle_two_series, ddof=1)
    test_n, control_n = len(vehicle_one_series), len(vehicle_two_series)
    
    # Priors
    mu0, s0, n0 = 0, 1, 0
    
    # Posterior parameters
    inv_vars_test = (n0 / s0**2, test_n / test_std**2)
    posterior_mean_test = np.average((mu0, test_mean), weights=inv_vars_test)
    posterior_std_test = 1 / np.sqrt(np.sum(inv_vars_test))
    
    inv_vars_control = (n0 / s0**2, control_n / control_std**2)
    posterior_mean_control = np.average((mu0, control_mean), weights=inv_vars_control)
    posterior_std_control = 1 / np.sqrt(np.sum(inv_vars_control))
    
    # Monte Carlo Sampling
    test_samples = norm.rvs(loc=posterior_mean_test, scale=posterior_std_test, size=n_simulations)
    control_samples = norm.rvs(loc=posterior_mean_control, scale=posterior_std_control, size=n_simulations)
    
    # Probability that vehicle one beats vehicle two
    prob_vehicle_one_beats_vehicle_two = round(np.mean(test_samples > control_samples), 2) * 100
    
    # Credible Interval for the difference
    diff_samples = test_samples - control_samples
    credible_interval = np.percentile(diff_samples, [2.5, 97.5])
    
    # Streamlit output with vehicle names
    st.write(f"""Probability that the *{vehicle_one_name}* has a better K/D than the *{vehicle_two_name}* = **<span style='color: green;'>{prob_vehicle_one_beats_vehicle_two}%</span>**""", unsafe_allow_html=True)

    
    st.write(f"95% Credibility Interval for difference: [{round(credible_interval[0],1)}, {round(credible_interval[1],1)}]")
    
    return test_samples, control_samples, diff_samples, credible_interval



####################

# Plotting function for posterior distributions with specific vehicle names
def create_posterior_plots(test_samples, control_samples, vehicle_one_name, vehicle_two_name, test_mean, control_mean):
    # Create the histogram using Plotly figure factory
    fig_a = ff.create_distplot([test_samples, control_samples], 
                               group_labels=[vehicle_one_name, vehicle_two_name], 
                               )
    fig_a.update_traces(nbinsx = 100, autobinx = True, selector = {'type':'histogram'})
    fig_a.add_vline(x=test_samples.mean(), line_width = 3, line_dash='dash', line_color= 'hotpink', annotation_text = f'mean <br> {round(test_mean,1)}', annotation_position = 'bottom')
    fig_a.add_vline(x=control_samples.mean(), line_width = 3, line_dash='dash', line_color= 'purple', annotation_text = f'mean <br> {round(control_mean,1)}', annotation_position = 'bottom')
    fig_a.update_layout(title_text = 'Posterior Distributions',
                        autosize = True,
                        height = 600)
    
    # Display in Streamlit
    st.plotly_chart(fig_a, use_container_width=True)

# Plotting function for difference distribution with specific vehicle names
def create_difference_plot(diff_samples, credible_interval, vehicle_one_name, vehicle_two_name):
    # Round the credible interval and median to 1 decimal
    credible_interval_rounded = [round(val, 1) for val in credible_interval]
    diff_samples_median = round(np.median(diff_samples), 1)

    # Create the histogram using Plotly figure factory
    fig2b = ff.create_distplot([diff_samples], 
                               group_labels=[f"{vehicle_one_name} - {vehicle_two_name}"], 
                               colors = ['aquamarine'],
    )

    fig2b.update_traces(nbinsx = 100, autobinx=True, selector = {'type':'histogram'})

    # Add vertical lines for credibility interval, 0 difference, and median
    fig2b.add_vline(x=credible_interval_rounded[0], line_width=3, line_dash='dash', line_color='red', 
                    annotation_text=f'95% Lower Bound <br>{credible_interval_rounded[0]}', annotation_position='top')
    fig2b.add_vline(x=credible_interval_rounded[1], line_width=3, line_dash='dash', line_color='red', 
                    annotation_text=f'95% Upper Bound <br>{credible_interval_rounded[1]}', annotation_position='top')
    fig2b.add_vline(x=diff_samples_median, line_width=3, line_color='dodgerblue', 
                    annotation_text=f'Median <br>{diff_samples_median}', annotation_position='bottom')
    fig2b.add_vline(x=0, line_width=3, line_dash='dot', line_color='orange', 
                    annotation_text='0', annotation_position='bottom')

    # Update the layout of the plot
    fig2b.update_layout(
        title_text=f"Distribution of Differences ({vehicle_one_name} - {vehicle_two_name})",
        height = 600,
        autosize=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig2b, use_container_width=True)


####################################

df_bayes = data.copy()

metric = 'rb_frags_per_death'
data_type = 'numeric' 

# Convert 'date' column to datetime
df_bayes['date'] = pd.to_datetime(df_bayes['date'], errors='coerce')

# Initial DataFrame filtering for 'Ground_vehicles' and removing nulls
df_bayes_filtered = df_bayes[(df_bayes['cls'] == 'Ground_vehicles') & df_bayes['rb_ground_frags_per_death'].notna()]

# Filter for dates within the last 60 days
sixty_days_ago = datetime.now() - timedelta(days=60)
df_bayes_filtered = df_bayes_filtered[df_bayes_filtered['date'] >= sixty_days_ago]

# Create Streamlit columns for separate selections
col1, col2 = st.columns(2)

# Column 1: First vehicle selection process
with col1:
    st.subheader("First Vehicle Selection")
    
    # Dropdown for selecting nation with placeholder option
    nation_one = st.selectbox("Select Nation for First Vehicle:", ["Select a nation..."] + sorted(df_bayes_filtered['nation'].unique()))
    
    # Filter after checking if a valid nation is selected
    if nation_one != "Select a nation...":
        df_nation_one = df_bayes_filtered[df_bayes_filtered['nation'] == nation_one]
        
        # Dropdown for selecting rb_br with placeholder option
        rb_br_one = st.selectbox("Select BR Rating for First Vehicle:", ["Select a BR..."] + sorted(df_nation_one['rb_br'].unique()))
        
        if rb_br_one != "Select a BR...":
            df_br_one = df_nation_one[df_nation_one['rb_br'] == rb_br_one]
            
            # Dropdown for selecting the first vehicle name with placeholder
            vehicle_one_name = st.selectbox("Select the First Vehicle:", ["Select a vehicle..."] + sorted(df_br_one['name'].unique()))
            
            if vehicle_one_name != "Select a vehicle...":
                vehicle_one_series = df_br_one[df_br_one['name'] == vehicle_one_name]['rb_ground_frags_per_death']

# Column 2: Second vehicle selection process
with col2:
    st.subheader("Second Vehicle Selection")
    
    # Dropdown for selecting nation with placeholder option
    nation_two = st.selectbox("Select Nation for Second Vehicle:", ["Select a nation..."] + sorted(df_bayes_filtered['nation'].unique()), key="nation_two")
    
    if nation_two != "Select a nation...":
        df_nation_two = df_bayes_filtered[df_bayes_filtered['nation'] == nation_two]
        
        # Dropdown for selecting rb_br with placeholder option
        rb_br_two = st.selectbox("Select BR Rating for Second Vehicle:", ["Select a BR..."] + sorted(df_nation_two['rb_br'].unique()), key="rb_br_two")
        
        if rb_br_two != "Select a BR...":
            df_br_two = df_nation_two[df_nation_two['rb_br'] == rb_br_two]
            
            # Dropdown for selecting the second vehicle name with placeholder
            vehicle_two_name = st.selectbox("Select the Second Vehicle:", ["Select a vehicle..."] + sorted(df_br_two['name'].unique()), key="vehicle_two_name")
            
            if vehicle_two_name != "Select a vehicle...":
                vehicle_two_series = df_br_two[df_br_two['name'] == vehicle_two_name]['rb_ground_frags_per_death']


   # Run Bayesian A/B testing and display plots
if 'vehicle_one_series' in locals() and 'vehicle_two_series' in locals():
    st.write(f"**{vehicle_one_name} vs {vehicle_two_name}**")

    # Run Bayesian A/B testing
    test_samples, control_samples, diff_samples, credible_interval = bayesian_ab_test_numeric(
        vehicle_one_series, vehicle_two_series, vehicle_one_name, vehicle_two_name
    )
    
    # Calculate means after the test samples are available
    test_mean = np.mean(test_samples)
    control_mean = np.mean(control_samples)

    # Display posterior distributions
    # st.write(f"Posterior Distributions for {vehicle_one_name} and {vehicle_two_name}:")
    # Now call the function with test_mean and control_mean as arguments
    create_posterior_plots(test_samples, control_samples, vehicle_one_name, vehicle_two_name, test_mean, control_mean)

    # Display difference distribution plot
    # st.write(f"Difference Distribution between {vehicle_one_name} and {vehicle_two_name} with Credibility Interval:")
    create_difference_plot(diff_samples, credible_interval, vehicle_one_name, vehicle_two_name)
else:
    st.write("Please select both vehicles to run the Bayesian A/B test.")


