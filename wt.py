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

title_col, space_col, logo_col = st.columns([4,1,1])

with title_col:
    st.write("""
            # War Thunder Data Science App :boom:
            **Bayesian A/B Testing, K-Means Clustering, and Statistical Techniques Applied to War Thunder Data**
            """)
    st.write('Developed by **A.C. Sanders** - also known in the War Thunder community as *DrKnoway*')

with space_col:
    st.empty() # used for padding

with logo_col:
    # st.empty()  # Create an empty space to help formmat the image location
    st.image("knoway_eye.png",
              width=200,
              )

# header for the section that generates line charts for vehicle performance over time
st.header("Vehicle Trends")

# Load and cache the DataFrame
@st.cache_data  # You can upgrade Streamlit to use @st.cache_data for newer versions
def load_data():
    return pd.read_csv("full_data.csv") # full_data is available in my Github repo

# Load the data
data = load_data()

# Make a copy of the DataFrame
df_copy = data.copy()

# Ensure 'date' column is in datetime format
df_copy['date'] = pd.to_datetime(df_copy['date'])

# Create columns for the dropdown menus
col1, col2, col3, col4 = st.columns(4)

# 1st dropdown menu for the type of vehicle - default set to ground vehicles
with col1:
    vehicle_types = df_copy['cls'].unique()
    # Set default to "Ground" if it exists in vehicle_types
    default_vehicle_type = "Ground_vehicles" if "Ground_vehicles" in vehicle_types else vehicle_types[0]
    selected_vehicle_types = st.selectbox("Vehicle Type", vehicle_types, index=list(vehicle_types).index(default_vehicle_type))

# Filter df_copy based on the selected vehicle type (cls in the dataset)
filtered_by_type_df = df_copy[df_copy['cls'] == selected_vehicle_types]

# 2nd dropdown for selecting nation
with col2:
    nations = filtered_by_type_df['nation'].unique()
    # Set default to "Germany" if it exists in nations
    default_nation = ["USA"] if "USA" in nations else [nations[0]]
    selected_nations = st.multiselect("Nation", nations, default=default_nation)

# Filter based on selected nation
filtered_by_nation_df = filtered_by_type_df[filtered_by_type_df['nation'].isin(selected_nations)]

# 3rd dropdown for BR (i.e., battle rating in the game)
with col3:
    br_values = filtered_by_nation_df['rb_br'].unique()
    # Set default to 1.0 if it exists in br_values
    default_br = [1.0] if 1.0 in br_values else [sorted(br_values)[0]]
    selected_br = st.multiselect("BR", sorted(br_values, reverse=True), default=default_br)

# Filter based on selected BR
filtered_by_br_df = filtered_by_nation_df[filtered_by_nation_df['rb_br'].isin(selected_br)]

# 4th dropdown for the metric to analyze/visualize
with col4:
    metrics = [ 
        'rb_ground_frags_per_death', 
        'rb_ground_frags_per_battle',
        'rb_win_rate',
        'rb_battles',  
        'rb_air_frags_per_death', 
        'rb_air_frags_per_battle' 
    ]
    selected_metric = st.selectbox("Metric", metrics, index=0)  # Defaults to the first metric

# 5th dropdown for Vehicle Name - filtered based on type and nation selected
vehicle_names = filtered_by_br_df['name'].unique()
selected_vehicle_names = st.multiselect("Vehicle Name", vehicle_names, default=list(vehicle_names) if vehicle_names.size > 0 else [])

# Filter the df_copy dataframe based on user selections from the dropdown menus
final_filtered_df = df_copy[
    (df_copy['cls'] == selected_vehicle_types) &
    (df_copy['nation'].isin(selected_nations)) &
    (df_copy['rb_br'].isin(selected_br)) &
    (df_copy['name'].isin(selected_vehicle_names))
]

# Sort by date and get unique dates - ensure formatted dates using dt.strftime()
unique_dates = sorted(df_copy['date'].dt.strftime('%m/%d/%y').unique())

# Add the date range select slider
date_range = st.select_slider(
    "Select date range:",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1]),
    key='date_range_slider'
)

# Convert selected start and end dates back to datetime for filtering
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter based on the date range selected
final_filtered_df = final_filtered_df[
    (final_filtered_df['date'] >= start_date) & 
    (final_filtered_df['date'] <= end_date)
]


# show row count in the filtered DataFrame - used for debugging - un-comment if needed
# st.write(f"Filtered DataFrame rows: {final_filtered_df.shape[0]}") 

# plot the rows if there is data
if final_filtered_df.shape[0] > 0:
 # Create boxplot
    fig_box = px.box(final_filtered_df, 
                     x='name',  # Categories (vehicles)
                     y=selected_metric,  # Metric
                     color='name',  # This will color the boxes by vehicle name
                     title=f"<b>Distribution of {selected_metric} by Vehicle</b>",
                     labels={'name': 'Vehicle Name', selected_metric: selected_metric})
    
    fig_box.update_layout(template='plotly')

    fig_box.update_layout(title=dict(font=dict(size=24)), 
                          xaxis_title=dict(font=dict(size=18)), 
                          yaxis_title=dict(font=dict(size=18)),
                          width=900,
                          height=600)

    # Create line plot
    fig = px.line(final_filtered_df, 
                  x='date', 
                  y=selected_metric, 
                  color='name',  # this will create separate lines for each vehicle
                  title=f"<b>Performance Over Time</b><br><span style='font-size:16px;'>Selected Metric: {selected_metric}</span>",
                  labels={'date': 'Date', selected_metric: selected_metric}, 
                  hover_data=['name', 'nation', 'rb_br'])
    
    fig.update_layout(template='plotly')

    fig.update_traces(line=dict(width=2), mode='lines+markers')  
    fig.update_layout(title=dict(font=dict(size=24)), 
                        xaxis_title=dict(font=dict(size=18)), 
                        yaxis_title=dict(font=dict(size=18)),
                        width = 900,
                        height = 600) 

    # Show the plot and fill the container width (use_container_width = True) -- stylistic/helps format the plot on the screen
    st.plotly_chart(fig, use_container_width=True) 

    # Show the boxplot
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.write("No data available") # display this message if the selections and resulting dataframe lack any data

####################################################################################################################################

# Heatmap of Aggregated Win Rates

####################################################################################################################################

st.subheader("Aggregated Win Rates by Nation")

# copy of our data
wr_df = data.copy()

# convert date column in datetime format
wr_df['date'] = pd.to_datetime(wr_df['date'])

# Get unique vehicle types from the 'cls' column
vehicle_types = wr_df['cls'].unique()

# get rid of "Fleet" from vehicle_types in selection
filtered_vehicle_types_heatmap = [vt for vt in vehicle_types if vt != "Fleet"]

# dropdown menu to select the vehicle type
default_index_heatmap = filtered_vehicle_types_heatmap.index("Ground_vehicles") if "Ground_vehicles" in filtered_vehicle_types_heatmap else 0
selected_vehicle_type = st.selectbox(
    "Select Vehicle Type:",
    options=filtered_vehicle_types_heatmap,
    index=default_index_heatmap  # default to "Ground_vehicles" otherwise the first type
)

# Sort by date and get unique dates - make sure it is formatted correctly (MM/DD/YY)
unique_dates = sorted(wr_df['date'].dt.strftime('%m/%d/%y').unique())

# date range slider - additional filtering by date
date_range = st.select_slider(
    "Select date range:",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1])
)

# convert start and end dates back to datetime for filtering
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# Filter for selected type and date
filtered_df = wr_df[
    (wr_df['cls'] == selected_vehicle_type) &
    (wr_df['date'] >= start_date) & 
    (wr_df['date'] <= end_date)
]

# make a br_range variable - this will be a more generic/broad category
filtered_df['br_range'] = np.floor(filtered_df['rb_br']).astype(int) # e.g., 1.0, 1.3, and 1.7 will all be 1 for br_range (a broad category)
agg_wr_df = filtered_df.groupby(['nation', 'br_range']).rb_win_rate.mean().reset_index()
agg_wr_pivot = agg_wr_df.pivot(index='nation', columns='br_range', values='rb_win_rate')

# create heatmap
fig_wr_heatmap = px.imshow(
    agg_wr_pivot,
    color_continuous_scale='RdBu',
    labels=dict(x='BR Range', y='Nation', color='rb_win_rate')
)

# let us show all x-axis ticks
fig_wr_heatmap.update_xaxes(tickmode='linear')

fig_wr_heatmap.update_traces(
    text=agg_wr_pivot.round(1).astype(str),  # round values to 1 decimal and convert to string
    texttemplate="%{text}",  # rounded values as text
    textfont=dict(size=10),  # font size
    hoverinfo='text'  # hover info - show the values
)

# layout updates and styling
fig_wr_heatmap.update_layout(
    template='plotly',
    width=900,   
    height=700   
)

# Show heatmap with customized dimensions
st.plotly_chart(fig_wr_heatmap, use_container_width=True)


#######################################################################################################################

# k-means clustering

#######################################################################################################################

st.header('k-Means Clustering')
st.subheader('Ranked Ground Vehicle Performance Groups')
st.write("""k-Means clustering is performed on several engagement variables like *K/D*
            and *vehicles destroyed per battle*. The algorithm clusters vehicles into one
            of three groups: **high performers**, **moderate performers**, and **low performers**.
            After clustering, an interactive scatterplot is generated showing displaying each cluster of vehicles for the variables
            K/D and Frags per Battle. The full results can be downloaded as a CSV file. 
         """)


# scatterplot function
def plot_scatter_plot(df, x_metric, y_metric, color_metric):
    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color=color_metric,
        hover_data=['name', 'cls', 'nation'],
        title="<b>Scatter Plot of K/D vs Frags per Battle Colored by Performance Cluster</b>"  # Bold title
    )
    
    fig.update_layout(
        title=dict(
            font=dict(size=20) 
        ),
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        legend_title=dict(text=color_metric, font=dict(size=14)), 
        xaxis=dict(title_font=dict(size=16)), 
        yaxis=dict(title_font=dict(size=16)),
        width = 900,
        height = 600
    )
    
    # markers
    fig.update_traces(marker=dict(size=8)) 
    
    return fig

# clustering metrics - I removed win rate  as it was causing some noise
# rare vehicles that were hardly used would have very high win rates, which sometimes skewed results
# win rates are not always the best indicator for individual vehicle performance as they depend heavily on the entire team
# to simplify things, I focused on two engagement metrics: K/D and Kills per battle
key_metrics = ['rb_ground_frags_per_death', 
               'rb_ground_frags_per_battle', 
              ]       
               # 'rb_win_rate'] # We could add this in. We might possibly restrict clustering to include non-rare vehicles

# processing function that filters, creates 'br_range' column, groups by vehicle, and segments data for the last month (30 days)
def filter_and_segment_data(df, key_metrics):
    recent_date = datetime.now() - timedelta(days=30)
    
    # filter data for the last months and drop rows with NaN in our metrics
    filtered_df = df[df['date'] >= recent_date].dropna(subset=key_metrics)
    
    # drop rows where 'cls' is 'Fleet' or 'Aviation' -- 
    filtered_df = filtered_df[~filtered_df['cls'].isin(['Fleet', 'Aviation'])]
    
    # make 'br_range' column based on 'rb_br'
    filtered_df['br_range'] = np.floor(filtered_df['rb_br']).astype(int)
    
    # Group by vehicle  and aggregate key metrics by taking the mean (from the date range of last 30 days)
    aggregated_df = filtered_df.groupby('name', as_index=False).agg({
        'rb_ground_frags_per_death': 'mean',
        'rb_ground_frags_per_battle': 'mean',
        'rb_win_rate': 'mean',
        'br_range': 'first',  # this keeps the first BR range per name
        'cls': 'first',       # Keep the first cls
        'nation': 'first'     # Keep the first nation
    })
    
    # Segment data by 'br_range'
    segmented_data = {br: aggregated_df[aggregated_df['br_range'] == br] for br in range(1, 14)}
    
    return segmented_data

# k-means function

def perform_kmeans_and_label(segmented_data, key_metrics):
    scaler = StandardScaler() #initialize a standard scaler obj
    results = {}

    for br, data in segmented_data.items():
        if not data.empty:
            # standardize data
            standardized_metrics = scaler.fit_transform(data[key_metrics])

            # Perform k-means clustering (set k = 3 to make 3 performance clusters)
            kmeans = KMeans(n_clusters=3, random_state=42)
            data['cluster'] = kmeans.fit_predict(standardized_metrics)
            
            # Get centroids - note that centroids represent the average position of each cluster in feature space
            centroids = kmeans.cluster_centers_

            # use inverse transform on centroids to get them back in original scale -- could display this?
            centroids_original_scale = scaler.inverse_transform(centroids)

            # get average value for each cluster's centroid across all key metrics - I'm going to use this to auto-assign labels
            avg_centroids = np.mean(centroids_original_scale, axis=1)

            # assign labels based on the average values of the centroids
            label_map = {}
            for i, avg_value in enumerate(avg_centroids):
                if avg_value == max(avg_centroids):
                    label_map[i] = 'high performance'
                elif avg_value == min(avg_centroids):
                    label_map[i] = 'low performance'
                else:
                    label_map[i] = 'moderate performance'

            # map clusters to these performance labels
            data['performance_label'] = data['cluster'].map(label_map)

            results[br] = data

            # we can use this to debug: let us print centroids and labels for verification
            print(f"BR {br}: Centroids (original scale)")
            for i, centroid in enumerate(centroids_original_scale):
                print(f"Cluster {i} - Centroid: {centroid}, Label: {label_map[i]}")

    return results

# Ensure the 'date' column is a datetime type
df_copy['date'] = pd.to_datetime(df_copy['date'])

# Filter data
segmented_data = filter_and_segment_data(df_copy, key_metrics)

# user input: select BR range
selected_br = st.selectbox("Select a BR range for clustering results", list(segmented_data.keys()))

# run k-means and display results
clustering_results = perform_kmeans_and_label(segmented_data, key_metrics)

# Debug: Print keys of clustering_results to ensure they are integers
print("Clustering results keys:", clustering_results.keys())

# check if selected BR data is available
selected_br_data = clustering_results.get(selected_br)

#show data and plot scatter plot
if selected_br_data is not None and not selected_br_data.empty:
    st.dataframe(selected_br_data)
    st.write("Clustering completed for BR:", selected_br)
    
    fig = plot_scatter_plot(
        selected_br_data,
        x_metric='rb_ground_frags_per_death',
        y_metric='rb_ground_frags_per_battle',
        color_metric='performance_label'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No data available for the selected BR.")

####################################################################################################################

# Bayesian Statistics

####################################################################################################################

st.header("Bayesian A/B Testing")
st.subheader("Calculate the Probability that One Vehicle (A) Has a Better K/D than Another Vehicle (B)")
st.write("""This tool tests the K/D rates between two vehicles using Bayesian statistical methods. 
            Historical data from the last 60 days is used in conjunction with non-informative priors. *Monte Carlo* simulation 
            is employed to create the posterior distributions for *K/D*, and the *probability* that the first selected vehicle's K/D is **better**
            than the second vehicle's K/D is computed. Additionally, the distribution of differences between the K/D rates are plotted along with 
            a *95% credibility interval*. 
            """)
st.write("**Please select two vehicles to run a Bayesian statistical analysis on *K/D***")

# function to run Bayesian testing on numeric or continuous data
def bayesian_ab_test_numeric(vehicle_one_series, vehicle_two_series, vehicle_one_name, vehicle_two_name, n_simulations=10000):
    # calculate sample statistics - we need these for posteriors
    test_mean, test_std = np.mean(vehicle_one_series), np.std(vehicle_one_series, ddof=1)
    control_mean, control_std = np.mean(vehicle_two_series), np.std(vehicle_two_series, ddof=1)
    test_n, control_n = len(vehicle_one_series), len(vehicle_two_series)
    
    # Priors - set these to be non-informmative gaussian priors
    mu0, s0, n0 = 0, 1, 0
    
    # posterior parameters needed for Monte Carlo simulation - non-informative priors lets the data speak for itself
    inv_vars_test = (n0 / s0**2, test_n / test_std**2)
    posterior_mean_test = np.average((mu0, test_mean), weights=inv_vars_test)
    posterior_std_test = 1 / np.sqrt(np.sum(inv_vars_test))
    
    inv_vars_control = (n0 / s0**2, control_n / control_std**2)
    posterior_mean_control = np.average((mu0, control_mean), weights=inv_vars_control)
    posterior_std_control = 1 / np.sqrt(np.sum(inv_vars_control))
    
    # Monte Carlo sampling time -- use 10,000 simulations
    test_samples = norm.rvs(loc=posterior_mean_test, scale=posterior_std_test, size=n_simulations)
    control_samples = norm.rvs(loc=posterior_mean_control, scale=posterior_std_control, size=n_simulations)
    
    # Probability that vehicle one beats vehicle two
    prob_vehicle_one_beats_vehicle_two = round(np.mean(test_samples > control_samples), 2) * 100
    
    # Credible Interval for the difference - using 95% credibile interval
    diff_samples = test_samples - control_samples
    credible_interval = np.percentile(diff_samples, [2.5, 97.5])
    
    # results - probability that A is better than B
    st.markdown(
    f"""Probability that the **{vehicle_one_name}** has a better K/D than the **{vehicle_two_name}** = 
    <span style='color: green; font-weight:bold;'>{prob_vehicle_one_beats_vehicle_two}%</span>""",
    unsafe_allow_html=True
    )

    # credibility interval
    st.write(f"95% Credibility Interval for difference: [{round(credible_interval[0],1)}, {round(credible_interval[1],1)}]")
    
    return test_samples, control_samples, diff_samples, credible_interval


# Plotting function for posterior distributions
def create_posterior_plots(test_samples, control_samples, vehicle_one_name, vehicle_two_name, test_mean, control_mean):
    # make histograms or distplots to visualize the distributions of the posterior samples
    fig_a = ff.create_distplot([test_samples, control_samples], 
                               group_labels=[vehicle_one_name, vehicle_two_name], 
                               )
    fig_a.update_traces(nbinsx = 100, autobinx = True, selector = {'type':'histogram'})
    fig_a.add_vline(x=test_samples.mean(), line_width = 3, line_dash='dash', line_color= 'hotpink', annotation_text = f'mean <br> {round(test_mean,1)}', annotation_position = 'bottom')
    fig_a.add_vline(x=control_samples.mean(), line_width = 3, line_dash='dash', line_color= 'purple', annotation_text = f'mean <br> {round(control_mean,1)}', annotation_position = 'bottom')
    fig_a.update_layout(# title_text = 'Posterior Distributions of Selected Vehicle K/D Ratios',
                        # title_font=dict(size=24, family="Inter Bold"), # I think Streamlit uses Inter font family
                        autosize = True,
                        height = 700)
    
    # show
    st.plotly_chart(fig_a, use_container_width=True)

# Plotting function for difference distribution - 
def create_difference_plot(diff_samples, credible_interval, vehicle_one_name, vehicle_two_name):
    # round the credible interval and median to 1 decimal
    credible_interval_rounded = [round(val, 1) for val in credible_interval]
    diff_samples_median = round(np.median(diff_samples), 1)

    # create plot
    fig2b = ff.create_distplot([diff_samples], 
                               group_labels=[f"{vehicle_one_name} - {vehicle_two_name}"], 
                               colors = ['aquamarine'],
    )

    fig2b.update_traces(nbinsx = 100, autobinx=True, selector = {'type':'histogram'})

    # Add vertical lines for credibility interval, 0 difference, and median (the median is the most likely lift)
    fig2b.add_vline(x=credible_interval_rounded[0], line_width=3, line_dash='dash', line_color='red', 
                    annotation_text=f'95% Lower Bound <br>{credible_interval_rounded[0]}', annotation_position='top')
    fig2b.add_vline(x=credible_interval_rounded[1], line_width=3, line_dash='dash', line_color='red', 
                    annotation_text=f'95% Upper Bound <br>{credible_interval_rounded[1]}', annotation_position='top')
    fig2b.add_vline(x=diff_samples_median, line_width=3, line_color='dodgerblue', 
                    annotation_text=f'Median <br>{diff_samples_median}', annotation_position='bottom')
    fig2b.add_vline(x=0, line_width=3, line_dash='dot', line_color='orange', 
                    annotation_text='0', annotation_position='bottom')

    fig2b.update_layout(
        # title_text=f"Distribution of Differences in K/D for 10,000 Simulations: {vehicle_one_name} K/D Minus {vehicle_two_name} K/D",
        # title_font=dict(size=24, family="Inter Bold"),
        height = 700,
        autosize=True
    )

    # show plot
    st.plotly_chart(fig2b, use_container_width=True)


# User inputs, filtering, and running the Bayesian test

# dataframe copy
df_bayes = data.copy()

# our metrics
metric = 'rb_frags_per_death'
data_type = 'numeric' 

# convert to datetime
df_bayes['date'] = pd.to_datetime(df_bayes['date'], errors='coerce')

# filter for ground vehicles only and removing nulls
df_bayes_filtered = df_bayes[(df_bayes['cls'] == 'Ground_vehicles') & df_bayes['rb_ground_frags_per_death'].notna()]

# filter for dates within the last 60 days
sixty_days_ago = datetime.now() - timedelta(days=60)
df_bayes_filtered = df_bayes_filtered[df_bayes_filtered['date'] >= sixty_days_ago]

# Create br_range variable for broad BR categories
df_bayes_filtered['br_range'] = np.floor(df_bayes_filtered['rb_br']).astype(int)

# streamlit columns used for filtering vehicle one and vehicle two
col1, col2 = st.columns(2)

# Column 1: First vehicle selection
with col1:
    st.subheader("First Vehicle Selection")
    
    # Select nation
    nation_one = st.selectbox("Select Nation for First Vehicle:", ["Select a nation..."] + sorted(df_bayes_filtered['nation'].unique()))
    
    if nation_one != "Select a nation...":
        df_nation_one = df_bayes_filtered[df_bayes_filtered['nation'] == nation_one]
        
        # Dropdown for BR range
        br_range_one = st.selectbox("Select BR Range for First Vehicle:", ["Select a BR Range..."] + sorted(df_nation_one['br_range'].unique()))
        
        if br_range_one != "Select a BR Range...":
            df_br_range_one = df_nation_one[df_nation_one['br_range'] == br_range_one]
            
            # Dropdown for selecting name
            vehicle_one_name = st.selectbox("Select the First Vehicle:", ["Select a vehicle..."] + sorted(df_br_range_one['name'].unique()))
            
            if vehicle_one_name != "Select a vehicle...":
                vehicle_one_series = df_br_range_one[df_br_range_one['name'] == vehicle_one_name]['rb_ground_frags_per_death']

# Column 2: Second vehicle selection
with col2:
    st.subheader("Second Vehicle Selection")
    
    # Select nation
    nation_two = st.selectbox("Select Nation for Second Vehicle:", ["Select a nation..."] + sorted(df_bayes_filtered['nation'].unique()), key="nation_two")
    
    if nation_two != "Select a nation...":
        df_nation_two = df_bayes_filtered[df_bayes_filtered['nation'] == nation_two]
        
        # Dropdown for BR range
        br_range_two = st.selectbox("Select BR Range for Second Vehicle:", ["Select a BR Range..."] + sorted(df_nation_two['br_range'].unique()), key="br_range_two")
        
        if br_range_two != "Select a BR Range...":
            df_br_range_two = df_nation_two[df_nation_two['br_range'] == br_range_two]
            
            # Select the second vehicle
            vehicle_two_name = st.selectbox("Select the Second Vehicle:", ["Select a vehicle..."] + sorted(df_br_range_two['name'].unique()), key="vehicle_two_name")
            
            if vehicle_two_name != "Select a vehicle...":
                vehicle_two_series = df_br_range_two[df_br_range_two['name'] == vehicle_two_name]['rb_ground_frags_per_death']


# Run Bayesian A/B testing and display plots
if 'vehicle_one_series' in locals() and 'vehicle_two_series' in locals():
    st.write(f"Selected Vehicles: the {vehicle_one_name} **versus** the {vehicle_two_name}**")

    # Run Bayesian A/B testing
    test_samples, control_samples, diff_samples, credible_interval = bayesian_ab_test_numeric(
        vehicle_one_series, vehicle_two_series, vehicle_one_name, vehicle_two_name
    )
    
    # Calculate means for posteriors
    test_mean = np.mean(test_samples)
    control_mean = np.mean(control_samples)

    # display posterior distributions
    st.subheader("Posterior Distributions of Selected Vehicle K/D Rates")
    create_posterior_plots(test_samples, control_samples, vehicle_one_name, vehicle_two_name, test_mean, control_mean)

    # display difference distribution plot
    st.subheader("Distribution of Differences in K/D for 10,000 Simulations")
    st.markdown(f"Difference calculated as **{vehicle_one_name}** K/D rate - **{vehicle_two_name}** K/D rate")
    create_difference_plot(diff_samples, credible_interval, vehicle_one_name, vehicle_two_name)
else:
    st.write("Please select both vehicles to run the Bayesian A/B test.")
