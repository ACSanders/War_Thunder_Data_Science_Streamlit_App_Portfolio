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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


title_col, space_col, logo_col = st.columns([4,1,1])

with title_col:
    st.write("""
            # War Thunder Data Science :boom:
            **Get Insights Using Player Stats, Bayesian Analyses, and k-Means Clustering**
            """)
    # st.write('Developed by **A.C. Sanders**')

with space_col:
    st.empty() # used to add padding for logo

with logo_col:
    st.image("war_thunder_stats_logo_hd.png",
              width=120,
              )

# header for the section that generates line charts for vehicle performance over time
st.header("Vehicle Trends")

# Load and cache the DataFrame
@st.cache_data 
def load_data():
    return pd.read_csv("full_data.csv") # full_data is available in my Github repo

# load the data
data = load_data()

# make a copy of the DataFrame
df_copy = data.copy()

# rename some columns for better interpretability/readability in the app
df_copy = df_copy.rename(columns = {'rb_ground_frags_per_death': 'RB Ground K/D',
                                    'rb_ground_frags_per_battle': 'RB Ground Kills per Battle',
                                    'rb_win_rate': 'RB Win Rate',
                                    'rb_battles': 'RB Battles',  
                                    'rb_air_frags_per_death': 'RB Air K/D', 
                                    'rb_air_frags_per_battle':'RB Air Kills per Battle'})

# ensure my 'date' column is in datetime format
df_copy['date'] = pd.to_datetime(df_copy['date'])

# make columns for the dropdown menus
col1, col2, col3, col4 = st.columns(4)

# 1st dropdown menu for the type of vehicle - default set to ground vehicles
with col1:
    vehicle_types = [vt for vt in df_copy['cls'].unique() if vt in ['Aviation', 'Ground_vehicles']]
    # set default to "Ground" if it exists in vehicle_types
    default_vehicle_type = "Ground_vehicles" if "Ground_vehicles" in vehicle_types else vehicle_types[0]
    selected_vehicle_types = st.selectbox("Vehicle Type", vehicle_types, index=list(vehicle_types).index(default_vehicle_type))

# filter df_copy based on the selected vehicle type (cls in the dataset)
filtered_by_type_df = df_copy[df_copy['cls'] == selected_vehicle_types]

# 2nd dropdown for selecting nation
with col2:
    nations = filtered_by_type_df['nation'].unique()
    # set default to "USA" if it exists in nations
    default_nation = ["USA"] if "USA" in nations else [nations[0]]
    selected_nations = st.multiselect("Nation", nations, default=default_nation)

# filter based on selected nation
filtered_by_nation_df = filtered_by_type_df[filtered_by_type_df['nation'].isin(selected_nations)]

# 3rd dropdown for BR (i.e., battle rating in the game)
with col3:
    br_values = filtered_by_nation_df['rb_br'].unique()
    
    # check if br_values is not empty
    if br_values.size > 0:  # Check if br_values has values
        # set default to max if it is in br_values - should currently be 12.0
        max_br = float(np.nanmax(br_values))
        # default_br = [1.0] if 1.0 in br_values else [sorted(br_values)[0]] # old code that set it to 1.0 default
        default_br = [max_br]
    else:
        default_br = []  # Or we can set to another default (empty list or some value)
    
    selected_br = st.multiselect("BR", sorted(br_values, reverse=True), default=default_br)

# filter based on selected BR
filtered_by_br_df = filtered_by_nation_df[filtered_by_nation_df['rb_br'].isin(selected_br)]

# 4th dropdown for the metric
with col4:
    metrics = [ 
        'RB Ground K/D', 
        'RB Ground Kills per Battle',
        'RB Win Rate',
        'RB Battles',  
        'RB Air K/D', 
        'RB Air Kills per Battle' 
    ]
    selected_metric = st.selectbox("Metric", metrics, index=0)  # this should default to the first metric

# 5th dropdown for  vehicle name - filtered based on type and nation
vehicle_names = filtered_by_br_df['name'].unique()
selected_vehicle_names = st.multiselect("Vehicle Name", vehicle_names, default=list(vehicle_names) if vehicle_names.size > 0 else [])

# filter the df_copy based on selections
final_filtered_df = df_copy[
    (df_copy['cls'] == selected_vehicle_types) &
    (df_copy['nation'].isin(selected_nations)) &
    (df_copy['rb_br'].isin(selected_br)) &
    (df_copy['name'].isin(selected_vehicle_names))
]

# sort by date - get unique dates - format using dt.strftime()
unique_dates = df_copy['date'].sort_values().dt.strftime('%m/%d/%y').unique()

#date range slider
date_range = st.select_slider(
    "Select date range:",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1]),
    key='date_range_slider'
)

# make start and end dates datetime
start_date, end_date = pd.to_datetime(date_range[0], format='%m/%d/%y'), pd.to_datetime(date_range[1])

# date range filtering
final_filtered_df = final_filtered_df[
    (final_filtered_df['date'] >= start_date) & 
    (final_filtered_df['date'] <= end_date)
]

# plot the rows if there is data
if final_filtered_df.shape[0] > 0:
 # make boxplot
    fig_box = px.box(final_filtered_df, 
                     x='name',  # the categories - vehicles
                     y=selected_metric,  # metric
                     color='name',  # color the boxes by vehicle
                     # title=f"<b>Distribution of {selected_metric} by Vehicle</b>",
                     labels={'name': 'Vehicle Name', selected_metric: selected_metric})

    # additional updates to help this look better on phones
    fig_box.update_layout(template = 'plotly',
                          # title=dict(font=dict(size=16), pad=dict(t=100)), 
                          xaxis_title=dict(font=dict(size=12)), 
                          yaxis_title=dict(font=dict(size=12)),
                          margin=dict(l=10, r=10, t=40, b=50),  # small margins -- should look better on phones
                          font=dict(size=10),  # smaller font
                          # dragmode='pan',
                          legend=dict(orientation='h',
                                      yanchor='top',
                                      y=-0.35,
                                      xanchor='center',
                                      x=0.5 
                                     )
    )

    st.write('**Performance Over Time**')
    st.write(f'Selected metric: **{selected_metric}**')
    # line plot
    fig = px.line(final_filtered_df, 
                  x='date', 
                  y=selected_metric, 
                  color='name',  # separate lines for each vehicle
                  # title=f"<b>Performance Over Time</b><br><span style='font-size:16px;'>Selected Metric: {selected_metric}</span>",
                  labels={'date': 'Date', selected_metric: selected_metric}, 
                  hover_data=['name', 'nation', 'rb_br'])
    
    fig.update_traces(line=dict(width=2), mode='lines+markers')  # markers - check line width?

    fig.update_layout(
        template='plotly',
        # title=dict(font=dict(size=16), pad=dict(t=100)),
        xaxis_title=dict(font=dict(size=12)),
        yaxis_title=dict(font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=10),  # margins -- should look better on phone when small
        font=dict(size=10),
        # dragmode='pan',  # panning
        legend=dict(orientation='h',
                    yanchor='top',
                    y=-0.1,
                    xanchor='center',
                    x=0.5 
                    )
    ) 

    # Show lines chart
    st.plotly_chart(fig, use_container_width=True) 

    st.write(f'**Distribution of {selected_metric} by Vehicle**')
    # Show boxplot
    st.plotly_chart(fig_box, use_container_width=True)

else:
    st.write("No data available") # display this if selections and dataframe lack data

st.divider()

####################################################################################################################################

# Nation Score Card Stats - K/D

####################################################################################################################################

st.header("Nation Stats")

# copy of our data
wr_df = data.copy()

# convert date column in datetime format
wr_df['date'] = pd.to_datetime(wr_df['date'])

# rename columns
wr_df = wr_df.rename(columns = {'rb_ground_frags_per_death': 'RB Ground K/D',
                                    'rb_ground_frags_per_battle': 'RB Ground Kills per Battle',
                                    'rb_win_rate': 'RB Win Rate',
                                    'rb_battles': 'RB Battles',  
                                    'rb_air_frags_per_death': 'RB Air K/D', 
                                    'rb_air_frags_per_battle':'RB Air Kills per Battle'})

# Get unique vehicle types from the 'cls' column
vehicle_types = wr_df['cls'].unique()

# get rid of "Fleet" from vehicle_types in selection -- I don't get much data for naval
filtered_vehicle_types_heatmap = [vt for vt in vehicle_types if vt != "Fleet"]

# dropdown menu vehicle type
default_index_heatmap = filtered_vehicle_types_heatmap.index("Ground_vehicles") if "Ground_vehicles" in filtered_vehicle_types_heatmap else 0
selected_vehicle_type = st.selectbox(
    "Select Vehicle Type:",
    options=filtered_vehicle_types_heatmap,
    index=default_index_heatmap  # default "Ground_vehicles"
)

# Sort by date and make sure it is formatted (MM/DD/YY) using dt.strftime()
unique_dates = df_copy['date'].sort_values().dt.strftime('%m/%d/%y').unique()

# date range slider - additional filter
date_range = st.select_slider(
    "Select date range:",
    options=unique_dates,
    value=(unique_dates[0], unique_dates[-1])
)

# convert start and end to datetime
start_date, end_date = pd.to_datetime(date_range[0], format='%m/%d/%y'), pd.to_datetime(date_range[1])

# Filter for type and date
filtered_df = wr_df[
    (wr_df['cls'] == selected_vehicle_type) &
    (wr_df['date'] >= start_date) & 
    (wr_df['date'] <= end_date)
]

# need to drop a few rows where the vehicles don't have values for the rb_br column
filtered_df = filtered_df[filtered_df['rb_br'].notnull()]

# make a br_range variable - this will be a more generic/broad category (too granular causes issues and takes up resources)
filtered_df['br_range'] = np.floor(filtered_df['rb_br']).astype(int) # e.g., 1.0, 1.3, and 1.7 will all be 1 for br_range (a broad category)
agg_wr_df = filtered_df.groupby(['nation', 'br_range'])['RB Win Rate'].mean().reset_index()
agg_wr_pivot = agg_wr_df.pivot(index='nation', columns='br_range', values='RB Win Rate')

##### nation K/D score cards (delta vs baseline) 
st.subheader("Nation K/D")

kd_col = None
if selected_vehicle_type == "Ground_vehicles":
    kd_col = "RB Ground K/D"
elif selected_vehicle_type == "Aviation":
    kd_col = "RB Air K/D"

if kd_col and kd_col in filtered_df.columns and not filtered_df.empty:
    # BR choices come from the *already* date+type filtered frame
    br_choices = (
        filtered_df["rb_br"]
        .dropna()
        .unique()
    )
    if br_choices.size:
        # sort high → low, default to MAX BR
        br_choices = sorted(map(float, br_choices), reverse=True)
        selected_br_for_kd = st.selectbox(
            "Select BR for K/D score cards (does not affect heatmap):",
            options=br_choices,
            index=0,  # highest BR first
            format_func=lambda x: f"{x:.1f}"
        )

        st.caption("**Note:** BR filter above applies only to the K/D score cards. The heatmap reflects all BRs.")

        # Filter ONLY for the chosen BR for the score cards
        kd_df = filtered_df.loc[filtered_df["rb_br"] == float(selected_br_for_kd), ["nation", kd_col]].copy()
        kd_df[kd_col] = pd.to_numeric(kd_df[kd_col], errors="coerce")
        kd_df = kd_df.dropna(subset=[kd_col, "nation"])

        if kd_df.empty:
            st.info(f"No K/D data for BR {selected_br_for_kd:.1f} with current filters.")
        else:
            # baseline across all nations for this BR
            baseline = kd_df[kd_col].mean()

            # mean K/D for this BR
            nation_kd = (
                kd_df.groupby("nation", sort=False, as_index=False)[kd_col]
                .mean()
                .rename(columns={kd_col: "KD"})
                .dropna(subset=["KD"])
            )

            # baseline k/d all nations
            st.info(
                f"**Baseline K/D (all nations, BR {selected_br_for_kd:.1f})** "
                f"for {selected_vehicle_type.replace('_',' ')} "
                f"from {start_date.date()} to {end_date.date()}: **{baseline:.3f}**"
            )

            # two rows of 5 score cards - I'm capping it at 10 if there's more
            nation_kd = nation_kd.head(10)
            row1 = st.columns(5)
            row2 = st.columns(5)

            for i, row in enumerate(nation_kd.itertuples(index=False), start=0):
                kd_val = round(row.KD, 3)
                delta_val = row.KD - baseline                     # >0 → green arrow; <0 → red
                delta_str = f"{delta_val:+.3f}"

                if i < 5:
                    with row1[i]:
                        st.metric(label=row.nation, value=kd_val, delta=delta_str)
                elif i < 10:
                    with row2[i - 5]:
                        st.metric(label=row.nation, value=kd_val, delta=delta_str)
    else:
        st.info("No BR values available for the selected date range and vehicle type.")
else:
    st.info("Select Ground or Aviation to view nation K/D score cards.")

st.caption(
    "**Arrows:** Green = above baseline K/D, Red = below."
)

st.divider()

####################################################################################################################################

# World Map of Battles

####################################################################################################################################

# ==== Battles by Nation (same BR as K/D cards)
st.subheader(f"Total Vehicle Battles by Nation at BR {selected_br_for_kd:.1f}")

battles_col = "RB Battles"
if battles_col not in filtered_df.columns:
    st.info("No 'RB Battles' column available for the map.")
else:
    # same slice as K/D cards: date + vehicle type + BR
    map_df = filtered_df.loc[
        filtered_df["rb_br"] == float(selected_br_for_kd),
        ["nation", battles_col]
    ].dropna(subset=["nation"])

    if map_df.empty or map_df[battles_col].sum() == 0:
        st.info(f"No battle data for BR {selected_br_for_kd:.1f} with current filters.")
    else:
        # sum battles per nation
        battles_df = (map_df
            .groupby("nation", as_index=False)[battles_col].sum()
            .rename(columns={battles_col: "Battles"}))

        # Map game nations -> ISO-3 (USSR -> Russia)
        nation_to_iso3 = {
            "USA": "USA", "United States": "USA",
            "USSR": "RUS", "Russia": "RUS",
            "Germany": "DEU",
            "Britain": "GBR", "Great Britain": "GBR", "UK": "GBR",
            "Japan": "JPN", "Italy": "ITA", "France": "FRA",
            "China": "CHN", "Sweden": "SWE", "Israel": "ISR",
        }
        battles_df["iso_alpha"] = battles_df["nation"].map(nation_to_iso3)
        battles_df = battles_df.dropna(subset=["iso_alpha"])

        fig_map = px.choropleth(
            battles_df,
            locations="iso_alpha",
            color="Battles",
            color_continuous_scale="Blues",
            hover_name="nation",
            projection="natural earth",
        )
        fig_map.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_colorbar=dict(title="Battles"),
            title=dict(text=f"BR {selected_br_for_kd:.1f}", x=0.5)
        )
        fig_map.update_traces(hovertemplate="<b>%{hovertext}</b><br>Battles: %{z:,}<extra></extra>")
        st.plotly_chart(fig_map, use_container_width=True)

        st.caption("Uses the same **BR**, vehicle type, and date range as the K/D score cards. The win-rate heatmap maintains all BRs for the selected date range.")


st.divider()

####################################################################################################################################

# Win Rate Heatmap

####################################################################################################################################

st.subheader("Nation Win Rate Heatmap")

# create heatmap
fig_wr_heatmap = px.imshow(
    agg_wr_pivot,
    color_continuous_scale='RdBu',
    labels=dict(x='BR Range', y='Nation', color='RB Win Rate')
)

# let us show all x-axis ticks
fig_wr_heatmap.update_xaxes(tickmode='linear')

fig_wr_heatmap.update_traces(
    showscale = False, # this takes up room on mobile -- experiment with removing it
    text=agg_wr_pivot.round(1).astype(str),  # round values to 1 decimal and convert to string
    texttemplate="%{text}",  # rounded values as text
    textfont=dict(size=8),  # smaller - adjust font size from 10 to 8
    hoverinfo='text'  # hover info - show the values
)

fig_wr_heatmap.update_layout(
    template='plotly',
    autosize = True, # attempt to make this look better on phones
    # dragmode = 'pan',
    # xaxis=dict(fixedrange=False),  # horiz zoom
    # yaxis=dict(fixedrange=False),  # vert zoom
    # width=700,   
    # height=500,
    margin=dict(l=10, r=10, t=30, b=10),  # attempt to adjust margin -- to make visually better on phones
    font=dict(size=8), # make font small
    coloraxis_colorbar=dict(
        orientation='h', 
        yanchor='bottom',  
        y=-0.4, 
        xanchor='center',  
        x=0.5)
)

# Show heatmap
st.plotly_chart(fig_wr_heatmap, use_container_width=True)

st.divider()

#######################################################################################################################

# k-means clustering

#######################################################################################################################

st.header('k-Means Clustering')
st.subheader('Ranked Ground Vehicle Performance Groups')

with st.popover("ℹ About K-Means Clustering"):
    st.markdown("""
                k-Means clustering is performed on several engagement variables like *K/D*
                and *vehicles destroyed per battle*. The algorithm identifies and clusters vehicles into one
                of three performance groups: 
                1) **high performers**
                2) **moderate performers**
                3) **low performers**
    """)

# scatter plot function
def plot_scatter_plot(df, x_metric, y_metric, color_metric):
    scatter_fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color=color_metric,
        hover_data=['name', 'cls', 'nation'],
        # add OLS regression line
        trendline = "ols",
        # apply it to the overall dataset and not the segments
        trendline_scope = "overall",
        # title="<b>Scatter Plot of K/D vs Win Rate Colored by Performance Cluster</b>"
    )
    
    scatter_fig.update_layout(
        template="plotly",
        # title=dict(font=dict(size=16), pad=dict(t=100)), 
        xaxis=dict(title=x_metric, title_font=dict(size=12)),
        yaxis=dict(title=y_metric, title_font=dict(size=12)),
        legend_title=dict(text=color_metric, font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=10),
        # dragmode="pan",  # panning
        legend=dict(orientation='h',
                    yanchor='top',
                    y=-0.1,
                    xanchor='center',
                    x=0.5 
                    )
    )
    
    # markers
    scatter_fig.update_traces(marker=dict(size=6))
    
    return scatter_fig

# the key metrics used for clustering - we can update this
key_metrics = ['RB Ground K/D', 'RB Ground Kills per Battle']

# cacheing -- ml and analyses take up resources - cache helps
@st.cache_data
def filter_and_segment_data(df, key_metrics):
    recent_date = datetime.now() - timedelta(days=30)
    # filter data for last 30 days and drop rows with NaN in key metrics - including win rate
    filtered_df = df[df['date'] >= recent_date].dropna(subset=key_metrics + ['RB Win Rate'])
    filtered_df = filtered_df[~filtered_df['cls'].isin(['Fleet', 'Aviation'])]
    filtered_df['br_range'] = np.floor(filtered_df['rb_br']).astype(int)
    
    # group by vehicle and aggregate metrics
    aggregated_df = filtered_df.groupby('name', as_index=False).agg({
        'RB Ground K/D': 'mean',
        'RB Ground Kills per Battle': 'mean',
        'RB Win Rate': 'mean',
        'br_range': 'first',
        'cls': 'first',
        'nation': 'first'
    })
    
    return aggregated_df

# k-means clustering on selected BR range -- note that I'm creating broad BR categories: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
def perform_kmeans_and_label(data, key_metrics):
    scaler = StandardScaler() 
    standardized_metrics = scaler.fit_transform(data[key_metrics])
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(standardized_metrics)
    
    # centroids
    centroids = kmeans.cluster_centers_
    centroids_original_scale = scaler.inverse_transform(centroids)
    avg_centroids = np.mean(centroids_original_scale, axis=1)
    
    # map labels
    label_map = {i: 'high performance' if avg_centroids[i] == max(avg_centroids)
                 else 'low performance' if avg_centroids[i] == min(avg_centroids)
                 else 'moderate performance' for i in range(3)}
    
    data['performance_label'] = data['cluster'].map(label_map)
    
    return data

# make date datetime format
df_copy['date'] = pd.to_datetime(df_copy['date'])

# filter
aggregated_df = filter_and_segment_data(df_copy, key_metrics)

# user input for br
selected_br = st.selectbox("Select a BR range for clustering results", list(range(1, 14)))

# filter 
selected_br_data = aggregated_df[aggregated_df['br_range'] == selected_br]

if not selected_br_data.empty:
    # k-means clustering only for selected BR
    clustering_results = perform_kmeans_and_label(selected_br_data, key_metrics)
    
    # display results
    st.dataframe(clustering_results)
    st.success(f"Clustering completed for BR {selected_br}.", icon="✅")

    st.divider()

    st.header(f"K/D and Win Rate for BR {selected_br} Clusters with Regression Line")

    # make scatter plots of vehicles
    scatter_fig = plot_scatter_plot(
        clustering_results,
        x_metric='RB Ground K/D',
        y_metric='RB Win Rate', 
        color_metric='performance_label'
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

else:
    st.write(f"No data available for BR {selected_br}.")

# regression info
trendline_results = px.get_trendline_results(scatter_fig)
if not trendline_results.empty:
    px_fit_results = trendline_results.iloc[0]["px_fit_results"]

    st.subheader("Linear Regression")
    st.caption(f"BR **{selected_br}** · Trendline from scatter above")

    # Pull core stats
    r2 = float(px_fit_results.rsquared)
    slope = float(np.asarray(px_fit_results.params)[1])  # assuming const + 1 predictor
    pval_slope = float(np.asarray(px_fit_results.pvalues)[1])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{r2:.3f}")
    with col2:
        st.metric("Slope", f"{slope:.3f}")
    with col3:
        st.metric("Slope p-value", f"{pval_slope:.3f}")

    with st.popover("💡 How to read this"):
        st.markdown("""
        - **R²**: % of variation explained by K/D for this BR.
        - **Slope**: change in outcome per 1 unit increase in K/D.
        - **p-value**: probability the slope is due to chance.
        """)

st.caption("Points represent vehicles and colors indicate performance clusters.")

st.divider()


####################################################################################################################

# Bayesian Statistics

####################################################################################################################

st.header("Bayesian A/B Testing")
st.subheader("Probability that One Nation (A) Has a Better Win Rate than Another Nation (B)")

with st.popover("ℹ About Bayesian A/B Testing"):
    st.markdown("""
    This analysis uses a **Bayesian approach** with non-informative priors and Monte Carlo
    simulation to estimate the probability that one nation outperforms the other.

    **The Results:**
    - **Posterior distributions** for each nation's win rate
    - **Lift distribution** = First nation (A) minus second nation (B) win rates
    - **Probability Test > Control** = proportion of simulated draws where lift > 0
    - **95% Credible Interval** = range where the true lift or difference lies with 95% probability

    Bayesian probabilities tell you *how likely* one nation has a better win rate.
    """)

st.write("**Select two nations and a BR to run a Bayesian analysis on *win rates***")

# function to run Bayesian testing on numeric or continuous data
# I originally developed this for test and control groups for A/B experiments and I've retained some of these names for variables
def bayesian_ab_test_numeric(nation_one_series, nation_two_series, nation_one, nation_two, n_simulations=5000):
    # calculate sample statistics - we need these for posteriors
    test_mean, test_std = np.mean(nation_one_series), np.std(nation_one_series, ddof=1)
    control_mean, control_std = np.mean(nation_two_series), np.std(nation_two_series, ddof=1)
    test_n, control_n = len(nation_one_series), len(nation_two_series)
    
    # priors - set these to be non-informmative gaussian priors
    mu0, s0, n0 = 0, 1, 0
    
    # posterior parameters and precision needed for Monte Carlo simulation - non-informative priors lets the data speak for itself
    inv_vars_test = (n0 / s0**2, test_n / test_std**2)
    posterior_mean_test = np.average((mu0, test_mean), weights=inv_vars_test)
    posterior_std_test = 1 / np.sqrt(np.sum(inv_vars_test))
    
    inv_vars_control = (n0 / s0**2, control_n / control_std**2)
    posterior_mean_control = np.average((mu0, control_mean), weights=inv_vars_control)
    posterior_std_control = 1 / np.sqrt(np.sum(inv_vars_control))
    
    # Monte Carlo sampling -- use 10,000 simulations
    test_samples = norm.rvs(loc=posterior_mean_test, scale=posterior_std_test, size=n_simulations)
    control_samples = norm.rvs(loc=posterior_mean_control, scale=posterior_std_control, size=n_simulations)

    # summaries
    prob_vehicle_one_beats_vehicle_two = 100.0 * float(np.mean(test_samples > control_samples))
    diff_samples = test_samples - control_samples
    credible_interval = np.percentile(diff_samples, [2.5, 97.5])
    median_lift = float(np.median(diff_samples))

    st.subheader("Bayesian A/B Test Summary")
    
    c1, c2, c3 = st.columns(3)

    delta_for_metric = '+' if prob_vehicle_one_beats_vehicle_two > 50 else '-'
    with c1:
        st.metric(
            label=f"Probability({nation_one} win rate > {nation_two})",
            value=f"{prob_vehicle_one_beats_vehicle_two:.1f}%",
            delta=delta_for_metric
        )

    lift_delta = f"{median_lift:+.1f}%"
    with c2:
        st.metric("Most Likely Lift", f"{median_lift:.1f}%", delta=lift_delta)

    with c3:
        st.metric("95% Credible Interval", f"{credible_interval[0]:.1f} to {credible_interval[1]:.1f}")

    st.caption(
    f"Lift = {nation_one} − {nation_two} (difference in win rate). "
    "Interval is 95% credible range."
    )   
    
    return test_samples, control_samples, diff_samples, credible_interval


# plotting function for posterior distributions
def create_posterior_plots(test_samples, control_samples, vehicle_one_name, vehicle_two_name, test_mean, control_mean):
    # make histograms or distplots for distributions of posterior samples
    fig_a = ff.create_distplot([test_samples, control_samples], 
                               group_labels=[vehicle_one_name, vehicle_two_name], 
                               )
    fig_a.update_traces(nbinsx = 100, autobinx = True, selector = {'type':'histogram'})
    fig_a.add_vline(x=test_samples.mean(), line_width = 3, line_dash='dash', line_color= 'hotpink', annotation_text = f'mean <br> {round(test_mean,1)}', annotation_position = 'bottom')
    fig_a.add_vline(x=control_samples.mean(), line_width = 3, line_dash='dash', line_color= 'purple', annotation_text = f'mean <br> {round(control_mean,1)}', annotation_position = 'bottom')
    
    # updates to help this look better on phones
    fig_a.update_layout(
        autosize=True, 
        height=700,
        # title=dict(font=dict(size=16)),
        xaxis_title=dict(font=dict(size=12)),
        yaxis_title=dict(font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=30), 
        font=dict(size=10), 
        # dragmode="pan",  # panning - should help on mobile
        legend=dict(orientation='h',
                    yanchor='top',
                    y=-0.1,
                    xanchor='center',
                    x=0.5 
                    )
    )
    
    # show
    st.plotly_chart(fig_a, use_container_width=True)
    return fig_a

# plotting function for difference distribution - 
def create_difference_plot(diff_samples, credible_interval, vehicle_one_name, vehicle_two_name):
    # round credible interval and median to 1 decimal
    credible_interval_rounded = [round(val, 1) for val in credible_interval]
    diff_samples_median = round(np.median(diff_samples), 1)

    # make plot
    fig2b = ff.create_distplot([diff_samples], 
                               group_labels=[f"{vehicle_one_name} - {vehicle_two_name}"], 
                               colors = ['aquamarine'],
    )

    fig2b.update_traces(nbinsx = 100, autobinx=True, selector = {'type':'histogram'})

    # vertical lines for credibility interval, 0 difference, and median (the median is the most likely lift)
    fig2b.add_vline(x=credible_interval_rounded[0], line_width=3, line_dash='dash', line_color='red', 
                    annotation_text=f'95% Lower Bound <br>{credible_interval_rounded[0]}', annotation_position='top')
    fig2b.add_vline(x=credible_interval_rounded[1], line_width=3, line_dash='dash', line_color='red', 
                    annotation_text=f'95% Upper Bound <br>{credible_interval_rounded[1]}', annotation_position='top')
    fig2b.add_vline(x=diff_samples_median, line_width=3, line_color='dodgerblue', 
                    annotation_text=f'Median <br>{diff_samples_median}', annotation_position='bottom')
    fig2b.add_vline(x=0, line_width=3, line_dash='dot', line_color='orange', 
                    annotation_text='0', annotation_position='bottom')

    # layout
    fig2b.update_layout(
        height=700,
        autosize=True,
        # title=dict(font=dict(size=16)), 
        xaxis_title=dict(font=dict(size=12)),
        yaxis_title=dict(font=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=30), 
        font=dict(size=10),
        # dragmode="pan",
        legend=dict(orientation='h',
                    yanchor='top',
                    y=-0.1,
                    xanchor='center',
                    x=0.5 
                    )
    )

    # show
    st.plotly_chart(fig2b, use_container_width=True)
    return fig2b


# User inputs, filtering, and running the Bayesian test #

# create copy
df_bayes = data.copy()

# make df for our bayesian data
df_bayes = df_bayes.rename(columns = {'rb_ground_frags_per_death': 'RB Ground K/D',
                                    'rb_ground_frags_per_battle': 'RB Ground Kills per Battle',
                                    'rb_win_rate': 'RB Win Rate',
                                    'rb_battles': 'RB Battles',  
                                    'rb_air_frags_per_death': 'RB Air K/D', 
                                    'rb_air_frags_per_battle':'RB Air Kills per Battle'})

# ground vehicles only and remove nulls in 'rb_win_rate'
df_bayes_filtered = df_bayes[(df_bayes['cls'] == 'Ground_vehicles') & df_bayes['RB Win Rate'].notna()]

# use datetime and filter for the last 60 days
df_bayes_filtered['date'] = pd.to_datetime(df_bayes_filtered['date'], errors='coerce')
sixty_days_ago = datetime.now() - timedelta(days=60)
df_bayes_filtered = df_bayes_filtered[df_bayes_filtered['date'] >= sixty_days_ago]

# make our 'br_range' variable for broad BR categories
df_bayes_filtered['br_range'] = np.floor(df_bayes_filtered['rb_br']).astype(int)

# input for BR range
selected_br_range = st.selectbox("Select BR Range:", sorted(df_bayes_filtered['br_range'].unique()))
df_bayes_filtered = df_bayes_filtered[df_bayes_filtered['br_range'] == selected_br_range]

# columns for nation selection
col1, col2 = st.columns(2)

# first nation
with col1:
    st.subheader("Nation One Selection")
    nation_one = st.selectbox("Select Nation for First Group:", sorted(df_bayes_filtered['nation'].unique()))
 
# Second
with col2:
    st.subheader("Nation Two Selection")
    nation_two = st.selectbox("Select Nation for Second Group:", sorted(df_bayes_filtered['nation'].unique()), key="nation_two")

st.write(":point_down: Click the button below to run the Bayesian A/B test on win rates")

# Run Bayesian A/B testing
if nation_one and nation_two:
    # button -- Bayesian A/B test
    if st.button("Perform Bayesian Analysis"):
        st.write(f"Comparing **{nation_one}** vs **{nation_two}** for BR Range: **{selected_br_range}**")
        
        # filter based on nations select
        nation_one_series = df_bayes_filtered[df_bayes_filtered['nation'] == nation_one]['RB Win Rate']
        nation_two_series = df_bayes_filtered[df_bayes_filtered['nation'] == nation_two]['RB Win Rate']

        # Go Bayesian
        test_samples, control_samples, diff_samples, credible_interval = bayesian_ab_test_numeric(
            nation_one_series, nation_two_series, nation_one, nation_two
        )
        
        # means for posteriors
        test_mean = np.mean(test_samples)
        control_mean = np.mean(control_samples)

        # Posterior distributions
        st.subheader(f"Posterior Distributions of Win Rates for {nation_one} and {nation_two}")
        fig_a = create_posterior_plots(test_samples, control_samples, nation_one, nation_two, test_mean, control_mean)

        # Difference distribution plot
        st.subheader("Distribution of Win Rate Differences from 5,000 Simulations")
        st.markdown(f"Difference calculated as **{nation_one}** win rate - **{nation_two}** win rate")
        fig2b = create_difference_plot(diff_samples, credible_interval, nation_one, nation_two)

        st.success(f"Bayesian analysis complete.", icon="✅")

st.divider()

st.write('2025 | Developed and maintained by **A. C. Sanders** | [adamsandersc@gmail.com](mailto:adamsandersc@gmail.com)')