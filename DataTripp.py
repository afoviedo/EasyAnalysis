import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Tuple, Optional
import matplotlib.ticker as ticker
import datetime
import io

# --- PLOT FUNCTION FOR DISTRIBUTIONS ---
def plot_distribution(
    data: List[float],
    variable_name: str,
    xlabel: str,
    ylabel: str,
    bins: int = 10,
    color: str = "skyblue",
    edgecolor: str = "black",
    grid_color: str = "lightgrey",
    grid_style: str = "--",
    figsize: Tuple[int, int] = (10, 6),
    font_size: int = 10,
    table_bbox: Optional[Tuple[float, float, float, float]] = (0.1, -0.45, 0.8, 0.2),
) -> None:
    """
    Generates a distribution plot combining a box plot and a histogram.
    """
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": (0.3, 0.7)}
    )

    # Calculate statistics
    stats = {
        "Min": np.min(data),
        "Max": np.max(data),
        "Q1": np.percentile(data, 25),
        "Q2": np.median(data),
        "Q3": np.percentile(data, 75),
        "Mean": np.mean(data),
        "Std": np.std(data),
        "Count": len(data),
    }

    # Box plot
    ax_box.boxplot(data, vert=False)
    ax_box.set(title=f"{variable_name}")

    # Histogram
    ax_hist.hist(data, bins=bins, color=color, edgecolor=edgecolor, linewidth=0.5)
    ax_hist.set(xlabel=xlabel, ylabel=ylabel)
    ax_hist.grid(True, color=grid_color, linestyle=grid_style)

    # Statistics table
    table_data = [[f"{v:.2f}" if isinstance(v, (int, float)) else v for v in stats.values()]]
    col_labels = list(stats.keys())
    table = ax_hist.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="upper center",
        bbox=table_bbox,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, 1.2)

    # Show plot
    st.pyplot(fig)
    return fig

# --- STREAMLIT APP ---
st.title("Data Analysis Tool")

# Header
st.write('This app enables users to upload datasets (CSV, Excel, or Parquet) and perform exploratory data analysis (EDA) with interactive visualizations. The app automatically detects numerical and categorical variables, providing relevant plots for each type.')

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your file (CSV, Excel, Parquet)", type=["csv", "xlsx", "parquet"])

if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            data = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file type!")
            data = None

        if data is not None:
            # Display data preview
            st.write("### Preview of the Uploaded Data:")
            st.dataframe(data.head())
            st.write(f'Dataset shape: {data.shape[0]} rows and {data.shape[1]} columns')

            # Select a column for analysis
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                column = st.selectbox("Select a column for analysis", data.columns)
            with col2:
                # Separate numerical and categorical data
                numeric_data = data.select_dtypes(include=[np.number])
                categorical_data = data.select_dtypes(exclude=[np.number])
                
                if column in numeric_data.columns:
                    selected_plot = st.selectbox("Select a plot type", [
                        "Histogram & Descriptive Stats", "Boxplot", "Correlation Matrix", "Pairplot", "Time Series"
                    ])
                else:
                    selected_plot = st.selectbox("Select a plot type", ["Barplot", "Pie Chart"])
                
            # --- NUMERICAL VARIABLES ---
            if column in numeric_data.columns:
                with col3:
                    # Option to filter by cohort
                    cohort_column = st.selectbox("Filter by cohort?", ["None"] + [col for col in categorical_data.columns if not np.issubdtype(data[col].dtype, np.datetime64)])

                if cohort_column != "None":
                    unique_cohorts = data[cohort_column].dropna().unique()
                    with col4:
                        selected_cohorts = st.multiselect("Select cohort values:", 
                                                          data[cohort_column].dropna().unique())
                        if selected_cohorts:
                            data_filtered = data[data[cohort_column].isin(selected_cohorts)]
                        else:
                            data_filtered = data
                else:
                    data_filtered = data
                
                if selected_plot == "Histogram & Descriptive Stats":
                    st.write("### Distribution Plot:")
                    hist_figure = plot_distribution(
                        data_filtered[column].dropna().tolist(),
                        variable_name=column,
                        xlabel=column,
                        ylabel="Frequency",
                    )
                    buf = io.BytesIO()
                    hist_figure.figure.savefig(buf, format='png', dpi=300)
                    buf.seek(0)
                    st.download_button(label="Download Plot", 
                                    data=buf, file_name=f"{selected_plot}.png", 
                                    mime="image/png")


                elif selected_plot == "Boxplot":
                    st.write("### Boxplot:")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    bp_figure = sns.boxplot(y=data_filtered[column], ax=ax, color="lightblue")
                    ax.set_ylabel(column)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    st.pyplot(fig)
                    buf = io.BytesIO()
                    bp_figure.figure.savefig(buf, format='png', dpi = 300)
                    buf.seek(0)
                    st.download_button(label="Download Plot", 
                                       data=buf, file_name=f"{selected_plot}.png", 
                                       mime="image/png")

                elif selected_plot == "Correlation Matrix":
                    st.write("### Correlation Matrix:")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    hm_figure = sns.heatmap(data_filtered[numeric_data.columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    st.pyplot(fig)
                    buf = io.BytesIO()
                    hm_figure.figure.savefig(buf, format='png', dpi = 300)
                    buf.seek(0)
                    st.download_button(label="Download Plot", 
                                       data=buf, file_name=f"{selected_plot}.png", 
                                       mime="image/png")

                elif selected_plot == "Pairplot":
                    st.write("### Pairplot:")
                    pairplot_fig = sns.pairplot(data_filtered[numeric_data.columns], height=1.5)
                    st.pyplot(pairplot_fig.figure)
                    buf = io.BytesIO()
                    pairplot_fig.figure.savefig(buf, format='png', dpi = 300)
                    buf.seek(0)
                    st.download_button(label="Download Plot", 
                                       data=buf, file_name=f"{selected_plot}.png", 
                                       mime="image/png")

                elif selected_plot == "Time Series":
                    st.write("### Time Series Plot:")
                    
                    if cohort_column != "None":
                        if selected_cohorts:
                            data_filtered = data[data[cohort_column].isin(selected_cohorts)]
                        else:
                            data_filtered = data
                    else:
                        data_filtered = data
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        date_columns = data_filtered.select_dtypes(include=[np.datetime64]).columns
                        time_series_column = st.selectbox("Select the time date column", date_columns)
                        # Filter columns that are datetime type
                        if len(date_columns) == 0:
                            st.error("No datetime columns found in the dataset.")    
                    with col2:
                        # User selects resampling frequency
                            resample_option = st.selectbox("Resample Data By:", ["Annual", "Quarterly", "Monthly", "ISO Week"])  
                    with col3:
                        # User selects aggregation method
                        aggregation_option = st.selectbox("Aggregation Method:", ["Mean", "Sum"])
                    
                        agg_func = "mean" if aggregation_option == "Mean" else "sum"
                    if np.issubdtype(data_filtered[time_series_column].dtype, np.datetime64):
                        # Convert pandas Timestamps to Python datetime
                        min_date = data_filtered[time_series_column].min().to_pydatetime()
                        max_date = data_filtered[time_series_column].max().to_pydatetime()

                        # Organizing the date selection in columns
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input("Select start date:", min_value=min_date, max_value=max_date, value=min_date)
                        with col2:
                            end_date = st.date_input("Select end date:", min_value=min_date, max_value=max_date, value=max_date)
                                        
                        # Convert back to pandas Timestamp for filtering
                        start_date = pd.Timestamp(start_date)
                        end_date = pd.Timestamp(end_date)
                        
                        # Filter data based on selected date range
                        data_filtered = data_filtered[(data_filtered[time_series_column] >= start_date) & (data_filtered[time_series_column] <= end_date)]

                        # Resample data based on selection
                        if resample_option == "Quarterly":
                            data_resampled = data_filtered.groupby(pd.Grouper(key=time_series_column, freq='Q'))[column].agg(agg_func).dropna()
                            data_resampled.index = data_resampled.index.to_period("Q").strftime("Q%q-%y")
                        elif resample_option == "Monthly":
                            data_resampled = data_filtered.groupby(pd.Grouper(key=time_series_column, freq='M'))[column].agg(agg_func).dropna()
                            data_resampled.index = data_resampled.index.strftime("%m-%y")
                        elif resample_option == "ISO Week":
                            data_resampled = data_filtered.groupby(pd.Grouper(key=time_series_column, freq='W'))[column].agg(agg_func).dropna()
                            data_resampled.index = data_resampled.index.strftime("W%U-%y")
                        else:
                            data_resampled = data_filtered.groupby(pd.Grouper(key=time_series_column, freq='Y'))[column].agg(agg_func).dropna()
                            data_resampled.index = data_resampled.index.strftime("%Y")

                        # Plot time series
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.set_title(f"{column} over time ({resample_option if resample_option != 'None' else 'Raw Data'}) - {aggregation_option}")
                        ax.set_xlabel(time_series_column)
                        ax.set_ylabel(column)

                        # Ensure proper x-axis format
                        sns.lineplot(x=data_resampled.index, y=data_resampled.values, ax=ax, marker="o", linestyle="-")
                        
                        # Improve x-label readability
                        ax.set_xticks(range(len(data_resampled.index)))
                        ax.set_xticklabels(data_resampled.index, rotation=45, ha='right')
                        
                        # Format y-axis labels dynamically
                        def human_format(num, pos):
                            if num >= 1_000_000:
                                return f'{num / 1_000_000:.1f}M'
                            elif num >= 1_000:
                                return f'{num / 1_000:.1f}K'
                            return f'{num:.0f}'
                        
                        ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
                        # Add x axis title if resample_option is Quarterly then add Quarter else add Month
                        ax.set_xlabel(f"{time_series_column} ({'Quarter' if resample_option == 'Quarterly' 
                                    else 'Month' if resample_option == 'Monthly' 
                                    else 'ISO Week' if resample_option == 'ISO Week' 
                                    else 'Year'})")
                        
                        # Remove all spines with a for loop
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        
                        st.pyplot(fig)
                        
                        # Option to download the plot
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi = 300)
                        buf.seek(0)
                        st.download_button(label="Download Plot", 
                                           data=buf, 
                                           file_name=f"{selected_plot}.png", 
                                           mime="image/png")

                    else:
                        st.error("Please select a valid datetime column for time series analysis.")
                      
            # --- CATEGORICAL VARIABLES ---
            elif column in categorical_data.columns:

                if selected_plot == "Barplot":
                    st.write("### Barplot:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Add a title to the barplot
                    ax.set_title(f"Distribution of {column}")
                    value_counts = data[column].value_counts()
                    bp_figure = sns.barplot(x = [f"{lbl[0].upper() + lbl[1:].lower()}" for lbl in value_counts.index], y = value_counts.values, ax=ax, palette="viridis", hue = value_counts.index, legend = False)
                    ax.set_xlabel(column)
                    ax.set_ylabel("Count")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    # Remove all spines with a for loop
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    st.pyplot(fig)
                    buf = io.BytesIO()
                    bp_figure.figure.savefig(buf, format='png', dpi = 300)
                    buf.seek(0)
                    st.download_button(label="Download Plot", 
                                       data=buf, file_name=f"{selected_plot}.png", 
                                       mime="image/png")

                elif selected_plot == "Pie Chart":
                    st.write("### Pie Chart:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Add a title to the pie chart
                    ax.set_title(f"Distribution of {column}")
                    value_counts = data[column].value_counts()
                    ax.pie(value_counts, labels = [f"{lbl[0].upper() + lbl[1:].lower()}" for lbl in value_counts.index], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("coolwarm", len(value_counts)))
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig)
                    buf = io.BytesIO()
                    fig.figure.savefig(buf, format='png', dpi = 300)
                    buf.seek(0)
                    st.download_button(label="Download Plot", 
                                       data=buf, file_name=f"{selected_plot}.png", 
                                       mime="image/png")

            else:
                st.error("Please select a numerical or categorical column for analysis.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.write("\n**Future Features Coming Soon:** Advanced analytics, machine learning integration, and more!")