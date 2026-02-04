"""
Master Plan - User Acquisition Analysis Web App

A Streamlit web application for analyzing mobile app user acquisition performance.
Analyzes Retention Rate, LTV, DAU projections, Revenue, and ROAS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from functools import reduce
import io

# =============================================================================
# Custom implementations (replacing theseus_growth for cloud compatibility)
# =============================================================================

class TheseusGrowth:
    """Custom implementation of theseus_growth functions for cloud deployment."""
    
    def create_profile(self, days, retention_values, profile_max=181):
        """Create a retention profile dictionary."""
        # Normalize retention values to 0-1 range if they're percentages
        normalized = [v / 100 if v > 1 else v for v in retention_values]
        
        # Create profile dict mapping day to retention
        profile = {}
        for d, r in zip(days, normalized):
            profile[d] = r
        
        # Interpolate for all days
        full_profile = {}
        for day in range(1, profile_max + 1):
            if day in profile:
                full_profile[day] = profile[day]
            else:
                # Linear interpolation between known points
                lower_days = [d for d in days if d < day]
                upper_days = [d for d in days if d > day]
                
                if not lower_days:
                    full_profile[day] = profile[min(days)]
                elif not upper_days:
                    full_profile[day] = profile[max(days)]
                else:
                    lower_day = max(lower_days)
                    upper_day = min(upper_days)
                    lower_val = profile[lower_day]
                    upper_val = profile[upper_day]
                    # Linear interpolation
                    ratio = (day - lower_day) / (upper_day - lower_day)
                    full_profile[day] = lower_val + ratio * (upper_val - lower_val)
        
        return full_profile
    
    def project_cohorted_DAU(self, profile, periods, cohorts, start_date=1):
        """Project DAU for cohorts based on retention profile."""
        num_cohorts = len(cohorts)
        
        # Create DataFrame with cohorts as rows and days as columns (including Day 0)
        columns = [str(i) for i in range(0, periods + 1)]  # Start from 0
        dau_df = pd.DataFrame(0.0, index=range(num_cohorts), columns=columns)
        
        for cohort_idx, cohort_size in enumerate(cohorts):
            cohort_start = start_date + cohort_idx
            
            for day in range(0, periods + 1):  # Start from 0
                days_since_install = day - cohort_start + 1
                
                if days_since_install == 0:
                    # Day 0 = install day = 100% of cohort
                    dau_df.iloc[cohort_idx, day] = cohort_size
                elif days_since_install >= 1 and days_since_install in profile:
                    retention = profile[days_since_install]
                    dau_df.iloc[cohort_idx, day] = cohort_size * retention
                elif days_since_install >= 1:
                    # Use last known retention for days beyond profile
                    max_profile_day = max(profile.keys())
                    if days_since_install > max_profile_day:
                        dau_df.iloc[cohort_idx, day] = cohort_size * profile[max_profile_day]
        
        return dau_df
    
    def DAU_total(self, forward_DAU):
        """Sum DAU across all cohorts."""
        total = forward_DAU.sum(axis=0).to_frame().T
        total.index = ['total']
        return total
    
    def combine_DAU(self, DAU_totals, labels):
        """Combine multiple DAU totals into one DataFrame."""
        combined = pd.concat(DAU_totals, axis=0)
        combined.index = labels
        return combined


# Initialize custom theseus instance
th = TheseusGrowth()

# =============================================================================
# Configuration
# =============================================================================
PROFILE_MAX_DAYS = 361  # Extend to D360
RETENTION_DAYS = [1, 3, 7, 14, 30, 60]

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Master Plan - UA Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Helper Functions
# =============================================================================

def power_function(x, a, b):
    """Power function for retention rate fitting: y = a * x^b"""
    return a * np.power(x, b)


def log_function(x, a, b):
    """Logarithmic function for LTV fitting: y = a * ln(x) + b"""
    return a * np.log(x) + b


def create_retention_profile(days_np, rr_values, profile_max=PROFILE_MAX_DAYS):
    """Fit power regression to retention data and create full profile."""
    params, _ = curve_fit(power_function, days_np[:len(rr_values)], rr_values)
    a, b = params
    
    new_days = np.arange(1, profile_max)
    predicted_values = power_function(new_days, a, b)
    
    result_df = pd.DataFrame({'day': new_days, 'rr': predicted_values})
    day_zero = pd.DataFrame({'day': [0], 'rr': [1.0]})
    result_df = pd.concat([result_df, day_zero]).sort_values('day').reset_index(drop=True)
    
    return result_df


def create_ltv_curve(days_np, ltv_values, ltv_d0, profile_max=PROFILE_MAX_DAYS):
    """Fit power-law regression to LTV data (ROAS model approach).
    
    Uses the formula: LTV = a × d^b
    - Excludes D1 from fitting (uses D2+ only)
    - Applies uncertainty discount for predictions beyond observed data
    """
    # Ensure arrays have the same length by pairing them
    min_len = min(len(days_np), len(ltv_values))
    days_arr = np.array(days_np[:min_len])
    ltv_arr = np.array(ltv_values[:min_len])
    
    # Create paired filter: D2+ AND positive LTV values
    valid_mask = (days_arr >= 2) & (ltv_arr > 0)
    filtered_days = days_arr[valid_mask]
    filtered_ltv = ltv_arr[valid_mask]
    
    # Fallback if not enough valid data points
    if len(filtered_days) < 2:
        # Use all days >= 1 with positive LTV
        valid_mask = (days_arr >= 1) & (ltv_arr > 0)
        filtered_days = days_arr[valid_mask]
        filtered_ltv = ltv_arr[valid_mask]
    
    # Final safety check
    if len(filtered_days) < 2:
        # Not enough data - return simple curve
        new_days = np.arange(1, profile_max)
        predicted_values = np.full(len(new_days), ltv_d0)
        result_df = pd.DataFrame({'day': new_days, 'ltv': predicted_values})
        day_zero = pd.DataFrame({'day': [0], 'ltv': [ltv_d0]})
        result_df = pd.concat([result_df, day_zero]).sort_values('day').reset_index(drop=True)
        return result_df, 1.0, 0.0
    
    # Log transformation for linear regression
    log_days = np.log(filtered_days)
    log_ltv = np.log(filtered_ltv)
    
    # Linear regression in log space: ln(LTV) = ln(a) + b * ln(d)
    n = len(log_days)
    sum_x = np.sum(log_days)
    sum_y = np.sum(log_ltv)
    sum_x2 = np.sum(log_days ** 2)
    sum_xy = np.sum(log_days * log_ltv)
    
    denominator = n * sum_x2 - sum_x ** 2
    
    if denominator == 0:
        # Fallback for edge case
        b = 0.5
        a = np.mean(filtered_ltv)
    else:
        b = (n * sum_xy - sum_x * sum_y) / denominator
        ln_a = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
        a = np.exp(ln_a)
    
    # Predict for all days
    last_observed_day = int(filtered_days.max())
    new_days = np.arange(1, profile_max)
    predicted_values = a * np.power(new_days.astype(float), b)
    
    # Apply uncertainty discount for predictions beyond observed data
    # Calibrated based on actual data:
    # - D180: ~18% discount (119 predicted vs 98 actual)
    # - D360: ~30% discount (168 predicted vs 118 actual)
    days_beyond = np.maximum(new_days - last_observed_day, 0)
    # Non-linear scaling with power 0.65 for better fit at both D180 and D360
    discount = np.where(
        days_beyond > 0,
        0.02 + np.power(days_beyond / 300, 0.65) * 0.28,  # Scale from 2% to 30%
        0.0
    )
    predicted_values = predicted_values * (1 - discount)
    
    result_df = pd.DataFrame({'day': new_days, 'ltv': predicted_values})
    day_zero = pd.DataFrame({'day': [0], 'ltv': [ltv_d0]})
    result_df = pd.concat([result_df, day_zero]).sort_values('day').reset_index(drop=True)
    
    return result_df, a, b  # Return coefficients for validation


def load_data(uploaded_file):
    """Load and parse the uploaded Excel file."""
    cpi_dict = {}
    rr_dict = {}
    ltv_dict = {}
    
    # Read file content into bytes for multiple reads
    file_content = io.BytesIO(uploaded_file.read())
    
    # Load CPI/Install data (columns A-C)
    file_content.seek(0)
    all_sheets = pd.read_excel(file_content, sheet_name=None, skiprows=1, usecols='A:C')
    channel_list = []
    
    for sheet_name, df in all_sheets.items():
        # Standardize column names
        df.columns = ['day', 'INSTALLS', 'CPI']
        # Convert to numeric
        df['INSTALLS'] = pd.to_numeric(df['INSTALLS'], errors='coerce')
        df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
        df['day'] = pd.to_numeric(df['day'], errors='coerce')
        # Drop rows with NaN
        df = df.dropna()
        # Filter positive installs
        df = df[df['INSTALLS'] > 0]
        
        if len(df) > 0:
            cpi_dict[sheet_name] = df.reset_index(drop=True)
            channel_list.append(sheet_name)
    
    # Load Retention Rate data (column F)
    file_content.seek(0)
    rr_sheets = pd.read_excel(file_content, sheet_name=None, skiprows=1, usecols='F', nrows=7)
    for sheet_name, df in rr_sheets.items():
        df.columns = ['RR']
        df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
        rr_dict[sheet_name] = df.dropna().reset_index(drop=True)
    
    # Load LTV data (column G)
    file_content.seek(0)
    ltv_sheets = pd.read_excel(file_content, sheet_name=None, skiprows=1, usecols='G', nrows=7)
    for sheet_name, df in ltv_sheets.items():
        df.columns = ['LTV']
        df['LTV'] = pd.to_numeric(df['LTV'], errors='coerce')
        ltv_dict[sheet_name] = df.dropna().reset_index(drop=True)
    
    return cpi_dict, rr_dict, ltv_dict, channel_list


def process_channels(cpi_dict, rr_dict, ltv_dict, channel_list):
    """Process all channels and build projections."""
    variables = {}
    days_np = np.array(RETENTION_DAYS)
    
    for channel in channel_list:
        # Get retention rate values (use all rows since header is already skipped)
        rr_values = np.array(rr_dict[channel]['RR'].tolist()[1:])  # Skip first value (day 0)
        
        # Create retention profile
        result_df_rr = create_retention_profile(days_np, rr_values)
        
        # Get LTV values
        ltv_list = ltv_dict[channel]['LTV'].tolist()
        ltv_d0 = ltv_list[0]
        ltv_values = np.array(ltv_list[1:])  # Skip first value (day 0)
        
        # Create LTV curve (power-law model)
        result_df_ltv, ltv_a, ltv_b = create_ltv_curve(days_np, ltv_values, ltv_d0)
        
        # Validate LTV coefficients - flag unusual patterns
        if ltv_a >= 1 or ltv_b >= 1 or ltv_a <= 0 or ltv_b <= 0:
            st.warning(f"⚠️ {channel}: Unusual LTV growth pattern detected (a={ltv_a:.3f}, b={ltv_b:.3f})")
        
        # Merge RR and LTV data
        merge_df = result_df_ltv.merge(result_df_rr, on='day')
        
        # Calculate ARPU
        ltv_d0_value = merge_df.loc[merge_df['day'] == 0, 'ltv'].values[0]
        merge_df['arpu'] = (
            (merge_df['ltv'] - merge_df['ltv'].shift(1)) * 100 / 
            (merge_df['rr'] * ltv_d0_value)
        )
        
        # Extract values for profile creation
        retention_values = [x * 100 for x in merge_df[merge_df['day'].isin(RETENTION_DAYS)]['rr'].tolist()]
        
        # Filter out zero/negative installs and get valid cohorts
        cpi_data = cpi_dict[channel].copy()
        cpi_data = cpi_data[cpi_data['INSTALLS'] > 0]  # Remove zero/negative installs
        
        if len(cpi_data) == 0:
            raise ValueError(f"Channel '{channel}' has no valid install data (all values are 0 or negative)")
        
        cohorts = [int(x) for x in cpi_data['INSTALLS'].tolist()]
        
        # Double-check cohorts are valid
        cohorts = [c for c in cohorts if c > 0]
        
        if len(cohorts) == 0:
            raise ValueError(f"Channel '{channel}' has no valid cohort values after filtering")
        
        cost = (cpi_data['INSTALLS'] * cpi_data['CPI']).sum()
        
        # Update cpi_dict with cleaned data
        cpi_dict[channel] = cpi_data
        
        # Create LTV DataFrame
        df_ltv = merge_df[['day', 'arpu']].copy()
        df_ltv['day'] = df_ltv['day'] + 1
        df_ltv = df_ltv.set_index('day')
        df_ltv = df_ltv.rename(columns={'arpu': 'ltv_curve'})
        df_ltv['ltv_curve'] = df_ltv['ltv_curve'] / 100
        
        # Store computed values
        variables[f"{channel}_ltv"] = df_ltv
        variables[f"{channel}"] = th.create_profile(
            days=RETENTION_DAYS, 
            retention_values=retention_values, 
            profile_max=PROFILE_MAX_DAYS
        )
        variables[f"{channel}_DAU"] = th.project_cohorted_DAU(
            profile=variables[f"{channel}"], 
            periods=PROFILE_MAX_DAYS,
            cohorts=cohorts, 
            start_date=1
        )
        variables[f"{channel}_ltv_d0"] = ltv_d0
        variables[f"{channel}_cost"] = cost
        variables[f"{channel}_start_date"] = cpi_dict[channel]['day'].min()
    
    return variables


def build_revenue_tables(variables, cpi_dict, channel_list):
    """Build revenue and DAU tables."""
    ltv_tables = {}
    dau_tables = {}
    
    for channel in channel_list:
        df_total = variables[f"{channel}_ltv"].copy()
        df_total.loc[:1, 'ltv_curve'] = 1
        
        # Add Day 0 with ltv_curve = 1 (100% for install day)
        if 0 not in df_total.index:
            day_zero_row = pd.DataFrame({'ltv_curve': [1.0]}, index=[0])
            df_total = pd.concat([day_zero_row, df_total]).sort_index()
        
        df_total = df_total.T
        df_total.reset_index(drop=True, inplace=True)
        
        num_rows = len(variables[f"{channel}_DAU"])
        
        # Build shifted rows for cohort analysis
        for z in range(num_rows):
            new_row = [0] * (z + 1) + list(df_total.iloc[0, :-z - 1])
            df_total = pd.concat(
                [df_total, pd.DataFrame([new_row], columns=df_total.columns)], 
                ignore_index=True
            )
        
        df_total = df_total.loc[:num_rows - 1, :]
        
        # Set proper index
        df_total.index = pd.Index(
            variables[f"{channel}_DAU"].index.tolist(), 
            name='cohort_date'
        )
        df_total.columns = [str(col) for col in df_total.columns]
        
        # Apply time shift
        start_day = cpi_dict[channel]['day'].min()
        variables[f"{channel}_DAU"] = variables[f"{channel}_DAU"].shift(
            periods=start_day, axis=1, fill_value=0
        )
        
        # Calculate revenue table
        ltv_table = (
            variables[f"{channel}_DAU"] * 
            df_total.shift(periods=start_day, axis=1, fill_value=0) * 
            variables[f"{channel}_ltv_d0"]
        )
        
        ltv_tables[channel] = ltv_table
        dau_tables[channel] = variables[f"{channel}_DAU"]
    
    return ltv_tables, dau_tables


def safe_dau_total(dau_df):
    """Safely compute DAU total, handling any number of rows."""
    # Our custom th.DAU_total already handles all cases
    return th.DAU_total(dau_df)


def create_dau_chart(dau_tables, channel_list):
    """Create DAU visualization."""
    if len(channel_list) > 1:
        # Get DAU totals for each channel
        dau_totals = []
        for name, df in dau_tables.items():
            dau_totals.append(safe_dau_total(df))
        
        combined_DAU = th.combine_DAU(
            DAU_totals=dau_totals,
            labels=list(dau_tables.keys())
        )
        combined_DAU_t = combined_DAU.transpose()
        
        fig = go.Figure()
        for col in combined_DAU_t.columns:
            fig.add_trace(go.Bar(
                x=pd.to_numeric(combined_DAU_t.index), 
                y=combined_DAU_t[col], 
                name=col
            ))
        
        fig.update_layout(
            barmode='stack', 
            title='Daily Active Users by Channel', 
            xaxis_title='Day', 
            yaxis_title='DAU',
            height=500
        )
    else:
        channel_name = channel_list[0]
        combined_DAU = safe_dau_total(dau_tables[channel_name])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=combined_DAU.columns, 
            y=combined_DAU.iloc[0], 
            name=channel_name
        ))
        
        fig.update_layout(
            title='Daily Active Users',
            xaxis_title='Day',
            yaxis_title='DAU',
            height=500
        )
    
    return fig, combined_DAU


def create_revenue_chart(ltv_tables, channel_list):
    """Create Revenue visualization."""
    if len(channel_list) > 1:
        # Get revenue totals for each channel
        revenue_totals = []
        for name, df in ltv_tables.items():
            revenue_totals.append(safe_dau_total(df))
        
        combined_revenue = th.combine_DAU(
            DAU_totals=revenue_totals,
            labels=list(ltv_tables.keys())
        )
        combined_revenue_t = combined_revenue.transpose()
        
        fig = go.Figure()
        for col in combined_revenue_t.columns:
            fig.add_trace(go.Bar(
                x=pd.to_numeric(combined_revenue_t.index), 
                y=combined_revenue_t[col], 
                name=col
            ))
        
        fig.update_layout(
            barmode='stack', 
            title='Daily Revenue by Channel', 
            xaxis_title='Day', 
            yaxis_title='Revenue ($)',
            height=500
        )
    else:
        channel_name = channel_list[0]
        combined_revenue = safe_dau_total(ltv_tables[channel_name])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=combined_revenue.columns, 
            y=combined_revenue.iloc[0], 
            name=channel_name
        ))
        
        fig.update_layout(
            title='Daily Revenue',
            xaxis_title='Day',
            yaxis_title='Revenue ($)',
            height=500
        )
    
    return fig, combined_revenue


def create_profit_chart(ltv_tables, variables, channel_list):
    """Create profit/loss visualization."""
    total_revenue = sum(ltv.sum().sum() for ltv in ltv_tables.values())
    total_cost = sum(variables[f"{channel}_cost"] for channel in channel_list)
    
    combined_ltv = reduce(lambda df1, df2: df1 + df2, ltv_tables.values())
    df_rev = pd.DataFrame(combined_ltv.sum()).reset_index(drop=False)
    df_rev.columns = ['day', 'daily_revenue']
    df_rev['cumulative_revenue'] = df_rev['daily_revenue'].cumsum()
    df_rev['cumulative_profit'] = df_rev['cumulative_revenue'] - total_cost
    df_rev['day'] = df_rev['day'].astype(int)
    
    # Calculate Revenue at D180
    revenue_d180 = 0
    d180_rows = df_rev[df_rev['day'] <= 180]
    if len(d180_rows) > 0:
        revenue_d180 = d180_rows['cumulative_revenue'].iloc[-1]
    
    # Calculate Break Even Day (first day where cumulative_profit >= 0)
    break_even_day = None
    profitable_days = df_rev[df_rev['cumulative_profit'] >= 0]
    if len(profitable_days) > 0:
        break_even_day = int(profitable_days['day'].iloc[0])
    
    fig = go.Figure()
    
    # Cumulative revenue line
    fig.add_trace(go.Scatter(
        x=df_rev['day'], 
        y=df_rev['cumulative_revenue'], 
        mode='lines', 
        name='Cumulative Revenue',
        line=dict(color='green', width=2)
    ))
    
    # Cumulative profit line
    fig.add_trace(go.Scatter(
        x=df_rev['day'], 
        y=df_rev['cumulative_profit'], 
        mode='lines', 
        name='Cumulative Profit',
        line=dict(color='blue', width=2)
    ))
    
    # Cost reference line
    fig.add_trace(go.Scatter(
        x=[df_rev['day'].min(), df_rev['day'].max()],
        y=[total_cost, total_cost],
        mode='lines',
        name='Total Cost',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Break-even line (y=0)
    fig.add_trace(go.Scatter(
        x=[df_rev['day'].min(), df_rev['day'].max()],
        y=[0, 0],
        mode='lines',
        name='Break-even Line',
        line=dict(color='gray', dash='dot')
    ))
    
    # Add Break Even Day marker if found
    if break_even_day is not None:
        be_revenue = df_rev.loc[df_rev['day'] == break_even_day, 'cumulative_revenue'].values[0]
        fig.add_trace(go.Scatter(
            x=[break_even_day],
            y=[be_revenue],
            mode='markers+text',
            name=f'Break Even (D{break_even_day})',
            marker=dict(size=15, color='gold', symbol='star'),
            text=[f'D{break_even_day}'],
            textposition='top center',
            textfont=dict(size=12, color='gold')
        ))
        
        # Add vertical line at break even day
        fig.add_vline(
            x=break_even_day, 
            line_dash="dash", 
            line_color="gold",
            annotation_text=f"Break Even: D{break_even_day}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title='Cumulative Revenue and Profit Over Time (D0-D360)',
        xaxis_title='Day',
        yaxis_title='Value ($)',
        height=500,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
    )
    
    return fig, total_revenue, total_cost, break_even_day, revenue_d180


def to_excel_download(df, sheet_name='Sheet1'):
    """Convert DataFrame to Excel bytes for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name)
    return output.getvalue()


# =============================================================================
# Main App
# =============================================================================

def main():
    # Header
    st.title("📊 Master Plan - User Acquisition Analysis")
    st.markdown("""
    Analyze mobile app user acquisition performance including:
    - **Retention Rate (RR)** projections
    - **Lifetime Value (LTV)** curves
    - **Daily Active Users (DAU)** projections
    - **Revenue** and **ROAS** calculations
    """)
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    st.sidebar.markdown("""
    Upload an Excel file with sheets for each channel containing:
    - **Columns A-C**: day, INSTALLS, CPI
    - **Column F**: RR (Retention Rate) values
    - **Column G**: LTV values
    """)
    
    # Template download button
    import os
    template_path = os.path.join(os.path.dirname(__file__), 'input_template.xlsx')
    if os.path.exists(template_path):
        with open(template_path, 'rb') as f:
            template_data = f.read()
        st.sidebar.download_button(
            label="📥 Download Input Template",
            data=template_data,
            file_name="input_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the template Excel file to fill in your data"
        )
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose Excel file", 
        type=['xlsx', 'xls'],
        help="Upload your input template Excel file"
    )
    
    if uploaded_file is None:
        st.info("👈 Please upload an Excel file to begin the analysis.")
        
        # Show sample format
        with st.expander("📋 Expected File Format"):
            st.markdown("""
            Your Excel file should have:
            - One sheet per channel (e.g., 'applovin', 'googleads')
            - Each sheet contains:
              - **Column A (day)**: Day numbers
              - **Column B (INSTALLS)**: Number of installs per day
              - **Column C (CPI)**: Cost per install
              - **Column F (RR)**: Retention rates at days 1, 3, 7, 14, 30, 60
              - **Column G (LTV)**: LTV values at days 0, 1, 3, 7, 14, 30, 60
            """)
        return
    
    # Process the uploaded file
    try:
        with st.spinner("Loading data..."):
            cpi_dict, rr_dict, ltv_dict, channel_list = load_data(uploaded_file)
        
        # Show data preview for debugging
        with st.sidebar.expander("🔍 Data Preview"):
            for channel in channel_list:
                st.write(f"**{channel}:**")
                st.write(f"Rows: {len(cpi_dict[channel])}")
                st.dataframe(cpi_dict[channel].head(), height=150)
        
        # Validate data before processing
        valid_channels = []
        for channel in channel_list:
            if channel not in cpi_dict:
                st.sidebar.warning(f"⚠️ Channel '{channel}' has no CPI data")
                continue
            
            df = cpi_dict[channel]
            if len(df) == 0:
                st.sidebar.warning(f"⚠️ Channel '{channel}' skipped (no valid data rows)")
                continue
                
            valid_channels.append(channel)
        
        if len(valid_channels) == 0:
            st.error("❌ No valid data found. Please check your file format.")
            st.info("""
            **Expected format:**
            - Column A: day numbers (1, 2, 3, ...)
            - Column B: INSTALLS (positive numbers)
            - Column C: CPI (cost per install)
            - Column F: RR (retention rates)
            - Column G: LTV (lifetime values)
            """)
            return
        
        channel_list = valid_channels
        
        st.sidebar.success(f"✅ Loaded {len(channel_list)} channel(s)")
        st.sidebar.markdown(f"**Channels:** {', '.join(channel_list)}")
        
        with st.spinner("Processing channels..."):
            variables = process_channels(cpi_dict, rr_dict, ltv_dict, channel_list)
        
        with st.spinner("Building revenue tables..."):
            ltv_tables, dau_tables = build_revenue_tables(variables, cpi_dict, channel_list)
        
        # Create tabs for different views (ROAS & Profit first)
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 ROAS & Profit",
            "📈 DAU Analysis", 
            "💰 Revenue Analysis", 
            "📋 Channel Details"
        ])
        
        # Tab 1: ROAS & Profit (moved to first)
        with tab1:
            st.header("ROAS & Profit Analysis (D0-D360)")
            
            fig_profit, total_revenue, total_cost, break_even_day, revenue_d180 = create_profit_chart(
                ltv_tables, variables, channel_list
            )
            
            # Summary metrics - 6 columns now including ROAS D180 and Break Even Day
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Total Revenue (D360)", f"${total_revenue:,.2f}")
            
            with col2:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            
            with col3:
                roas_d180 = revenue_d180 / total_cost if total_cost > 0 else 0
                st.metric("ROAS (D180)", f"{roas_d180:.2%}")
            
            with col4:
                roas = total_revenue / total_cost if total_cost > 0 else 0
                st.metric("ROAS (D360)", f"{roas:.2%}")
            
            with col5:
                profit = total_revenue - total_cost
                st.metric(
                    "Profit/Loss", 
                    f"${profit:,.2f}",
                    delta=f"{profit:,.2f}",
                    delta_color="normal" if profit >= 0 else "inverse"
                )
            
            with col6:
                if break_even_day is not None:
                    st.metric("⭐ Break Even Day", f"D{break_even_day}")
                else:
                    st.metric("⭐ Break Even Day", "Not reached (D360+)")
            
            st.plotly_chart(fig_profit, use_container_width=True)
        
        # Tab 2: DAU Analysis
        with tab2:
            st.header("Daily Active Users (DAU)")
            fig_dau, combined_DAU = create_dau_chart(dau_tables, channel_list)
            st.plotly_chart(fig_dau, use_container_width=True)
            
            # Download button
            excel_data = to_excel_download(combined_DAU, 'DAU')
            st.download_button(
                label="📥 Download DAU Data",
                data=excel_data,
                file_name="DAU_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Tab 3: Revenue Analysis
        with tab3:
            st.header("Daily Revenue")
            fig_rev, combined_revenue = create_revenue_chart(ltv_tables, channel_list)
            st.plotly_chart(fig_rev, use_container_width=True)
            
            # Download button
            excel_data = to_excel_download(combined_revenue, 'Revenue')
            st.download_button(
                label="📥 Download Revenue Data",
                data=excel_data,
                file_name="Revenue_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Tab 4: Channel Details
        with tab4:
            st.header("Channel Performance Breakdown")
            
            channel_data = []
            for channel in channel_list:
                channel_revenue = ltv_tables[channel].sum().sum()
                channel_cost = variables[f"{channel}_cost"]
                channel_roas = channel_revenue / channel_cost if channel_cost > 0 else 0
                
                channel_data.append({
                    'Channel': channel,
                    'Revenue': channel_revenue,
                    'Cost': channel_cost,
                    'ROAS': channel_roas,
                    'Profit': channel_revenue - channel_cost
                })
            
            summary_df = pd.DataFrame(channel_data)
            
            # Display as styled table
            st.dataframe(
                summary_df.style.format({
                    'Revenue': '${:,.2f}',
                    'Cost': '${:,.2f}',
                    'ROAS': '{:.2%}',
                    'Profit': '${:,.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Channel comparison chart
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                name='Revenue',
                x=summary_df['Channel'],
                y=summary_df['Revenue'],
                marker_color='green'
            ))
            
            fig_comparison.add_trace(go.Bar(
                name='Cost',
                x=summary_df['Channel'],
                y=summary_df['Cost'],
                marker_color='red'
            ))
            
            fig_comparison.update_layout(
                title='Revenue vs Cost by Channel',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
