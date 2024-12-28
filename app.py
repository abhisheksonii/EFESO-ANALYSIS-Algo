import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_matador_machine(machine_no: Any, is_night_shift: bool) -> bool:
    """
    Checks if a machine number is a Matador machine
    
    Args:
        machine_no: Machine number to check
        is_night_shift: Boolean indicating if it's night shift
    
    Returns:
        bool: True if it's a Matador machine, False otherwise
    """
    if pd.isna(machine_no):
        return False
    
    machine_str = str(machine_no).strip().upper()
    
    # Day shift patterns
    day_patterns = ["MT-1", "MT-2", "MT-3", "MT-4"]
    # Night shift patterns
    night_patterns = ["MT-1-N", "MT-2-N", "MT-3-N", "MT-4-N"]
    
    if is_night_shift:
        return any(pattern in machine_str for pattern in night_patterns)
    else:
        return any(pattern in machine_str for pattern in day_patterns)
def clean_matador_data(df_section: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and restructures Matador section data to handle special formatting
    
    Args:
        df_section: DataFrame containing Matador section data
    
    Returns:
        Cleaned DataFrame with properly structured data
    """
    cleaned_data = []
    
    for idx, row in df_section.iterrows():
        # Convert row to string and join all cells
        row_text = ' '.join(str(cell).strip() for cell in row if pd.notna(cell))
        
        # Skip empty rows or headers
        if not row_text or row_text.isspace():
            continue
            
        # Check if this is a Matador machine row
        if any(pattern in row_text.upper() for pattern in ['MT-1', 'MT-2', 'MT-3', 'MT-4']):
            try:
                # Split the row text into parts
                parts = row_text.split()
                
                machine_no = next(part for part in parts if 'MT-' in part.upper())
                
                # Find indices for key parts
                machine_idx = parts.index(machine_no)
                
                # Extract operator name (usually after machine number)
                operator_name = parts[machine_idx + 1] if len(parts) > machine_idx + 1 else ''
                
                # Extract hours (usually numeric value followed by potential cost)
                hours = next((float(part) for i, part in enumerate(parts) 
                            if part.replace('.', '').isdigit() and i > machine_idx), 0)
                
                # Extract cost (usually starts with £)
                cost = next((float(part.replace('£', '')) for part in parts if '£' in part), 0)
                
                # Extract SAP code (7-8 digit number)
                sap_code = next((part for part in parts 
                               if part.isdigit() and len(part) in [7, 8]), '')
                
                # Extract size (usually contains 'x' between numbers)
                size = next((part for part in parts 
                           if 'x' in part and any(c.isdigit() for c in part)), '')
                
                # Extract material description (between size and numeric values)
                size_idx = next((i for i, part in enumerate(parts) if 'x' in part), -1)
                material_desc = ' '.join(parts[size_idx + 1:size_idx + 8]) if size_idx != -1 else ''
                
                # Extract numeric values
                numbers = [float(part.replace(',', '')) 
                          for part in parts 
                          if part.replace(',', '').replace('.', '').isdigit()]
                
                # Create row dictionary
                row_dict = {
                    'Machine_No': machine_no,
                    'NAME': operator_name,
                    'Hours': hours,
                    'Operator_Cost': cost,
                    'Size': size,
                    'SAP_Code': sap_code,
                    'Material_Description': material_desc,
                }
                
                # Add numeric values to appropriate columns
                if len(numbers) >= 10:
                    row_dict.update({
                        'Net_Weight': numbers[0] if len(numbers) > 0 else 0,
                        'Per_Pack': numbers[1] if len(numbers) > 1 else 0,
                        'Bag_Produce': numbers[2] if len(numbers) > 2 else 0,
                        'Packet_Produce': numbers[3] if len(numbers) > 3 else 0,
                        'In_Kgs': numbers[4] if len(numbers) > 4 else 0,
                        'target_Bag_Produce': numbers[5] if len(numbers) > 5 else 0,
                        'Pkt': numbers[6] if len(numbers) > 6 else 0,
                        'KG_Target': numbers[7] if len(numbers) > 7 else 0,
                        'Pkt_Var_%': numbers[8] if len(numbers) > 8 else 0,
                        'KG_Variance_%': numbers[9] if len(numbers) > 9 else 0,
                    })
                
                cleaned_data.append(row_dict)
                
            except Exception as e:
                logger.warning(f"Error processing Matador row: {row_text}\nError: {str(e)}")
                continue
    
    return pd.DataFrame(cleaned_data)

def should_exclude_machine(machine_no: Any) -> bool:
    """
    Checks if a machine number should be excluded from analysis
    
    Args:
        machine_no: Machine number to check
    
    Returns:
        bool: True if machine should be excluded, False otherwise
    """
    if pd.isna(machine_no):
        return True
    
    machine_str = str(machine_no).strip().upper()
    exclude_patterns = [
        "MACHINE MACHINE NO",
        "MACHINE BS/M/NO",
        "MACHINE NO",
        "BS/M/NO"
    ]
    return any(pattern.upper() in machine_str for pattern in exclude_patterns)

def identify_sections(df_full: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identifies different sections and their row ranges in the data, separating day and night shifts
    
    Args:
        df_full: Full DataFrame containing the production data
    
    Returns:
        List of dictionaries containing section information
    """
    sections = []
    current_section = None
    start_idx = None
    is_night_shift = False
    matador_start_idx = None
    
    section_headers = [
        "BEASLEY DAY SHIFT PRODUCTION",
        "SOS SECTION",
        "Amazon ( Holweg + Garant ) Machines",
        "Garant Section",
        "Handle SECTION",
        "MATADOR SECTION",
        "Matador Section",
        "Sheeter Section",
        "DAILY  PRODUCTION  REPORT (NIGHT SHIFT)",
        "BEASLEY SECTION"
    ]
    
    # Add the first section by default
    current_section = "BEASLEY DAY SHIFT PRODUCTION"
    start_idx = 0
    
    for idx, row in df_full.iterrows():
        row_text = ' '.join(str(cell).strip() for cell in row if pd.notna(cell))
        
        # Check for night shift marker
        if any(night_marker in row_text.upper() for night_marker in ["NIGHT SHIFT", "NIGHT  SHIFT"]):
            is_night_shift = True
            if current_section is not None and start_idx is not None:
                sections.append({
                    'name': current_section,
                    'start': start_idx,
                    'end': idx - 1,
                    'shift': 'Day'
                })
                current_section = None
                start_idx = None
            continue

        # Check for Matador machines
        if any(pd.notna(cell) and isinstance(cell, str) and 
               is_matador_machine(cell, is_night_shift) for cell in row):
            if matador_start_idx is None:
                matador_start_idx = idx
                current_section = "Matador Section"
                start_idx = idx
            continue
        
        # Special handling for Matador section
        if "MATADOR" in row_text.upper() and "SECTION" in row_text.upper():
            if current_section is not None and start_idx is not None:
                sections.append({
                    'name': current_section,
                    'start': start_idx,
                    'end': idx - 1,
                    'shift': 'Night' if is_night_shift else 'Day'
                })
            current_section = "Matador Section"
            start_idx = idx + 1
            continue
            
        for header in section_headers:
            if header.strip().upper() in row_text.strip().upper():
                if current_section is not None and start_idx is not None:
                    sections.append({
                        'name': current_section,
                        'start': start_idx,
                        'end': idx - 1,
                        'shift': 'Night' if is_night_shift else 'Day'
                    })
                
                current_section = header
                start_idx = idx + 1
                break
        
        # Check for section end markers or next section start
        if current_section == "Matador Section":
            if (any(marker in row_text.upper() for marker in ['SUMMARY', 'TOTAL', 'GRAND TOTAL']) or
                any(header.strip().upper() in row_text.strip().upper() for header in section_headers)):
                sections.append({
                    'name': current_section,
                    'start': start_idx,
                    'end': idx - 1,
                    'shift': 'Night' if is_night_shift else 'Day'
                })
                current_section = None
                start_idx = None
                matador_start_idx = None
        elif current_section and any(marker in row_text.upper() for marker in ['SUMMARY', 'TOTAL', 'GRAND TOTAL']):
            sections.append({
                'name': current_section,
                'start': start_idx,
                'end': idx - 1,
                'shift': 'Night' if is_night_shift else 'Day'
            })
            current_section = None
            start_idx = None
    
    # Add the last section if it exists
    if current_section is not None and start_idx is not None:
        sections.append({
            'name': current_section,
            'start': start_idx,
            'end': len(df_full) - 1,
            'shift': 'Night' if is_night_shift else 'Day'
        })
    
    return sections
def combine_shift_data(sections_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Combines day and night shift data for each section type
    
    Args:
        sections_data: Dictionary containing section DataFrames
    
    Returns:
        Dictionary containing combined section DataFrames
    """
    combined_sections = {}
    section_mapping = {
        'BEASLEY': ['BEASLEY DAY SHIFT PRODUCTION', 'BEASLEY SECTION'],
        'SOS': ['SOS SECTION'],
        'AMAZON': ['Amazon ( Holweg + Garant ) Machines'],
        'GARANT': ['Garant Section'],
        'HANDLE': ['Handle SECTION'],
        'MATADOR': ['Matador Section'],
        'SHEETER': ['Sheeter Section']
    }
    
    # Initialize combined DataFrames for each section type
    for section_type in section_mapping.keys():
        combined_sections[section_type] = pd.DataFrame()
    
    # Combine data from day and night shifts
    for section_name, df in sections_data.items():
        for section_type, patterns in section_mapping.items():
            if any(pattern in section_name for pattern in patterns):
                if combined_sections[section_type].empty:
                    combined_sections[section_type] = df.copy()
                else:
                    # Combine data while maintaining unique machine count
                    combined_sections[section_type] = pd.concat([
                        combined_sections[section_type],
                        df
                    ], ignore_index=True)
    
    # Process combined data to maintain correct machine counts
    for section_type in combined_sections:
        if not combined_sections[section_type].empty:
            df = combined_sections[section_type]
            
            # Sum numeric columns
            numeric_cols = [
                'Hours', 'Operator_Cost', 'Net_Weight', 'Per_Pack', 'Bag_Produce',
                'Packet_Produce', 'In_Kgs', 'target_Bag_Produce', 'Pkt', 'KG_Target',
                'Pkt_Var', 'KG_Variance', 'target_Bag_Produce_Variance'
            ]
            
            # Group by machine position (not machine number) and sum
            df['machine_position'] = range(len(df))
            df_grouped = df.groupby(df.index % (len(df) // 2))[numeric_cols].sum()
            
            # Calculate new efficiencies
            df_grouped['Pkt_Var_%'] = (df_grouped['Packet_Produce'] / df_grouped['Pkt'] * 100) if 'Pkt' in df_grouped.columns else 0
            df_grouped['KG_Variance_%'] = (df_grouped['In_Kgs'] / df_grouped['KG_Target'] * 100) if 'KG_Target' in df_grouped.columns else 0
            
            combined_sections[section_type] = df_grouped.reset_index(drop=True)
    
    return combined_sections

def process_section_data(df_section: pd.DataFrame, columns: List[str], is_matador: bool = False) -> pd.DataFrame:
    """
    Process a section's DataFrame by renaming columns and converting numeric data
    
    Args:
        df_section: DataFrame containing section data
        columns: List of column names to use
        is_matador: Boolean indicating if this is the Matador section
    
    Returns:
        Processed DataFrame
    """
    # Rename columns
    for i, col in enumerate(columns):
        if i < len(df_section.columns):
            df_section.rename(columns={df_section.columns[i]: col}, inplace=True)
    
    # Clean numeric columns
    numeric_columns = [
        'Hours', 'Operator_Cost', 'Net_Weight', 'Per_Pack', 'Bag_Produce',
        'Packet_Produce', 'In_Kgs', 'target_Bag_Produce', 'Pkt', 'KG_Target',
        'Pkt_Var', 'KG_Variance', 'target_Bag_Produce_Variance', 'Pkt_Var_%',
        'KG_Variance_%'
    ]
    
    for col in numeric_columns:
        if col in df_section.columns:
            df_section[col] = pd.to_numeric(df_section[col], errors='coerce')
    
    if is_matador:
        # Filter only Matador machines based on shift
        is_night = any("Night" in str(val) for val in df_section['Machine_No'] if pd.notna(val))
        df_section = df_section[df_section['Machine_No'].apply(lambda x: is_matador_machine(x, is_night))]
    
    return df_section


def read_production_data(uploaded_file) -> Tuple[Dict[str, pd.DataFrame], Any]:
    """
    Reads production data from uploaded file and separates it into sections by shift
    
    Args:
        uploaded_file: Uploaded Excel file
    
    Returns:
        Tuple containing processed sections dictionary and production date
    """
    try:
        df_full = pd.read_excel(uploaded_file, skiprows=2)
        sections = identify_sections(df_full)
        processed_sections = {}
        
        columns = [
            'Machine_No', 'Supervisor', 'Hours', 'Operator_Cost', 'NAME', 'SAP_Code',
            'Net_Weight', 'Size', 'Material_Description', 'Per_Pack', 'Bag_Produce',
            'Packet_Produce', 'In_Kgs', 'target_Bag_Produce', 'Pkt', 'KG_Target',
            'Pkt_Var', 'KG_Variance', 'target_Bag_Produce_Variance', 'Pkt_Var_%',
            'KG_Variance_%'
        ]
        
        for section in sections:
            section_key = f"{section['name']} ({section['shift']})"
            df_section = df_full.iloc[section['start']:section['end']].copy()
            
            if df_section.empty:
                continue
            
            is_matador = "Matador" in section['name']
            df_section = process_section_data(df_section, columns, is_matador)
            df_section = df_section[~df_section['Machine_No'].apply(should_exclude_machine)]
            
            if not df_section.empty:
                processed_sections[section_key] = df_section
        
        date_row = pd.read_excel(uploaded_file, nrows=1)
        production_date = date_row.columns[0]
        
        return processed_sections, production_date
    
    except Exception as e:
        logger.error(f"Error reading production data: {str(e)}")
        raise Exception(f"Error reading production data: {str(e)}")

def analyze_production_data(sections_data: Dict[str, pd.DataFrame], production_date: Any) -> Dict[str, Any]:
    """
    Analyzes production data for combined sections and returns structured results
    
    Args:
        sections_data: Dictionary containing section DataFrames
        production_date: Production date
    
    Returns:
        Dictionary containing analysis results
    """
    # Combine shift data first
    combined_sections = combine_shift_data(sections_data)
    sections_analysis = {}
    
    for section_type, df in combined_sections.items():
        if df.empty:
            continue
            
        analysis = {
            'date': production_date,
            'section_name': section_type,
            'summary': {
                'total_machines': len(df),  # Now represents unique machines
                'total_hours': df['Hours'].sum(),
                'total_operator_cost': df['Operator_Cost'].sum(),
                'production': {
                    'total_per_pack': df['Per_Pack'].sum(),
                    'total_bags': df['Bag_Produce'].fillna(0).sum(),
                    'total_packets': df['Packet_Produce'].fillna(0).sum(),
                    'total_kgs': df['In_Kgs'].fillna(0).sum(),
                },
                'targets': {
                    'total_bag_target': df['target_Bag_Produce'].fillna(0).sum(),
                    'total_pkt_target': df['Pkt'].fillna(0).sum(),
                    'total_kg_target': df['KG_Target'].fillna(0).sum(),
                },
                'variances': {
                    'total_pkt_variance': df['Pkt_Var'].fillna(0).sum(),
                    'total_kg_variance': df['KG_Variance'].fillna(0).sum(),
                    'total_bag_variance': df['target_Bag_Produce_Variance'].fillna(0).sum(),
                }
            },
            'performance': {
                'average_efficiency': df['Pkt_Var_%'].mean(),
                'machines_above_target': len(df[df['Pkt_Var_%'] > 100]),
                'machines_below_target': len(df[df['Pkt_Var_%'] <= 100]),
            }
        }
        
        sections_analysis[section_type] = analysis
    
    return sections_analysis

def calculate_advanced_metrics(section_analysis: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates advanced metrics for a section with improved error handling
    
    Args:
        section_analysis: Dictionary containing section analysis data
    
    Returns:
        Dictionary containing calculated metrics
    """
    summary = section_analysis['summary']
    
    # Extract base values with safe defaults
    total_hours = float(summary['total_hours']) if summary['total_hours'] else 0
    total_machines = int(summary['total_machines']) if summary['total_machines'] else 0
    avg_efficiency = float(section_analysis['performance']['average_efficiency']) if section_analysis['performance']['average_efficiency'] else 0
    
    # Production values
    total_packets = float(summary['production']['total_packets']) if summary['production'].get('total_packets') else 0
    target_packets = float(summary['targets']['total_pkt_target']) if summary['targets'].get('total_pkt_target') else 0
    total_bags = float(summary['production']['total_bags']) if summary['production'].get('total_bags') else 0
    target_bags = float(summary['targets']['total_bag_target']) if summary['targets'].get('total_bag_target') else 0
    total_kgs = float(summary['production']['total_kgs']) if summary['production'].get('total_kgs') else 0
    target_kgs = float(summary['targets']['total_kg_target']) if summary['targets'].get('total_kg_target') else 0
    total_operator_cost = float(summary['total_operator_cost']) if summary['total_operator_cost'] else 0

    # Factory settings
    FACTORY_SHIFT_PATTERN = 110  # Hours per shift pattern
    WEEKLY_WORKING_HOURS = 37.5  # Standard weekly working hours

    try:
        # Weekly Volume calculation
        weekly_volume = (total_packets / target_packets * target_packets * 7) if target_packets > 0 else 0
        
        # Value Adding Hours
        value_adding_hours = (total_hours * avg_efficiency) / 100 if total_hours > 0 else 0
        
        # Actual OEE
        actual_oee = value_adding_hours / total_hours if total_hours > 0 else 0
        
        # Saturation
        saturation = total_hours / (total_machines * FACTORY_SHIFT_PATTERN) if total_machines > 0 else 0
        
        # Production Per Pack
        production_per_pack = total_packets / total_hours if total_hours > 0 else 0
        
        # Operator Cost Per Hour
        operator_cost_per_hour = total_operator_cost / total_hours if total_hours > 0 else 0
        
        # Annual Efficiency
        annual_efficiency = (WEEKLY_WORKING_HOURS / FACTORY_SHIFT_PATTERN) * 100
        
        # Variance calculations with error handling
        bag_variance = ((total_bags - target_bags) / target_bags * 100) if target_bags > 0 else 0
        packet_variance = ((total_packets - target_packets) / target_packets * 100) if target_packets > 0 else 0
        kg_variance = ((total_kgs - target_kgs) / target_kgs * 100) if target_kgs > 0 else 0

    except ZeroDivisionError:
        logger.warning(f"Zero division encountered in section {section_analysis['section_name']}")
        return {
            'weekly_volume': 0,
            'value_adding_hours': 0,
            'actual_oee': 0,
            'saturation': 0,
            'production_per_pack': 0,
            'operator_cost_per_hour': 0,
            'annual_efficiency': 0,
            'bag_variance': 0,
            'packet_variance': 0,
            'kg_variance': 0
        }

    return {
        'weekly_volume': weekly_volume,
        'value_adding_hours': value_adding_hours,
        'actual_oee': actual_oee,
        'saturation': saturation,
        'production_per_pack': production_per_pack,
        'operator_cost_per_hour': operator_cost_per_hour,
        'annual_efficiency': annual_efficiency,
        'bag_variance': bag_variance,
        'packet_variance': packet_variance,
        'kg_variance': kg_variance
    }
def display_section_metrics(sections_analysis: Dict[str, Any]) -> None:
    """
    Displays metrics for combined sections with improved formatting and error handling
    """
    st.header("Combined Section Performance Metrics (Day + Night)")
    
    for section_type, analysis in sections_analysis.items():
        st.subheader(f"{section_type} Section")
        
        metrics = calculate_advanced_metrics(analysis)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Weekly Volume", f"{metrics['weekly_volume']:,.0f}")
            st.metric("Value Adding Hours", f"{metrics['value_adding_hours']:.2f}")
            st.metric("Actual OEE", f"{metrics['actual_oee']:.2%}")
        
        with col2:
            st.metric("Saturation", f"{metrics['saturation']:.2%}")
            st.metric("Production Per Pack Total", f"{metrics['production_per_pack']:.2f}")
            st.metric("Operator Cost Per Hour", f"£{metrics['operator_cost_per_hour']:.2f}")
        
        with col3:
            st.metric("Annual Efficiency", f"{metrics['annual_efficiency']:.1f}%")
            st.metric("Bag Variance", f"{metrics['bag_variance']:.1f}%")
            st.metric("Packet Variance", f"{metrics['packet_variance']:.1f}%")
        
        with col4:
            st.metric("KG Variance", f"{metrics['kg_variance']:.1f}%")
            st.metric("Total Machines", analysis['summary']['total_machines'])
            st.metric("Total Hours", f"{analysis['summary']['total_hours']:.1f}")
def display_section_analysis(section_analysis: Dict[str, Any]) -> None:
    """
    Displays analysis results for a single section in Streamlit
    
    Args:
        section_analysis: Dictionary containing section analysis data
    """
    st.header(section_analysis['section_name'])
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Machines", section_analysis['summary']['total_machines'])
        st.metric("Total Hours", f"{section_analysis['summary']['total_hours']:.2f}")
        st.metric("Total Operator Cost", f"£{section_analysis['summary']['total_operator_cost']:,.2f}")
    
    with col2:
        st.subheader("Production")
        st.metric("Per Pack Total", f"{section_analysis['summary']['production']['total_per_pack']:,.2f}")
        st.metric("Total Bags", f"{section_analysis['summary']['production']['total_bags']:,.0f}")
        st.metric("Total Packets", f"{section_analysis['summary']['production']['total_packets']:,.0f}")
        st.metric("Total KGs", f"{section_analysis['summary']['production']['total_kgs']:,.2f}")
    
    with col3:
        st.subheader("Targets & Variances")
        st.metric("Target Bags", f"{section_analysis['summary']['targets']['total_bag_target']:,.0f}")
        st.metric("Target Packets", f"{section_analysis['summary']['targets']['total_pkt_target']:,.0f}")
        st.metric("Target KGs", f"{section_analysis['summary']['targets']['total_kg_target']:,.2f}")
        st.metric("Average Efficiency", f"{section_analysis['performance']['average_efficiency']:.1f}%")
    
    # Display machine details
    st.subheader("Machine Details")
    for machine in section_analysis['machine_details']:
        with st.expander(f"Machine {machine['machine_no']}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("**Basic Info**")
                st.write(f"Supervisor: {machine['supervisor']}")
                st.write(f"Operator: {machine['operator']}")
                st.write(f"Hours: {machine['hours']}")
                st.write(f"Operator Cost: £{machine['operator_cost']:.2f}")
            
            with col2:
                st.write("**Production**")
                st.write(f"Per Pack: {machine['production']['per_pack']:,.2f}")
                st.write(f"Bags: {machine['production']['bags']:,.0f}")
                st.write(f"Packets: {machine['production']['packets']:,.0f}")
                st.write(f"KGs: {machine['production']['kgs']:,.2f}")
            
            with col3:
                st.write("**Targets**")
                st.write(f"Bag Target: {machine['targets']['bag_target']:,.0f}")
                st.write(f"Packet Target: {machine['targets']['pkt_target']:,.0f}")
                st.write(f"KG Target: {machine['targets']['kg_target']:,.2f}")
            
            with col4:
                st.write("**Variances**")
                st.write(f"Packet Variance: {machine['variances']['pkt_variance']:,.0f}")
                st.write(f"KG Variance: {machine['variances']['kg_variance']:,.2f}")
                st.write(f"Bag Variance: {machine['variances']['bag_variance']:,.0f}")
                st.write(f"Efficiency: {machine['variances']['efficiency']:.1f}%")
                st.write(f"KG Efficiency: {machine['variances']['kg_efficiency']:.1f}%")

def main():
    """
    Main function to run the Streamlit application
    """
    st.title("Production Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload Production Data", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            sections_data, production_date = read_production_data(uploaded_file)
            sections_analysis = analyze_production_data(sections_data, production_date)
            
            st.write(f"## Production Analysis - {production_date}")
            
            # Display the combined metrics
            display_section_metrics(sections_analysis)
            
            # Add export functionality
            if st.button("Export Analysis"):
                export_data = []
                for section_type, analysis in sections_analysis.items():
                    metrics = calculate_advanced_metrics(analysis)
                    export_data.append({
                        'Date': analysis['date'],
                        'Section': section_type,
                        'Total_Machines': analysis['summary']['total_machines'],
                        'Weekly_Volume': metrics['weekly_volume'],
                        'Value_Adding_Hours': metrics['value_adding_hours'],
                        'Actual_OEE': metrics['actual_oee'],
                        'Saturation': metrics['saturation'],
                        'Production_Per_Pack': metrics['production_per_pack'],
                        'Operator_Cost_Per_Hour': metrics['operator_cost_per_hour'],
                        'Bag_Variance': metrics['bag_variance'],
                        'Packet_Variance': metrics['packet_variance'],
                        'KG_Variance': metrics['kg_variance']
                    })
                
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis CSV",
                        data=csv,
                        file_name=f"production_metrics_{production_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()