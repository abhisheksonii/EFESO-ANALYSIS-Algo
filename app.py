import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
from anthropic import Anthropic
import io 
# Set up logging and Anthropic client
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
anthropic = Anthropic()

def read_file(uploaded_file: Any) -> pd.ExcelFile:
    """Read different file formats and return as ExcelFile object"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type in ['xlsx', 'xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
        return pd.ExcelFile(uploaded_file)
    elif file_type == 'csv':
        # Create a buffer to store the CSV as Excel
        buffer = io.BytesIO()
        
        # Read CSV and write to Excel format in memory
        df = pd.read_csv(uploaded_file)
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        
        buffer.seek(0)
        return pd.ExcelFile(buffer)
    else:
        raise ValueError(f"Unsupported file format: {file_type}")

def batch_process_sheets(uploaded_files: List[Any], max_sheets: int, file_stats: Dict[str, int]) -> Tuple[List[Tuple[pd.DataFrame, Any]], int]:
    """Process uploaded files up to user-defined sheet limit with progress tracking"""
    processed_sheets = []
    total_sheets_processed = 0
    skipped_sheets = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for file_index, uploaded_file in enumerate(uploaded_files):
        try:
            excel_file = read_file(uploaded_file)
            file_stats['processed_files'] += 1
            
            for sheet_index, sheet_name in enumerate(excel_file.sheet_names):
                if total_sheets_processed >= max_sheets:
                    skipped_sheets += 1
                    continue
                    
                try:
                    # Update progress
                    progress = (total_sheets_processed + 1) / min(max_sheets, file_stats['total_sheets'])
                    progress_bar.progress(progress)
                    status_text.text(f"Processing file {file_index + 1}/{file_stats['total_files']} - Sheet {sheet_name}")
                    
                    # Read the date from first row
                    date_row = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=1)
                    production_date = pd.to_datetime(date_row.columns[0])
                    
                    # Read actual data
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=2)
                    processed_sheets.append((df, production_date))
                    total_sheets_processed += 1
                    file_stats['processed_sheets'] += 1
                    
                    logger.info(f"Processed sheet {sheet_name} ({total_sheets_processed}/{max_sheets})")
                    
                except Exception as e:
                    logger.warning(f"Error reading sheet {sheet_name}: {e}")
                    continue
                
                if total_sheets_processed >= max_sheets:
                    break
                    
        except Exception as e:
            logger.warning(f"Error processing file {uploaded_file.name}: {e}")
            continue
    
    # Clear progress bar and status text after completion
    progress_bar.empty()
    status_text.empty()
    
    return processed_sheets, skipped_sheets

def identify_sections(df_full: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify different sections and their row ranges"""
    sections = []
    current_section = None
    start_idx = None
    is_night_shift = False
    
    section_headers = [
        "BEASLEY DAY SHIFT PRODUCTION",
        "SOS SECTION",
        "MATADOR SECTION",
        "SHEETER SECTION",
        "HANDLE SECTION",
        "GARANT SECTION",
        "AMAZON SECTION",
        "DAILY PRODUCTION REPORT (NIGHT SHIFT)"
    ]
    
    # Add first section by default
    current_section = "BEASLEY DAY SHIFT PRODUCTION"
    start_idx = 0
    
    for idx, row in df_full.iterrows():
        row_text = ' '.join(str(cell).strip() for cell in row if pd.notna(cell))
        
        # Check for night shift marker
        if "NIGHT SHIFT" in row_text.upper():
            is_night_shift = True
            if current_section and start_idx is not None:
                # Find the actual end by looking for summary row
                end_idx = idx - 1
                while end_idx > start_idx:
                    summary_row_text = ' '.join(str(cell).strip() for cell in df_full.iloc[end_idx] if pd.notna(cell))
                    if "Machine No" in summary_row_text and "Supervisor" in summary_row_text and "NAME" in summary_row_text:
                        end_idx -= 1
                    else:
                        break
                        
                sections.append({
                    'name': current_section,
                    'start': start_idx,
                    'end': end_idx,
                    'shift': 'Day'
                })
                current_section = None
                start_idx = None
            continue
        
        # Check for section headers
        for header in section_headers:
            if header.strip().upper() in row_text.strip().upper():
                if current_section and start_idx is not None:
                    # Find the actual end by looking for summary row
                    end_idx = idx - 1
                    while end_idx > start_idx:
                        summary_row_text = ' '.join(str(cell).strip() for cell in df_full.iloc[end_idx] if pd.notna(cell))
                        if "Machine No" in summary_row_text and "Supervisor" in summary_row_text and "NAME" in summary_row_text:
                            end_idx -= 1
                        else:
                            break
                            
                    sections.append({
                        'name': current_section,
                        'start': start_idx,
                        'end': end_idx,
                        'shift': 'Night' if is_night_shift else 'Day'
                    })
                
                current_section = header
                start_idx = idx + 1
                break
        
        # Check for section end markers
        if current_section and any(marker in row_text.upper() for marker in ['SUMMARY', 'TOTAL']):
            # Find the actual end by looking for summary row
            end_idx = idx - 1
            while end_idx > start_idx:
                summary_row_text = ' '.join(str(cell).strip() for cell in df_full.iloc[end_idx] if pd.notna(cell))
                if "Machine No" in summary_row_text and "Supervisor" in summary_row_text and "NAME" in summary_row_text:
                    end_idx -= 1
                else:
                    break
                    
            sections.append({
                'name': current_section,
                'start': start_idx,
                'end': end_idx,
                'shift': 'Night' if is_night_shift else 'Day'
            })
            current_section = None
            start_idx = None
    
    # Add last section if exists
    if current_section and start_idx is not None:
        # Find the actual end by looking for summary row
        end_idx = len(df_full) - 1
        while end_idx > start_idx:
            summary_row_text = ' '.join(str(cell).strip() for cell in df_full.iloc[end_idx] if pd.notna(cell))
            if "Machine No" in summary_row_text and "Supervisor" in summary_row_text and "NAME" in summary_row_text:
                end_idx -= 1
            else:
                break
                
        sections.append({
            'name': current_section,
            'start': start_idx,
            'end': end_idx,
            'shift': 'Night' if is_night_shift else 'Day'
        })
    
    return sections

def process_section_data(df_section: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Process section DataFrame by cleaning and standardizing data"""
    # Create a copy to avoid modifying the original
    df_section = df_section.copy()
    
    # Filter out summary rows where Machine_No/Supervisor/NAME contains their column names
    df_section = df_section[
        ~(
            (df_section.iloc[:, 0].astype(str).str.contains('Machine No', case=False, na=False)) &
            (df_section.iloc[:, 1].astype(str).str.contains('Supervisor', case=False, na=False)) &
            (df_section.iloc[:, 2].astype(str).str.contains('NAME', case=False, na=False))
        )
    ]
    
    # Rename columns
    for i, col in enumerate(columns):
        if i < len(df_section.columns):
            df_section.rename(columns={df_section.columns[i]: col}, inplace=True)
    
    # Filter out rows where Machine_No equals "Machine No" (additional safety check)
    df_section = df_section[df_section['Machine_No'] != 'Machine No']
    
    # Convert SAP_Code to string explicitly
    if 'SAP_Code' in df_section.columns:
        df_section['SAP_Code'] = df_section['SAP_Code'].fillna('').astype(str)
    
    # Clean numeric columns
    numeric_columns = [
        'Hours', 'Operator_Cost', 'Net_Weight', 'Per_Pack', 'Bag_Produce',
        'Packet_Produce', 'In_Kgs', 'target_Bag_Produce', 'Pkt', 'KG_Target',
        'Pkt_Var_%', 'KG_Variance_%'
    ]
    
    for col in numeric_columns:
        if col in df_section.columns:
            df_section[col] = pd.to_numeric(df_section[col], errors='coerce').fillna(0)
    
    return df_section

def aggregate_machine_data(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate data for each machine across multiple dates"""
    if not df_list:
        return pd.DataFrame()
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Group by machine number and aggregate
    aggregated = combined_df.groupby('Machine_No').agg({
        'Supervisor': 'first',
        'NAME': 'first',
        'Hours': 'sum',
        'Operator_Cost': 'sum',
        'SAP_Code': 'first',
        'Net_Weight': 'mean',
        'Size': 'first',
        'Material_Description': 'first',
        'Per_Pack': 'sum',
        'Bag_Produce': 'sum',
        'Packet_Produce': 'sum',
        'In_Kgs': 'sum',
        'target_Bag_Produce': 'sum',
        'Pkt': 'sum',
        'KG_Target': 'sum',
        'Production_Date': list
    }).reset_index()
    
    # Calculate efficiency percentages
    aggregated['Pkt_Var_%'] = (aggregated['Packet_Produce'] / aggregated['Pkt'] * 100).round(2)
    aggregated['KG_Variance_%'] = (aggregated['In_Kgs'] / aggregated['KG_Target'] * 100).round(2)
    
    return aggregated

def calculate_ref_speed(section_summary):
    """Calculate reference speed based on weighted production rates"""
    production = section_summary['production']['total_packets']
    hours = section_summary['total_hours']
    return (production / (hours * 60)) if hours > 0 else 0

def get_actual_manhours(section_name):
    """Get predefined actual manhours for specific sections"""
    # Remove shift information from section name
    base_section = section_name.split(' (')[0].upper()
    manhours_map = {
        'BEASLEY DAY SHIFT PRODUCTION': 2010,
        'SOS SECTION': 810,
        'GARANT SECTION': 495,
        'SHEETER SECTION': 120,
        'HANDLE SECTION': 285
    }
    return manhours_map.get(base_section)

def transform_to_capacity_report(sections_analysis: Dict[str, Any]) -> pd.DataFrame:
    """Transform section analysis data into capacity report format with updated calculations"""
    # Factory parameters
    FACTORY_SHIFT_PATTERN = 110
    WORKING_DAYS_RATIO = 22 / 30
    FIXED_MC_HOURS = 60
    
    report_data = []
    processed_sections = set()  # Track processed sections to avoid duplicates
    
    for section_name, analysis in sections_analysis.items():
        # Extract base section name (without shift)
        base_section = section_name.split(' (')[0]
        
        # Skip if we've already processed this base section
        if base_section in processed_sections:
            continue
            
        processed_sections.add(base_section)
        
        # Basic metrics
        machines = analysis['summary']['total_machines']
        actual_hours = analysis['summary']['total_hours']
        
        # Calculate weekly volume with working days ratio
        weekly_volume = analysis['summary']['production']['total_packets'] * WORKING_DAYS_RATIO
        
        # Calculate reference speed
        ref_speed = calculate_ref_speed(analysis['summary'])
        
        # Calculate value adding hours
        value_adding_hours = (weekly_volume / (ref_speed * 60)) / machines if (ref_speed > 0 and machines > 0) else 0
        
        # Get actual manhours from mapping or use calculated value
        actual_manhours = get_actual_manhours(base_section) or actual_hours
        
        # Calculate direct operators per machine/shift
        total_operators_hours = sum(
            detail.get('hours', 0) for detail in analysis['machine_details']
        )
        dir_per_shift = (total_operators_hours / actual_hours) if actual_hours > 0 else 0
        
        # Calculate required manhours
        required_manhours = dir_per_shift * FIXED_MC_HOURS * machines
        
        # Calculate organizational losses
        org_losses = (actual_manhours / required_manhours - 1) if required_manhours > 0 else None
        
        row = {
            'Process': base_section,
            '# Mach. Avail.': machines,
            'Production Volume (Weekly)': weekly_volume,
            'Meas. Unit': 'pk',
            'Value Adding Mc Hours / Week / Mc': value_adding_hours,
            'Ref Speed (Meas. Unit per min)': ref_speed,
            'Actual OEE': value_adding_hours / FIXED_MC_HOURS if FIXED_MC_HOURS > 0 else None,
            'Actual Mc Hours / Week / Mc': FIXED_MC_HOURS,
            'Saturation vs. 110 hrs/wk': (FIXED_MC_HOURS / FACTORY_SHIFT_PATTERN) * 100,
            'Dir op/ mach /shift': 1,  # Fixed to 1 as per requirements
            'Required Manhours/ Week': required_manhours,
            'Actual Manhours/ Week': actual_manhours,
            'Org. Losses': org_losses,
            '% Overtime': 3.0,
            '% Absence': 2.0,
            'Actual No of Dir Ops': machines * 1  # Using fixed dir_per_shift of 1
        }
        report_data.append(row)
    
    # Create DataFrame and sort by Process name
    df = pd.DataFrame(report_data)
    df = df.sort_values('Process')
    
    
    
    return df

def generate_claude_analysis(sections_analysis: Dict[str, Any]) -> str:
    """Generate AI analysis using Claude"""
    section_summaries = []
    for section_name, analysis in sections_analysis.items():
        summary = (
            f"Section: {section_name}\n"
            f"Machines: {analysis['summary']['total_machines']}\n"
            f"Hours: {analysis['summary']['total_hours']:.2f}\n"
            f"Production Volume: {analysis['summary']['production']['total_packets']:,.0f} packets\n"
            f"Efficiency: {analysis['performance']['average_efficiency']:.1f}%"
        )
        section_summaries.append(summary)
    
    prompt = (
        "Analyze the following production data and provide insights on efficiency, "
        "capacity utilization, and potential improvements:\n\n"
        f"{chr(10).join(section_summaries)}\n\n"
        "Consider:\n"
        "1. Overall equipment effectiveness\n"
        "2. Resource utilization\n"
        "3. Bottlenecks and constraints\n"
        "4. Improvement opportunities"
    )

    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system="You are a production analysis expert. Provide concise, actionable insights.",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        return response.content
    except Exception as e:
        logger.error(f"Error getting Claude analysis: {e}")
        return "Error generating analysis. Please try again."

def analyze_production_data(sections_data: Dict[str, pd.DataFrame], date_range: str) -> Dict[str, Any]:
    """Analyze aggregated production data for each section"""
    sections_analysis = {}
    
    for section_key, df in sections_data.items():
        analysis = {
            'date_range': date_range,
            'section_name': section_key,
            'summary': {
                'total_machines': len(df['Machine_No'].unique()),
                'total_hours': df['Hours'].sum(),
                'total_operator_cost': df['Operator_Cost'].sum(),
                'production': {
                    'total_per_pack': df['Per_Pack'].sum(),
                    'total_bags': df['Bag_Produce'].sum(),
                    'total_packets': df['Packet_Produce'].sum(),
                    'total_kgs': df['In_Kgs'].sum(),
                },
                'targets': {
                    'total_bag_target': df['target_Bag_Produce'].sum(),
                    'total_pkt_target': df['Pkt'].sum(),
                    'total_kg_target': df['KG_Target'].sum(),
                }
            },
            'performance': {
                'average_efficiency': df['Pkt_Var_%'].mean(),
                'machines_above_target': len(df[df['Pkt_Var_%'] > 100]),
                'machines_below_target': len(df[df['Pkt_Var_%'] <= 100]),
            },
            'machine_details': []
        }
        
        for _, row in df.iterrows():
            machine_data = {
                'machine_no': row['Machine_No'],
                'supervisor': row['Supervisor'],
                'operator': row['NAME'],
                'hours': row['Hours'],
                'operator_cost': row['Operator_Cost'],
                'production': {
                    'per_pack': row['Per_Pack'],
                    'bags': row['Bag_Produce'],
                    'packets': row['Packet_Produce'],
                    'kgs': row['In_Kgs']
                },
                'material': {
                    'sap_code': row['SAP_Code'],
                    'net_weight': row['Net_Weight'],
                    'size': row['Size'],
                    'description': row['Material_Description']
                },
                'targets': {
                    'bag_target': row['target_Bag_Produce'],
                    'pkt_target': row['Pkt'],
                    'kg_target': row['KG_Target']
                },
                'dates': row['Production_Date'],
                'efficiency': row['Pkt_Var_%'],
                'kg_efficiency': row['KG_Variance_%']
            }
            analysis['machine_details'].append(machine_data)
        
        sections_analysis[section_key] = analysis
    
    return sections_analysis

def read_production_data(uploaded_files: List[Any], max_sheets: int) -> Tuple[Dict[str, pd.DataFrame], List[Any], int, Dict[str, int]]:
    """Read and process production data from uploaded files"""
    try:
        all_sections = {}
        all_dates = set()
        file_stats = {
            'total_files': len(uploaded_files),
            'processed_files': 0,
            'total_sheets': 0,
            'processed_sheets': 0
        }
        
        # Count total sheets first
        for uploaded_file in uploaded_files:
            try:
                excel_file = read_file(uploaded_file)
                file_stats['total_sheets'] += len(excel_file.sheet_names)
            except Exception as e:
                logger.warning(f"Error counting sheets in {uploaded_file.name}: {e}")
            finally:
                # Reset file position for later reading
                uploaded_file.seek(0)
        
        sheets_data, skipped_sheets = batch_process_sheets(uploaded_files, max_sheets, file_stats)
        section_data_collection = {}
        
        columns = [
            'Machine_No', 'Supervisor', 'Hours', 'Operator_Cost', 'NAME', 'SAP_Code',
            'Net_Weight', 'Size', 'Material_Description', 'Per_Pack', 'Bag_Produce',
            'Packet_Produce', 'In_Kgs', 'target_Bag_Produce', 'Pkt', 'KG_Target',
            'Pkt_Var_%', 'KG_Variance_%'
        ]
        
        for df_full, production_date in sheets_data:
            sections = identify_sections(df_full)
            if isinstance(production_date, str):
                try:
                    # Try to parse the date if it's a string
                    production_date = pd.to_datetime(production_date)
                except:
                    pass
            all_dates.add(production_date)
            
            for section in sections:
                section_key = f"{section['name']} ({section['shift']})"
                df_section = df_full.iloc[section['start']:section['end']].copy()
                
                if not df_section.empty:
                    df_section = process_section_data(df_section, columns)
                    df_section['Production_Date'] = production_date
                    
                    if section_key not in section_data_collection:
                        section_data_collection[section_key] = []
                    section_data_collection[section_key].append(df_section)
        
        # Aggregate data for each section
        for section_key, df_list in section_data_collection.items():
            all_sections[section_key] = aggregate_machine_data(df_list)
        
        return all_sections, sorted(list(all_dates)), skipped_sheets, file_stats
    
    except Exception as e:
        logger.error(f"Error reading production data: {e}")
        raise Exception(f"Error reading production data: {e}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Capacity Labour Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Capacity Labour Dashboard")
    st.write("Upload production files for analysis")
    
    # Add configuration section
    with st.expander("Configuration", expanded=True):
        max_sheets = st.number_input(
            "Maximum number of sheets to process",
            min_value=1,
            max_value=1000,
            value=28,
            help="Set the maximum number of sheets to process from all uploaded files. Higher numbers will take longer to process."
        )
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['xlsx', 'xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt', 'csv'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        try:
            with st.spinner('Processing production data...'):
                # Read and process the uploaded files with user-defined limit
                sections_data, date_range, skipped_sheets, file_stats = read_production_data(uploaded_files, max_sheets)
                
                # Display file processing statistics
                st.success(f"Successfully processed {file_stats['processed_files']} files containing {file_stats['processed_sheets']} sheets")
                
                if skipped_sheets > 0:
                    st.warning(f"Skipped {skipped_sheets} sheets due to reaching the configured limit of {max_sheets} sheets")
                
                # Display date range with proper formatting
                st.subheader("Analysis Period")
                date_col1, date_col2 = st.columns(2)
                with date_col1:
                    start_date = min(date_range)
                    if isinstance(start_date, pd.Timestamp):
                        start_date = start_date.strftime('%Y-%m-%d')
                    st.write(f"Start Date: {start_date}")
                with date_col2:
                    end_date = max(date_range)
                    if isinstance(end_date, pd.Timestamp):
                        end_date = end_date.strftime('%Y-%m-%d')
                    st.write(f"End Date: {end_date}")
                
                # Generate analysis for all sections
                sections_analysis = analyze_production_data(
                    sections_data,
                    f"{min(date_range)} - {max(date_range)}"
                )
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Section Overview", 
                    "Machine Details", 
                    "Capacity Report",
                    "AI Analysis"
                ])
                
                with tab1:
                    st.subheader("Section Performance Overview")
                    for section_name, analysis in sections_analysis.items():
                        with st.expander(f"{section_name}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Total Machines", 
                                    analysis['summary']['total_machines']
                                )
                                st.metric(
                                    "Total Hours", 
                                    f"{analysis['summary']['total_hours']:,.1f}"
                                )
                                
                            with col2:
                                st.metric(
                                    "Total Packets", 
                                    f"{analysis['summary']['production']['total_packets']:,.0f}"
                                )
                                st.metric(
                                    "Average Efficiency", 
                                    f"{analysis['performance']['average_efficiency']:.1f}%"
                                )
                                
                            with col3:
                                st.metric(
                                    "Machines Above Target", 
                                    analysis['performance']['machines_above_target']
                                )
                                st.metric(
                                    "Machines Below Target", 
                                    analysis['performance']['machines_below_target']
                                )
                
                with tab2:
                    st.subheader("Machine Details")
                    selected_section = st.selectbox(
                        "Select Section",
                        options=list(sections_data.keys()),
                        key="section_selector"
                    )
                    
                    if selected_section:
                        # Display machine data
                        machine_data = sections_data[selected_section]
                        st.dataframe(
                            machine_data,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Create efficiency charts
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            st.subheader("Packet Efficiency by Machine")
                            efficiency_chart = pd.DataFrame({
                                'Machine': machine_data['Machine_No'],
                                'Efficiency (%)': machine_data['Pkt_Var_%']
                            }).set_index('Machine')
                            st.bar_chart(efficiency_chart)
                        
                        with chart_col2:
                            st.subheader("KG Efficiency by Machine")
                            kg_efficiency_chart = pd.DataFrame({
                                'Machine': machine_data['Machine_No'],
                                'KG Efficiency (%)': machine_data['KG_Variance_%']
                            }).set_index('Machine')
                            st.bar_chart(kg_efficiency_chart)
                
                with tab3:
                    st.subheader("Capacity Report")
                    capacity_report = transform_to_capacity_report(sections_analysis)
                    
                    # Display capacity report
                    st.dataframe(
                        capacity_report,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download options
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        csv = capacity_report.to_csv(index=False)
                        st.download_button(
                            "Download CSV Report",
                            csv,
                            "capacity_report.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    
                    with download_col2:
                        # Fix for Excel download
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            capacity_report.to_excel(writer, index=False)
                            writer.close()
                        
                        st.download_button(
                            "Download Excel Report",
                            buffer.getvalue(),
                            "capacity_report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='download-excel'
                        )
                
                with tab4:
                    st.subheader("AI Analysis")
                    if st.button("Generate AI Analysis", key="generate_analysis"):
                        with st.spinner("Generating analysis..."):
                            analysis = generate_claude_analysis(sections_analysis)
                            st.markdown(analysis)
                            
                            # Option to download analysis
                            st.download_button(
                                "Download Analysis",
                                analysis,
                                "production_analysis.txt",
                                "text/plain",
                                key='download-analysis'
                            )
        
        except Exception as e:
            st.error(f"Error processing files: {e}")
            logger.error(f"Application error: {e}")
            st.stop()
    
    else:
        st.info("Please upload Excel files to begin analysis")
    
    # Add footer with timestamp and version info
    st.markdown("---")
    footer_col1, footer_col2 = st.columns(2)
    with footer_col1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with footer_col2:
        st.caption("Capacity Labour Analysis Dashboard v1.0")

if __name__ == "__main__":
    main()