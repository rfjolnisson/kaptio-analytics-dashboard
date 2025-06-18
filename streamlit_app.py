#!/usr/bin/env python3
"""
Kaptio Travel Analytics - Simple Version
========================================

This script analyzes Salesforce data to provide insights into Kaptio Travel system
performance based on the user feedback about timeout issues and slow performance.

Usage:
    python kaptio_simple_analytics.py
    
Or for Streamlit dashboard:
    streamlit run kaptio_simple_analytics.py
"""

import os
import sys
import json
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from simple_salesforce import Salesforce

# Optional imports for PDF functionality
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from io import BytesIO
    import base64
    PDF_AVAILABLE = True
except ImportError as e:
    # PDF functionality not available - this is optional
    PDF_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaptioAnalytics:
    """Simple analytics class for Kaptio Travel performance analysis"""
    
    def __init__(self):
        self.sf = None
        
    def connect_salesforce(self):
        """Connect to Salesforce using environment variables"""
        try:
            self.sf = Salesforce(
                username=os.getenv('SF_USERNAME'),
                password=os.getenv('SF_PASSWORD'),
                security_token=os.getenv('SF_SECURITY_TOKEN'),
                domain=os.getenv('SF_DOMAIN', 'login')
            )
            logger.info("Connected to Salesforce successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Salesforce: {e}")
            return False
    
    def get_callout_performance(self, days_back=30, user_filter=None, state_filter=None):
        """Get CalloutRequest performance data with optional filters"""
        
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Build dynamic query with filters
        where_conditions = [f"CreatedDate >= {start_date}T00:00:00Z"]
        
        if user_filter and user_filter != "All Users":
            where_conditions.append(f"KaptioTravel__InitiatedBy__r.Name = '{user_filter}'")
        
        if state_filter and state_filter != "All States":
            where_conditions.append(f"KaptioTravel__State__c = '{state_filter}'")
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        SELECT 
            Id, CreatedDate, 
            KaptioTravel__State__c,
            KaptioTravel__InitializationDateTime__c,
            KaptioTravel__CompletionDateTime__c,
            KaptioTravel__ErrorDetails__c,
            KaptioTravel__InitiatedBy__r.Name,
            KaptioTravel__Itinerary__c
        FROM KaptioTravel__CalloutRequest__c 
        WHERE {where_clause}
        ORDER BY CreatedDate DESC
        LIMIT 5000
        """
        
        logger.info("Fetching CalloutRequest data...")
        result = self.sf.query_all(query)
        
        if result['totalSize'] == 0:
            return pd.DataFrame()
        
        # Convert to DataFrame
        records = []
        for record in result['records']:
            flat_record = {}
            for key, value in record.items():
                if key == 'attributes':
                    continue
                elif key == 'KaptioTravel__InitiatedBy__r' and value:
                    flat_record['User_Name'] = value.get('Name', 'Unknown')
                else:
                    flat_record[key] = value
            records.append(flat_record)
        
        df = pd.DataFrame(records)
        
        # Convert datetime columns
        datetime_cols = ['CreatedDate', 'KaptioTravel__InitializationDateTime__c', 'KaptioTravel__CompletionDateTime__c']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate duration in minutes
        if 'KaptioTravel__InitializationDateTime__c' in df.columns and 'KaptioTravel__CompletionDateTime__c' in df.columns:
            df['Duration_Minutes'] = (
                df['KaptioTravel__CompletionDateTime__c'] - 
                df['KaptioTravel__InitializationDateTime__c']
            ).dt.total_seconds() / 60
        
        # Add flags for analysis
        df['IsTimeout'] = df['KaptioTravel__ErrorDetails__c'].str.contains('timeout|Timeout|TIMEOUT', na=False)
        df['IsLongRunning'] = df['Duration_Minutes'] > 60  # Over 1 hour
        df['IsFailed'] = df['KaptioTravel__State__c'].isin(['Failed', 'Error'])
        df['IsSuccessful'] = df['KaptioTravel__State__c'] == 'Completed'
        
        logger.info(f"Fetched {len(df)} CalloutRequest records")
        return df
    
    def get_filter_options(self, days_back=90):
        """Get available users and states for filtering (using longer period for options)"""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Get unique users
            user_query = f"""
            SELECT KaptioTravel__InitiatedBy__r.Name
            FROM KaptioTravel__CalloutRequest__c 
            WHERE CreatedDate >= {start_date}T00:00:00Z
            AND KaptioTravel__InitiatedBy__r.Name != null
            GROUP BY KaptioTravel__InitiatedBy__r.Name
            ORDER BY KaptioTravel__InitiatedBy__r.Name
            LIMIT 200
            """
            
            user_result = self.sf.query_all(user_query)
            users = ["All Users"] + [record['KaptioTravel__InitiatedBy__r']['Name'] 
                                   for record in user_result['records'] 
                                   if record.get('KaptioTravel__InitiatedBy__r')]
            
            # Get unique states
            state_query = f"""
            SELECT KaptioTravel__State__c
            FROM KaptioTravel__CalloutRequest__c 
            WHERE CreatedDate >= {start_date}T00:00:00Z
            AND KaptioTravel__State__c != null
            GROUP BY KaptioTravel__State__c
            ORDER BY KaptioTravel__State__c
            """
            
            state_result = self.sf.query_all(state_query)
            states = ["All States"] + [record['KaptioTravel__State__c'] 
                                     for record in state_result['records']]
            
            return users, states
            
        except Exception as e:
            logger.error(f"Error getting filter options: {e}")
            return ["All Users"], ["All States"]
    
    def analyze_performance(self, df):
        """Analyze performance metrics"""
        if df.empty:
            return {}
        
        # Overall metrics
        total_requests = len(df)
        successful_requests = len(df[df['IsSuccessful']])
        failed_requests = len(df[df['IsFailed']])
        timeout_requests = len(df[df['IsTimeout']])
        long_running_requests = len(df[df['IsLongRunning']])
        
        # Duration statistics (only for completed requests)
        completed_df = df[df['Duration_Minutes'].notna()]
        
        metrics = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'timeout_requests': timeout_requests,
            'long_running_requests': long_running_requests,
            
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            'failure_rate': (failed_requests / total_requests * 100) if total_requests > 0 else 0,
            'timeout_rate': (timeout_requests / total_requests * 100) if total_requests > 0 else 0,
            
            'avg_duration_minutes': completed_df['Duration_Minutes'].mean() if not completed_df.empty else 0,
            'median_duration_minutes': completed_df['Duration_Minutes'].median() if not completed_df.empty else 0,
            'max_duration_minutes': completed_df['Duration_Minutes'].max() if not completed_df.empty else 0,
            'p95_duration_minutes': completed_df['Duration_Minutes'].quantile(0.95) if not completed_df.empty else 0,
        }
        
        return metrics
    
    def get_itinerary_complexity_data(self, itinerary_ids):
        """Get complexity metrics for itineraries linked to CalloutRequests"""
        if itinerary_ids is None or len(itinerary_ids) == 0:
            return pd.DataFrame()
        
        # Convert to list and create query-safe string
        id_list = list(itinerary_ids)
        if len(id_list) == 0:
            return pd.DataFrame()
        
        # Build query with proper escaping
        id_string = "','".join([str(id) for id in id_list])
        
        try:
            # Query for itinerary basic data
            itinerary_query = f"""
            SELECT Id, Name, KaptioTravel__Start_Date__c, KaptioTravel__End_Date__c,
                   KaptioTravel__GroupSizes__c, KaptioTravel__GroupTravel__c
            FROM KaptioTravel__Itinerary__c 
            WHERE Id IN ('{id_string}')
            """
            
            logger.info(f"Fetching itinerary data for {len(id_list)} itineraries...")
            itinerary_result = self.sf.query_all(itinerary_query)
            
            if itinerary_result['totalSize'] == 0:
                logger.warning("No itinerary data found")
                return pd.DataFrame()
            
            # Query for itinerary items (services)
            items_query = f"""
            SELECT KaptioTravel__Itinerary__c, Id,
                   KaptioTravel__DateFrom__c, KaptioTravel__DateTo__c,
                   KaptioTravel__Item__r.Name, KaptioTravel__Price_Category__r.Name,
                   KaptioTravel__Package__c, KaptioTravel__PackageDeparture__c
            FROM KaptioTravel__Itinerary_Item__c 
            WHERE KaptioTravel__Itinerary__c IN ('{id_string}')
            AND KaptioTravel__Mode__c = 'Active'
            """
            
            logger.info("Fetching itinerary items data...")
            items_result = self.sf.query_all(items_query)
            
            # Query for configurations (via ConfigurationGroup relationship)
            config_query = f"""
            SELECT KaptioTravel__ConfigurationGroup__r.KaptioTravel__Itinerary__c, Id, Name, KaptioTravel__ConfigurationGroup__c
            FROM KaptioTravel__ItineraryConfiguration__c 
            WHERE KaptioTravel__ConfigurationGroup__r.KaptioTravel__Itinerary__c IN ('{id_string}')
            """
            
            logger.info("Fetching configuration data...")
            try:
                config_result = self.sf.query_all(config_query)
            except Exception as e:
                logger.warning(f"Could not fetch configuration data with relationship field: {e}")
                # Try alternative field name
                try:
                    logger.info("Trying alternative configuration query...")
                    alt_config_query = f"""
                    SELECT Id, Name, KaptioTravel__ConfigurationGroup__c
                    FROM KaptioTravel__ItineraryConfiguration__c 
                    LIMIT 10
                    """
                    test_result = self.sf.query_all(alt_config_query)
                    logger.info(f"Alternative config query successful - found {test_result['totalSize']} total configs")
                    
                    # If this works, we know the object exists but the field name is wrong
                    if test_result['totalSize'] > 0:
                        logger.warning("ItineraryConfiguration object exists but relationship field name is incorrect")
                    config_result = {'records': []}
                except Exception as e2:
                    logger.warning(f"ItineraryConfiguration object might not exist: {e2}")
                    config_result = {'records': []}
            
            # Query for tour departures
            departure_query = f"""
            SELECT KaptioTravel__Itinerary__r.Id, Id, KaptioTravel__DepartureDate__c
            FROM KaptioTravel__TourDeparture__c 
            WHERE KaptioTravel__Itinerary__r.Id IN ('{id_string}')
            """
            
            logger.info("Fetching departure data...")
            try:
                departure_result = self.sf.query_all(departure_query)
            except Exception as e:
                logger.warning(f"Could not fetch departure data: {e}")
                departure_result = {'records': []}
            
            # Process the data
            complexity_data = []
            
            for itinerary in itinerary_result['records']:
                itin_id = itinerary['Id']
                
                # Count services
                services_count = len([item for item in items_result['records'] 
                                    if item['KaptioTravel__Itinerary__c'] == itin_id])
                
                # Calculate days
                start_date = itinerary.get('KaptioTravel__Start_Date__c')
                end_date = itinerary.get('KaptioTravel__End_Date__c')
                days_count = 0
                if start_date and end_date:
                    try:
                        start = pd.to_datetime(start_date).date()
                        end = pd.to_datetime(end_date).date()
                        days_count = (end - start).days
                    except:
                        days_count = 0
                
                # Count configurations
                configurations_count = len([config for config in config_result['records'] 
                                          if config.get('KaptioTravel__ConfigurationGroup__r', {}).get('KaptioTravel__Itinerary__c') == itin_id])
                
                # Count departures
                departures_count = len([dep for dep in departure_result['records'] 
                                      if dep.get('KaptioTravel__Itinerary__r', {}).get('Id') == itin_id])
                
                # Parse group sizes
                group_sizes = itinerary.get('KaptioTravel__GroupSizes__c', '')
                group_size_variations = 0
                if group_sizes:
                    try:
                        group_size_variations = len([size.strip() for size in group_sizes.split(',') if size.strip()])
                    except:
                        group_size_variations = 0
                
                # Calculate complexity factors
                factors = self._calculate_complexity_factors(
                    services_count, days_count, configurations_count, 
                    departures_count, group_size_variations
                )
                
                # Build data record with individual factors
                data_record = {
                    'Itinerary_Id': itin_id,
                    'Itinerary_Name': itinerary.get('Name', ''),
                    'Services_Count': services_count,
                    'Days_Count': days_count,
                    'Configurations_Count': configurations_count,
                    'Departures_Count': departures_count,
                    'GroupSize_Variations': group_size_variations,
                }
                
                # Add all complexity factors
                data_record.update(factors)
                complexity_data.append(data_record)
            
            logger.info(f"Processed complexity data for {len(complexity_data)} itineraries")
            return pd.DataFrame(complexity_data)
            
        except Exception as e:
            logger.error(f"Error fetching complexity data: {e}")
            return pd.DataFrame()
    
    def _calculate_complexity_factors(self, services, days, configs, departures, group_sizes):
        """Calculate individual complexity factors and identify risk levels"""
        factors = {
            'services_count': services,
            'days_count': days,
            'configs_count': configs,
            'departures_count': departures,
            'group_size_variations': group_sizes,
            
            # Risk level classifications based on likely breakpoints
            'high_service_load': services > 25,  # Hypothesis: >25 services cause issues
            'config_heavy': configs > 10,        # Hypothesis: >10 configs slow things down
            'multi_departure': departures > 3,   # Hypothesis: >3 departures add complexity
            'long_duration': days > 14,          # Hypothesis: >2 weeks trips are complex
            'variable_groups': group_sizes > 3,  # Hypothesis: >3 group size options complicate pricing
            
            # Business context flags
            'likely_group_travel': services > 15 and days > 7,  # Group travel characteristics
            'likely_individual': services < 8 and days < 7,     # Individual travel characteristics
            'likely_corporate': configs > 5 and departures > 2,  # Corporate travel patterns
        }
        
        return factors
    
    def analyze_performance_drivers(self, df_callouts, df_complexity):
        """Analyze what actually drives performance issues - actionable insights"""
        if df_callouts.empty or df_complexity.empty:
            return {}
        
        # Merge the datasets on Itinerary
        merged_df = df_callouts.merge(
            df_complexity, 
            left_on='KaptioTravel__Itinerary__c', 
            right_on='Itinerary_Id', 
            how='inner'
        )
        
        logger.info(f"Callouts data: {len(df_callouts)} records")
        logger.info(f"Complexity data: {len(df_complexity)} records")
        logger.info(f"Merged data: {len(merged_df)} records")
        
        if merged_df.empty:
            logger.warning("No matching records between callouts and complexity data")
            logger.warning(f"Sample callout itinerary IDs: {df_callouts['KaptioTravel__Itinerary__c'].dropna().head().tolist()}")
            logger.warning(f"Sample complexity itinerary IDs: {df_complexity['Itinerary_Id'].head().tolist()}")
            return {}
        
        # Get completed requests only for performance analysis
        completed_df = merged_df[merged_df['Duration_Minutes'].notna()]
        
        if completed_df.empty:
            return {'error': 'No completed requests with duration data'}
        
        analysis = {
            'total_analyzed': len(merged_df),
            'completed_requests': len(completed_df),
        }
        
        # 1. INDIVIDUAL FACTOR CORRELATIONS - What actually matters?
        factors = ['Services_Count', 'Days_Count', 'Configurations_Count', 'Departures_Count', 'GroupSize_Variations']
        factor_analysis = {}
        
        for factor in factors:
            if factor in completed_df.columns:
                corr_duration = completed_df[factor].corr(completed_df['Duration_Minutes'])
                corr_timeout = completed_df[factor].corr(completed_df['IsTimeout'].astype(int))
                
                # R-squared for explained variance
                r_squared = corr_duration ** 2 if pd.notna(corr_duration) else 0
                
                factor_analysis[factor] = {
                    'correlation_duration': corr_duration,
                    'correlation_timeout': corr_timeout,
                    'r_squared': r_squared,
                    'explanation_power': 'High' if r_squared > 0.25 else 'Medium' if r_squared > 0.1 else 'Low'
                }
        
        analysis['individual_factors'] = factor_analysis
        
        # 2. THRESHOLD ANALYSIS - Find the breakpoints where things go wrong
        threshold_analysis = {}
        
        for factor in factors:
            if factor in completed_df.columns and completed_df[factor].max() > completed_df[factor].min():
                # Find natural breakpoints using quartiles
                factor_data = completed_df[factor]
                q25, q50, q75 = factor_data.quantile([0.25, 0.5, 0.75])
                
                # Performance by factor ranges
                ranges = {
                    f'Low (≤{q25:.0f})': completed_df[factor_data <= q25]['Duration_Minutes'].mean(),
                    f'Medium ({q25:.0f}-{q75:.0f})': completed_df[(factor_data > q25) & (factor_data <= q75)]['Duration_Minutes'].mean(),
                    f'High (>{q75:.0f})': completed_df[factor_data > q75]['Duration_Minutes'].mean()
                }
                
                # Find the problematic threshold
                low_avg = ranges[f'Low (≤{q25:.0f})']
                high_avg = ranges[f'High (>{q75:.0f})']
                performance_degradation = (high_avg / low_avg) if low_avg > 0 else 1
                
                threshold_analysis[factor] = {
                    'ranges': ranges,
                    'performance_degradation': performance_degradation,
                    'breakpoint_severity': 'Critical' if performance_degradation > 3 else 'High' if performance_degradation > 2 else 'Moderate'
                }
        
        analysis['threshold_analysis'] = threshold_analysis
        
        # 3. BUSINESS CONTEXT ANALYSIS - Group by business patterns
        business_patterns = {}
        
        # Group Travel Analysis
        group_travel = completed_df[completed_df['likely_group_travel'] == True]
        individual_travel = completed_df[completed_df['likely_individual'] == True]
        corporate_travel = completed_df[completed_df['likely_corporate'] == True]
        
        for pattern_name, pattern_df in [
            ('Group Travel', group_travel), 
            ('Individual Travel', individual_travel), 
            ('Corporate Travel', corporate_travel)
        ]:
            if not pattern_df.empty:
                business_patterns[pattern_name] = {
                    'count': len(pattern_df),
                    'avg_duration': pattern_df['Duration_Minutes'].mean(),
                    'timeout_rate': pattern_df['IsTimeout'].mean() * 100,
                    'avg_services': pattern_df['Services_Count'].mean(),
                    'avg_configs': pattern_df['Configurations_Count'].mean(),
                }
        
        analysis['business_patterns'] = business_patterns
        
        # 4. RISK FLAG ANALYSIS - Which combinations are dangerous?
        risk_flags = ['high_service_load', 'config_heavy', 'multi_departure', 'long_duration', 'variable_groups']
        risk_analysis = {}
        
        for flag in risk_flags:
            if flag in completed_df.columns:
                flagged_requests = completed_df[completed_df[flag] == True]
                normal_requests = completed_df[completed_df[flag] == False]
                
                if not flagged_requests.empty and not normal_requests.empty:
                    risk_analysis[flag] = {
                        'flagged_count': len(flagged_requests),
                        'normal_count': len(normal_requests),
                        'flagged_avg_duration': flagged_requests['Duration_Minutes'].mean(),
                        'normal_avg_duration': normal_requests['Duration_Minutes'].mean(),
                        'performance_impact': flagged_requests['Duration_Minutes'].mean() / normal_requests['Duration_Minutes'].mean(),
                        'timeout_rate_flagged': flagged_requests['IsTimeout'].mean() * 100,
                        'timeout_rate_normal': normal_requests['IsTimeout'].mean() * 100,
                    }
        
        analysis['risk_flags'] = risk_analysis
        
        # 5. ACTIONABLE RECOMMENDATIONS
        recommendations = []
        
        # Find the most impactful factor
        if factor_analysis:
            highest_impact = max(factor_analysis.items(), key=lambda x: x[1]['r_squared'])
            if highest_impact[1]['r_squared'] > 0.1:
                factor_name = highest_impact[0].replace('_', ' ').title()
                recommendations.append(f"🎯 PRIMARY TARGET: {factor_name} explains {highest_impact[1]['r_squared']*100:.1f}% of performance variance")
        
        # Risk flag recommendations
        for flag, data in risk_analysis.items():
            if data['performance_impact'] > 2:
                flag_name = flag.replace('_', ' ').title()
                recommendations.append(f"🚨 HIGH RISK: {flag_name} causes {data['performance_impact']:.1f}x slower calculations")
        
        # Threshold recommendations
        for factor, data in threshold_analysis.items():
            if data['performance_degradation'] > 2:
                factor_name = factor.replace('_', ' ').title()
                recommendations.append(f"⚠️ THRESHOLD BREACH: {factor_name} shows {data['performance_degradation']:.1f}x degradation at high values")
        
        analysis['actionable_recommendations'] = recommendations
        
        return analysis
    
    def generate_pdf_report(self, df, metrics, filters, filename):
        """Generate a comprehensive PDF performance report"""
        if not PDF_AVAILABLE:
            logger.error("PDF functionality not available - missing required packages")
            return False
            
        try:
            # Create the PDF document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1f77b4')
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#2c3e50')
            )
            
            # Title
            story.append(Paragraph("🚀 Kaptio Travel Performance Report", title_style))
            story.append(Spacer(1, 12))
            
            # Report metadata
            story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph(f"<b>Time Period:</b> {filters['time_period']}", styles['Normal']))
            story.append(Paragraph(f"<b>User Filter:</b> {filters['user_filter']}", styles['Normal']))
            story.append(Paragraph(f"<b>State Filter:</b> {filters['state_filter']}", styles['Normal']))
            story.append(Paragraph(f"<b>Total Records:</b> {len(df):,}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("📊 Executive Summary", heading_style))
            
            summary_data = [
                ['Metric', 'Value', 'Status'],
                ['Total Requests', f"{metrics['total_requests']:,}", ''],
                ['Success Rate', f"{metrics['success_rate']:.1f}%", 
                 '✅ Good' if metrics['success_rate'] >= 95 else '⚠️ Attention' if metrics['success_rate'] >= 90 else '🚨 Critical'],
                ['Average Duration', f"{metrics['avg_duration_minutes']:.1f} min", 
                 '✅ Good' if metrics['avg_duration_minutes'] <= 15 else '⚠️ Attention' if metrics['avg_duration_minutes'] <= 30 else '🚨 Critical'],
                ['Timeout Rate', f"{metrics['timeout_rate']:.1f}%", 
                 '✅ Good' if metrics['timeout_rate'] <= 2 else '⚠️ Attention' if metrics['timeout_rate'] <= 5 else '🚨 Critical'],
                ['Long Running (>1hr)', f"{metrics['long_running_requests']}", 
                 '✅ Good' if metrics['long_running_requests'] == 0 else '🚨 Critical'],
                ['95th Percentile', f"{metrics['p95_duration_minutes']:.1f} min", ''],
                ['Max Duration', f"{metrics['max_duration_minutes']:.1f} min", '']
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Create and add charts
            story.append(Paragraph("📈 Performance Analysis", heading_style))
            
            # Duration histogram
            if not df.empty and 'Duration_Minutes' in df.columns:
                completed_df = df[df['Duration_Minutes'].notna()]
                if not completed_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(completed_df['Duration_Minutes'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
                    ax.set_xlabel('Duration (Minutes)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Calculation Times')
                    ax.axvline(x=30, color='orange', linestyle='--', label='30 min threshold')
                    ax.axvline(x=60, color='red', linestyle='--', label='1 hour threshold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Save chart as image
                    chart_buffer = BytesIO()
                    plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                    chart_buffer.seek(0)
                    plt.close()
                    
                    # Add chart to PDF
                    story.append(Image(chart_buffer, width=6*inch, height=3.6*inch))
                    story.append(Spacer(1, 20))
            
            # Daily trend analysis
            if not df.empty and 'CreatedDate' in df.columns:
                df['Date'] = pd.to_datetime(df['CreatedDate']).dt.date
                daily_stats = df.groupby('Date').agg({
                    'Id': 'count',
                    'IsFailed': 'sum',
                    'IsTimeout': 'sum'
                }).reset_index()
                
                if not daily_stats.empty:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Daily requests
                    ax1.plot(daily_stats['Date'], daily_stats['Id'], marker='o', color='#3498db', linewidth=2)
                    ax1.set_title('Daily Request Volume')
                    ax1.set_ylabel('Number of Requests')
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Daily failures and timeouts
                    ax2.plot(daily_stats['Date'], daily_stats['IsFailed'], marker='s', color='red', label='Failures', linewidth=2)
                    ax2.plot(daily_stats['Date'], daily_stats['IsTimeout'], marker='^', color='orange', label='Timeouts', linewidth=2)
                    ax2.set_title('Daily Failures and Timeouts')
                    ax2.set_ylabel('Count')
                    ax2.set_xlabel('Date')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    
                    # Save chart as image
                    trend_buffer = BytesIO()
                    plt.savefig(trend_buffer, format='png', dpi=300, bbox_inches='tight')
                    trend_buffer.seek(0)
                    plt.close()
                    
                    # Add chart to PDF
                    story.append(Image(trend_buffer, width=6*inch, height=4.8*inch))
                    story.append(Spacer(1, 20))
            
            # Top users analysis
            if 'User_Name' in df.columns:
                user_stats = df.groupby('User_Name').agg({
                    'Id': 'count',
                    'IsFailed': 'sum',
                    'IsTimeout': 'sum'
                }).reset_index()
                user_stats['Failure_Rate'] = (user_stats['IsFailed'] / user_stats['Id'] * 100).round(1)
                top_users = user_stats.sort_values('Id', ascending=False).head(10)
                
                story.append(Paragraph("👤 Top 10 Most Active Users", heading_style))
                
                user_data = [['User', 'Total Requests', 'Failures', 'Timeouts', 'Failure Rate %']]
                for _, row in top_users.iterrows():
                    user_data.append([
                        row['User_Name'][:25] + ('...' if len(row['User_Name']) > 25 else ''),
                        f"{row['Id']:,}",
                        f"{row['IsFailed']}",
                        f"{row['IsTimeout']}",
                        f"{row['Failure_Rate']:.1f}%"
                    ])
                
                user_table = Table(user_data, colWidths=[2*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
                user_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9)
                ]))
                
                story.append(user_table)
                story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("💡 Recommendations", heading_style))
            
            recommendations = []
            if metrics['timeout_rate'] > filters.get('timeout_threshold', 5.0):
                recommendations.append(f"🔥 CRITICAL: Timeout rate ({metrics['timeout_rate']:.1f}%) exceeds threshold - investigate system capacity")
            
            if metrics['avg_duration_minutes'] > filters.get('duration_threshold', 30):
                recommendations.append(f"⚡ HIGH: Average duration ({metrics['avg_duration_minutes']:.1f} min) too slow - optimize pricing engine")
            
            if metrics['long_running_requests'] > 0:
                recommendations.append(f"🎯 HIGH: {metrics['long_running_requests']} calculations >1 hour - implement time limits")
            
            if metrics['failure_rate'] > 15:
                recommendations.append(f"🛠️ MEDIUM: Failure rate ({metrics['failure_rate']:.1f}%) high - improve error handling")
            
            recommendations.append("📊 ONGOING: Monitor these metrics daily and set up automated alerts")
            
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return False
    
    def create_dashboard(self):
        """Create interactive Streamlit dashboard with filters"""
        
        st.title("🚀 Kaptio  Performance Analytics for Intrepid Travel")
        st.markdown("Analysis of pricing calculation performance and timeout issues")
        
        # Sidebar for filters
        st.sidebar.header("📊 Data Filters")
        
        # Time period selection
        time_periods = {
            "Last 7 days": 7,
            "Last 14 days": 14,
            "Last 30 days": 30,
            "Last 60 days": 60,
            "Last 90 days": 90
        }
        
        selected_period = st.sidebar.selectbox(
            "📅 Time Period",
            options=list(time_periods.keys()),
            index=2  # Default to 30 days
        )
        days_back = time_periods[selected_period]
        
        # Get filter options
        with st.spinner("Loading filter options..."):
            users, states = self.get_filter_options()
        
        # User filter
        selected_user = st.sidebar.selectbox(
            "👤 User Filter",
            options=users,
            index=0
        )
        
        # State filter
        selected_state = st.sidebar.selectbox(
            "🔄 State Filter", 
            options=states,
            index=0
        )
        
        # Performance thresholds
        st.sidebar.header("⚙️ Alert Thresholds")
        
        timeout_threshold = st.sidebar.slider(
            "🚨 Timeout Rate Alert (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Alert when timeout rate exceeds this percentage"
        )
        
        duration_threshold = st.sidebar.slider(
            "⏱️ Duration Alert (minutes)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help="Alert when average duration exceeds this threshold"
        )
        
        # Add refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Current Selection")
        st.sidebar.markdown(f"**Period:** {selected_period}")
        st.sidebar.markdown(f"**User:** {selected_user}")
        st.sidebar.markdown(f"**State:** {selected_state}")
        
        # Fetch data with filters
        with st.spinner(f"🔄 Fetching data for {selected_period.lower()}..."):
            df = self.get_callout_performance(
                days_back=days_back,
                user_filter=selected_user if selected_user != "All Users" else None,
                state_filter=selected_state if selected_state != "All States" else None
            )
        
        if df.empty:
            st.warning("⚠️ No data found for the selected filters. Try adjusting your selection.")
            return
        
        # Analyze data
        metrics = self.analyze_performance(df)
        
        # Update sidebar with data summary
        st.sidebar.markdown("### 📊 Data Summary")
        st.sidebar.metric("Total Records", len(df))
        st.sidebar.metric("Success Rate", f"{metrics['success_rate']:.1f}%")
        st.sidebar.metric("Avg Duration", f"{metrics['avg_duration_minutes']:.1f} min")
        
        # Export options
        st.sidebar.markdown("### 📥 Export Data")
        
        if st.sidebar.button("📊 Export Current Data"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"kaptio_analysis_{selected_period.replace(' ', '_').lower()}_{timestamp}"
            
            # Create export directory
            import os
            os.makedirs('./exports', exist_ok=True)
            
            # Export CSV
            csv_file = f'./exports/{filename}.csv'
            df.to_csv(csv_file, index=False)
            
            # Export metrics
            json_file = f'./exports/{filename}_metrics.json'
            export_metrics = {
                **metrics,
                'filters': {
                    'time_period': selected_period,
                    'user_filter': selected_user,
                    'state_filter': selected_state,
                    'timeout_threshold': timeout_threshold,
                    'duration_threshold': duration_threshold
                }
            }
            
            import json
            with open(json_file, 'w') as f:
                json.dump(export_metrics, f, indent=2, default=str)
            
            st.sidebar.success(f"✅ Data exported to:\n- {csv_file}\n- {json_file}")
        
        # PDF Export (only if libraries are available)
        if PDF_AVAILABLE:
            if st.sidebar.button("📄 Generate PDF Report"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                pdf_filename = f"./exports/kaptio_performance_report_{selected_period.replace(' ', '_').lower()}_{timestamp}.pdf"
                
                with st.spinner("🔄 Generating PDF report..."):
                    filter_info = {
                        'time_period': selected_period,
                        'user_filter': selected_user,
                        'state_filter': selected_state,
                        'timeout_threshold': timeout_threshold,
                        'duration_threshold': duration_threshold
                    }
                    
                    success = self.generate_pdf_report(df, metrics, filter_info, pdf_filename)
                    
                    if success:
                        st.sidebar.success(f"✅ PDF report generated:\n{pdf_filename}")
                        
                        # Provide download link
                        with open(pdf_filename, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                        
                        st.sidebar.download_button(
                            label="⬇️ Download PDF Report",
                            data=PDFbyte,
                            file_name=f"kaptio_report_{timestamp}.pdf",
                            mime='application/pdf'
                        )
                    else:
                        st.sidebar.error("❌ Failed to generate PDF report. Check logs for details.")
        else:
            st.sidebar.info("📄 PDF export requires additional packages.\nRun: pip install matplotlib reportlab kaleido")
        
        # Display current filters
        st.info(f"📊 Showing data for **{selected_period.lower()}** | User: **{selected_user}** | State: **{selected_state}** | **{len(df)} records**")
        
        # Complexity Analysis with Configuration Debug
        df_complexity = pd.DataFrame()
        complexity_stats = {}
        
        # Get complexity data if we have itinerary IDs
        if 'KaptioTravel__Itinerary__c' in df.columns:
            unique_itineraries = df['KaptioTravel__Itinerary__c'].dropna().unique()
            
            if len(unique_itineraries) > 0:
                with st.spinner(f"🔍 Analyzing complexity data for {len(unique_itineraries)} itineraries..."):
                    try:
                        df_complexity = self.get_itinerary_complexity_data(unique_itineraries)
                        
                        # Debug configuration issue
                        if not df_complexity.empty:
                            config_count = df_complexity['Configurations_Count'].sum()
                            if config_count == 0:
                                st.warning("🔧 **Configuration Debug**: No configuration data found. This could indicate:")
                                st.write("- The ItineraryConfiguration object might not exist or have different field names")
                                st.write("- The relationship field to Itinerary might be named differently")
                                st.write("- Permissions issue accessing configuration data")
                                
                                # Show what we did get
                                non_zero_data = []
                                if df_complexity['Services_Count'].sum() > 0:
                                    non_zero_data.append("✅ Services data retrieved")
                                if df_complexity['Departures_Count'].sum() > 0:
                                    non_zero_data.append("✅ Departures data retrieved") 
                                if df_complexity['Days_Count'].sum() > 0:
                                    non_zero_data.append("✅ Days data calculated")
                                
                                if non_zero_data:
                                    st.info("**Data successfully retrieved:**")
                                    for item in non_zero_data:
                                        st.write(item)
                            
                            try:
                                complexity_stats = self.analyze_performance_drivers(df, df_complexity)
                            except Exception as e:
                                logger.error(f"Error in complexity statistics calculation: {str(e)}")
                                st.error(f"Error in complexity analysis: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error fetching complexity data: {str(e)}")
                        st.error(f"Error fetching complexity data: {str(e)}")
        
        # Continue with dashboard content
        self._render_dashboard_content(df, metrics, df_complexity, complexity_stats, timeout_threshold, duration_threshold)
    
    def _render_dashboard_content(self, df, metrics, df_complexity, complexity_stats, timeout_threshold=5.0, duration_threshold=30):
        """Render the main dashboard content with dynamic thresholds"""
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Success Rate", 
                f"{metrics['success_rate']:.1f}%",
                delta=f"-{metrics['failure_rate']:.1f}% failures"
            )
        
        with col2:
            st.metric(
                "Avg Calculation Time", 
                f"{metrics['avg_duration_minutes']:.1f} min",
                delta=f"Max: {metrics['max_duration_minutes']:.0f} min"
            )
        
        with col3:
            st.metric(
                "Timeout Rate", 
                f"{metrics['timeout_rate']:.1f}%",
                delta=f"{metrics['timeout_requests']} timeouts"
            )
        
        with col4:
            st.metric(
                "Long Running (>1hr)", 
                f"{metrics['long_running_requests']}",
                delta=f"Out of {metrics['total_requests']} total"
            )
        
        # Performance Issues Summary
        st.subheader("🚨 Critical Issues Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.error(f"**{metrics['timeout_requests']} TIMEOUT ERRORS** - Causing user panic and system confidence issues")
            st.warning(f"**{metrics['long_running_requests']} CALCULATIONS >1 HOUR** - Unacceptable performance impacting deadlines")
            
        with col2:
            st.info(f"**{metrics['failure_rate']:.1f}% FAILURE RATE** - Users struggling to identify and correct errors")
            if metrics['avg_duration_minutes'] > 30:
                st.warning(f"**AVERAGE TIME {metrics['avg_duration_minutes']:.1f} MIN** - Price overview calculator too slow")
        
        # Detailed Analysis
        if not df.empty:
            st.subheader("📊 Performance Analysis")
            
            # Duration Distribution
            completed_df = df[df['Duration_Minutes'].notna()]
            if not completed_df.empty:
                fig = px.histogram(
                    completed_df, 
                    x='Duration_Minutes',
                    nbins=50,
                    title="Distribution of Calculation Times",
                    labels={'Duration_Minutes': 'Duration (Minutes)', 'count': 'Frequency'}
                )
                
                # Add threshold lines
                fig.add_vline(x=30, line_dash="dash", line_color="orange", 
                             annotation_text="30 min - User patience threshold")
                fig.add_vline(x=60, line_dash="dash", line_color="red", 
                             annotation_text="1 hour - Critical threshold")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Daily Trends
            df['Date'] = df['CreatedDate'].dt.date
            daily_stats = df.groupby('Date').agg({
                'Id': 'count',
                'IsFailed': 'sum',
                'IsTimeout': 'sum',
                'IsLongRunning': 'sum'
            }).reset_index()
            
            daily_stats.columns = ['Date', 'Total_Requests', 'Failures', 'Timeouts', 'Long_Running']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Requests', 'Daily Failures', 'Daily Timeouts', 'Long Running Requests'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(go.Scatter(x=daily_stats['Date'], y=daily_stats['Total_Requests'], 
                                   name='Total Requests', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=daily_stats['Date'], y=daily_stats['Failures'], 
                                   name='Failures', line=dict(color='red')), row=1, col=2)
            fig.add_trace(go.Scatter(x=daily_stats['Date'], y=daily_stats['Timeouts'], 
                                   name='Timeouts', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Scatter(x=daily_stats['Date'], y=daily_stats['Long_Running'], 
                                   name='Long Running', line=dict(color='purple')), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False, title_text="Daily Performance Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # User Analysis
            if 'User_Name' in df.columns:
                st.subheader("👤 User Impact Analysis")
                
                user_stats = df.groupby('User_Name').agg({
                    'Id': 'count',
                    'IsFailed': 'sum',
                    'IsTimeout': 'sum',
                    'IsLongRunning': 'sum'
                }).reset_index()
                
                user_stats.columns = ['User', 'Total_Requests', 'Failures', 'Timeouts', 'Long_Running']
                user_stats['Failure_Rate'] = (user_stats['Failures'] / user_stats['Total_Requests'] * 100).round(1)
                
                # Show users with high failure rates
                problem_users = user_stats[
                    (user_stats['Total_Requests'] >= 5) & 
                    (user_stats['Failure_Rate'] > 20)
                ].sort_values('Failure_Rate', ascending=False)
                
                if not problem_users.empty:
                    st.error("**Users Experiencing High Failure Rates (>20%)**")
                    st.dataframe(problem_users, use_container_width=True)
                
                # Top active users
                top_users = user_stats.sort_values('Total_Requests', ascending=False).head(10)
                
                fig = px.bar(
                    top_users, 
                    x='User', 
                    y='Total_Requests',
                    color='Failure_Rate',
                    color_continuous_scale='Reds',
                    title="Most Active Users (Colored by Failure Rate %)"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance Driver Analysis - Always show this section
        st.header("🎯 Performance Driver Analysis")
        st.markdown("""
        **What actually drives slow pricing calculations?** This analysis identifies specific, actionable factors 
        that impact performance - no more guessing or composite scores that don't tell you what to fix.
        """)
        
        if not df_complexity.empty and complexity_stats:
            
            # 1. ACTIONABLE RECOMMENDATIONS FIRST
            if 'actionable_recommendations' in complexity_stats and complexity_stats['actionable_recommendations']:
                st.subheader("🎯 Immediate Actions Based on Your Data")
                
                for i, recommendation in enumerate(complexity_stats['actionable_recommendations'], 1):
                    st.markdown(f"**{i}.** {recommendation}")
            
            # 2. INDIVIDUAL FACTOR ANALYSIS - R² shows what actually matters
            if 'individual_factors' in complexity_stats:
                st.subheader("📊 Which Factors Actually Drive Performance?")
                st.markdown("**R² = How much of the performance variance each factor explains**")
                
                factor_analysis = complexity_stats['individual_factors']
                factor_data = []
                
                for factor, data in factor_analysis.items():
                    factor_name = factor.replace('_', ' ').title()
                    factor_data.append({
                        'Factor': factor_name,
                        'R² (Explained Variance)': f"{data['r_squared']*100:.1f}%",
                        'Explanation Power': data['explanation_power'],
                        'Correlation with Duration': f"{data['correlation_duration']:.3f}",
                        'Correlation with Timeouts': f"{data['correlation_timeout']:.3f}"
                    })
                
                if factor_data:
                    st.dataframe(pd.DataFrame(factor_data), use_container_width=True)
                    
                    # Show R² visualization
                    r_squared_values = [data['r_squared'] for data in factor_analysis.values()]
                    factor_names = [factor.replace('_', ' ').title() for factor in factor_analysis.keys()]
                    
                    fig = px.bar(
                        x=factor_names,
                        y=[r*100 for r in r_squared_values],
                        title="Performance Variance Explained by Each Factor (R²)",
                        labels={'x': 'Factor', 'y': 'Explained Variance (%)'},
                        color=r_squared_values,
                        color_continuous_scale='RdYlGn'
                    )
                    fig.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="25% - High Impact Threshold")
                    fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="10% - Medium Impact Threshold")
                    st.plotly_chart(fig, use_container_width=True)
            
            # 3. THRESHOLD ANALYSIS - When do things break?
            if 'threshold_analysis' in complexity_stats:
                st.subheader("⚡ Performance Breakpoints - When Do Things Go Wrong?")
                
                threshold_data = complexity_stats['threshold_analysis']
                
                for factor, data in threshold_data.items():
                    factor_name = factor.replace('_', ' ').title()
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**{factor_name}**")
                        
                        # Create performance breakdown chart
                        ranges = list(data['ranges'].keys())
                        durations = list(data['ranges'].values())
                        
                        fig = px.bar(
                            x=ranges,
                            y=durations,
                            title=f"{factor_name} Performance Breakdown",
                            labels={'x': f'{factor_name} Range', 'y': 'Avg Duration (min)'},
                            color=durations,
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        degradation = data['performance_degradation']
                        severity = data['breakpoint_severity']
                        
                        if severity == 'Critical':
                            st.error(f"🚨 **{severity}**")
                        elif severity == 'High':
                            st.warning(f"⚠️ **{severity}**")
                        else:
                            st.info(f"ℹ️ **{severity}**")
                        
                        st.metric(
                            "Performance Degradation",
                            f"{degradation:.1f}x slower",
                            f"at high {factor_name.lower()}"
                        )
            
            # 4. BUSINESS CONTEXT ANALYSIS
            if 'business_patterns' in complexity_stats:
                st.subheader("🏢 Business Context Analysis")
                st.markdown("**How do different business travel patterns perform?**")
                
                business_data = complexity_stats['business_patterns']
                
                if business_data:
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (pattern_name, data) in enumerate(business_data.items()):
                        col = [col1, col2, col3][i % 3]
                        
                        with col:
                            st.markdown(f"**{pattern_name}**")
                            st.metric("Count", f"{data['count']}")
                            st.metric("Avg Duration", f"{data['avg_duration']:.1f} min")
                            st.metric("Timeout Rate", f"{data['timeout_rate']:.1f}%")
                            st.metric("Avg Services", f"{data['avg_services']:.1f}")
                            st.metric("Avg Configs", f"{data['avg_configs']:.1f}")
                    
                    # Business pattern comparison chart
                    pattern_names = list(business_data.keys())
                    durations = [data['avg_duration'] for data in business_data.values()]
                    timeout_rates = [data['timeout_rate'] for data in business_data.values()]
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Average Duration by Business Pattern', 'Timeout Rate by Business Pattern')
                    )
                    
                    fig.add_trace(go.Bar(x=pattern_names, y=durations, name='Duration (min)', marker_color='lightblue'), row=1, col=1)
                    fig.add_trace(go.Bar(x=pattern_names, y=timeout_rates, name='Timeout Rate (%)', marker_color='orange'), row=1, col=2)
                    
                    fig.update_layout(height=400, showlegend=False, title_text="Business Pattern Performance Comparison")
                    st.plotly_chart(fig, use_container_width=True)
            
            # 5. RISK FLAG ANALYSIS
            if 'risk_flags' in complexity_stats:
                st.subheader("🚨 Risk Flag Analysis")
                st.markdown("**Which risk conditions cause the biggest performance impact?**")
                
                risk_data = complexity_stats['risk_flags']
                
                if risk_data:
                    risk_summary = []
                    
                    for flag, data in risk_data.items():
                        flag_name = flag.replace('_', ' ').title()
                        risk_summary.append({
                            'Risk Condition': flag_name,
                            'Flagged Requests': data['flagged_count'],
                            'Normal Requests': data['normal_count'],
                            'Performance Impact': f"{data['performance_impact']:.1f}x slower",
                            'Flagged Avg Duration': f"{data['flagged_avg_duration']:.1f} min",
                            'Normal Avg Duration': f"{data['normal_avg_duration']:.1f} min",
                            'Flagged Timeout Rate': f"{data['timeout_rate_flagged']:.1f}%",
                            'Normal Timeout Rate': f"{data['timeout_rate_normal']:.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(risk_summary), use_container_width=True)
                    
                    # Risk impact visualization
                    impact_values = [data['performance_impact'] for data in risk_data.values()]
                    flag_names = [flag.replace('_', ' ').title() for flag in risk_data.keys()]
                    
                    fig = px.bar(
                        x=flag_names,
                        y=impact_values,
                        title="Performance Impact by Risk Condition",
                        labels={'x': 'Risk Condition', 'y': 'Performance Impact (x slower)'},
                        color=impact_values,
                        color_continuous_scale='Reds'
                    )
                    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="2x slower - Critical threshold")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            # Show what would be here if performance driver data was available
            st.warning("⚠️ **Performance Driver Analysis Not Available**")
            st.markdown("""
            **This section would normally show actionable insights instead of abstract metrics:**
            
            1. **🎯 Immediate Actions Based on Your Data**
               - Specific recommendations with R² values showing impact
               - Clear prioritization of what to optimize first
            
            2. **📊 Which Factors Actually Drive Performance?**
               - R² values showing % of variance explained by each factor
               - Color-coded charts showing High/Medium/Low impact factors
               - No more guessing - see exactly what matters
            
            3. **⚡ Performance Breakpoints - When Do Things Go Wrong?**
               - Specific thresholds where performance degrades (e.g., ">25 services = 3x slower")
               - Critical/High/Moderate severity classifications
               - Performance impact charts by factor ranges
            
            4. **🏢 Business Context Analysis**
               - Group Travel vs Individual vs Corporate performance patterns
               - Business-meaningful categories instead of abstract complexity scores
            
            5. **🚨 Risk Flag Analysis**
               - Which specific conditions cause 2x+ slower performance
               - Flagged vs normal request comparisons
               - Actionable risk condition identification
            
            **To enable this analysis:**
            - Ensure CalloutRequest records have linked Itinerary IDs
            - Verify access to itinerary and related object data
            
            **The key difference:** This gives you **what to fix** and **by how much**, not just "complexity correlates with duration" 🎯
            """)
        
        # Recommendations
        st.subheader("💡 Immediate Action Items")
        
        recommendations = []
        
        if metrics['timeout_rate'] > timeout_threshold:
            recommendations.append(f"🔥 **CRITICAL**: Timeout rate ({metrics['timeout_rate']:.1f}%) exceeds threshold ({timeout_threshold}%) - investigate system capacity and optimize long-running queries")
        
        if metrics['avg_duration_minutes'] > duration_threshold:
            recommendations.append(f"⚡ **HIGH**: Average calculation time ({metrics['avg_duration_minutes']:.1f} min) exceeds threshold ({duration_threshold} min) - review pricing engine performance")
        
        if metrics['long_running_requests'] > 0:
            recommendations.append("🎯 **HIGH**: Calculations taking >1 hour - implement calculation limits and user notifications")
        
        if metrics['failure_rate'] > 15:
            recommendations.append("🛠️ **MEDIUM**: High failure rate - improve error messaging and user guidance")
        
        recommendations.append("📊 **ONGOING**: Monitor these metrics daily and set up automated alerts for degradation")
        
        # Add performance driver-based recommendations
        if complexity_stats and 'actionable_recommendations' not in complexity_stats:
            # If we don't have the new analysis, show generic recommendations
            recommendations.append("📊 **ANALYSIS**: Enable performance driver analysis for specific, actionable recommendations")
        elif complexity_stats and 'actionable_recommendations' in complexity_stats:
            # The specific recommendations are already shown in the analysis section
            recommendations.append("📊 **ANALYSIS**: See 'Performance Driver Analysis' section above for specific, data-driven recommendations")
        
        for rec in recommendations:
            st.markdown(rec)
        
        # Export data
        if st.button("📥 Export Data"):
            # Create export directory
            os.makedirs('./exports', exist_ok=True)
            
            # Export CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_file = f'./exports/kaptio_performance_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            
            # Export metrics
            json_file = f'./exports/kaptio_metrics_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            st.success(f"Data exported to {csv_file} and {json_file}")

def main():
    """Main function"""
    
    analytics = KaptioAnalytics()
    
    # Check if we're in Streamlit mode
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and 'streamlit' in ' '.join(sys.argv)):
        # Streamlit mode - set page config first
        st.set_page_config(
            page_title="Kaptio Travel Performance Analytics",
            page_icon="🚀",
            layout="wide"
        )
        
        if not analytics.connect_salesforce():
            st.error("❌ Failed to connect to Salesforce. Please check your credentials in environment variables.")
            st.info("Set SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN in your .env file")
            return
        
        # Create interactive dashboard (handles data fetching internally)
        analytics.create_dashboard()
        
    else:
        # Command line mode
        if not analytics.connect_salesforce():
            print("Failed to connect to Salesforce")
            return
        
        print("Fetching data...")
        df = analytics.get_callout_performance()
        
        if df.empty:
            print("No data found")
            return
        
        print("Analyzing performance...")
        metrics = analytics.analyze_performance(df)
        
        # Analyze complexity if possible
        complexity_summary = ""
        if 'KaptioTravel__Itinerary__c' in df.columns:
            unique_itineraries = df['KaptioTravel__Itinerary__c'].dropna().unique()
            if len(unique_itineraries) > 0:
                print(f"Analyzing complexity for {len(unique_itineraries)} itineraries...")
                df_complexity = analytics.get_itinerary_complexity_data(unique_itineraries)
                if not df_complexity.empty:
                    complexity_stats = analytics.analyze_performance_drivers(df, df_complexity)
                    if complexity_stats:
                        if 'actionable_recommendations' in complexity_stats and complexity_stats['actionable_recommendations']:
                            complexity_summary = f"""
🎯 PERFORMANCE DRIVER ANALYSIS:
  Total Analyzed: {complexity_stats.get('total_analyzed', 0)} requests
  Completed Requests: {complexity_stats.get('completed_requests', 0)}
  
ACTIONABLE RECOMMENDATIONS:"""
                            for i, rec in enumerate(complexity_stats['actionable_recommendations'], 1):
                                complexity_summary += f"\n  {i}. {rec}"
                        else:
                            complexity_summary = f"""
🎯 PERFORMANCE DRIVER ANALYSIS:
  Total Analyzed: {complexity_stats.get('total_analyzed', 0)} requests  
  Completed Requests: {complexity_stats.get('completed_requests', 0)}
  Analysis complete - see detailed breakdown in dashboard mode."""
        
        # Print summary
        print("\n" + "="*60)
        print("KAPTIO TRAVEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"📊 Total Requests: {metrics['total_requests']:,}")
        print(f"✅ Success Rate: {metrics['success_rate']:.1f}%")
        print(f"⏱️ Average Duration: {metrics['avg_duration_minutes']:.1f} minutes")
        print(f"🚨 Timeout Rate: {metrics['timeout_rate']:.1f}%")
        print(f"🐌 Long Running (>1hr): {metrics['long_running_requests']}")
        print(f"📈 95th Percentile: {metrics['p95_duration_minutes']:.1f} minutes")
        print("="*60)
        
        # Print complexity analysis if available
        if complexity_summary:
            print(complexity_summary)
            print("="*60)
        
        # Critical alerts
        if metrics['timeout_rate'] > 5:
            print("🔥 CRITICAL ALERT: High timeout rate detected!")
        if metrics['long_running_requests'] > 0:
            print("🔥 CRITICAL ALERT: Calculations taking over 1 hour!")
        if metrics['avg_duration_minutes'] > 30:
            print("⚠️ WARNING: Average calculation time exceeds 30 minutes")
        
        # Save results
        os.makedirs('./exports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with open(f'./exports/analysis_results_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        df.to_csv(f'./exports/callout_data_{timestamp}.csv', index=False)
        print(f"\n📁 Results saved to ./exports/analysis_results_{timestamp}.json")

if __name__ == "__main__":
    main() 