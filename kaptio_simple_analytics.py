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
from dotenv import load_dotenv
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
    logger.warning(f"PDF functionality not available: {e}")
    PDF_AVAILABLE = False

# Load environment variables
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
            KaptioTravel__InitiatedBy__r.Name
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
    
    def get_detailed_itinerary_data(self, callout_df, limit=20):
        """Get detailed itinerary characteristics for the longest calculating requests"""
        if callout_df.empty:
            return pd.DataFrame()
        
        try:
            # Get the longest running calculations
            longest_calculations = callout_df[callout_df['Duration_Minutes'].notna()].nlargest(limit, 'Duration_Minutes')
            
            if longest_calculations.empty:
                return pd.DataFrame()
            
            # Extract callout request IDs
            callout_ids = longest_calculations['Id'].tolist()
            callout_ids_str = "'" + "','".join(callout_ids) + "'"
            
            # Query CalloutRequest with direct Itinerary relationship
            query = f"""
            SELECT 
                Id,
                KaptioTravel__Itinerary__r.Name,
                KaptioTravel__Itinerary__r.KaptioTravel__No_of_days__c,
                KaptioTravel__Itinerary__r.Group_Size__c,
                KaptioTravel__Itinerary__r.Status__c,
                KaptioTravel__Itinerary__c
            FROM KaptioTravel__CalloutRequest__c 
            WHERE Id IN ({callout_ids_str})
            AND KaptioTravel__Itinerary__r.Name != null
            LIMIT {limit}
            """
            
            logger.info(f"Executing query: {query}")
            result = self.sf.query_all(query)
            
            if result['totalSize'] == 0:
                logger.warning("No itinerary data found, returning basic longest calculations")
                return longest_calculations[['Id', 'Duration_Minutes', 'User_Name', 'CreatedDate', 'KaptioTravel__State__c']]
            
            # Process the results
            itinerary_records = []
            itinerary_ids = set()
            
            for record in result['records']:
                itinerary_data = {}
                itinerary_data['CalloutRequest_Id'] = record.get('Id')
                
                if record.get('KaptioTravel__Itinerary__r'):
                    itin = record['KaptioTravel__Itinerary__r']
                    itinerary_data['Itinerary_Name'] = itin.get('Name', 'Unknown')
                    itinerary_data['Days'] = itin.get('KaptioTravel__No_of_days__c', 0)
                    itinerary_data['Pax'] = itin.get('Group_Size__c', 0)
                    itinerary_data['Status'] = itin.get('Status__c', 'Unknown')
                    itinerary_data['Itinerary_Id'] = record.get('KaptioTravel__Itinerary__c')
                    
                    if record.get('KaptioTravel__Itinerary__c'):
                        itinerary_ids.add(record.get('KaptioTravel__Itinerary__c'))
                else:
                    itinerary_data['Itinerary_Name'] = 'Unknown'
                    itinerary_data['Days'] = 0
                    itinerary_data['Pax'] = 0
                    itinerary_data['Status'] = 'Unknown'
                    itinerary_data['Itinerary_Id'] = None
                
                itinerary_records.append(itinerary_data)
            
            # Get departure counts for each itinerary
            departure_counts = {}
            if itinerary_ids:
                itinerary_ids_str = "'" + "','".join(itinerary_ids) + "'"
                departure_query = f"""
                SELECT 
                    KaptioTravel__Itinerary__c,
                    COUNT(Id) DepartureCount
                FROM KaptioTravel__TourDeparture__c 
                WHERE KaptioTravel__Itinerary__c IN ({itinerary_ids_str})
                GROUP BY KaptioTravel__Itinerary__c
                """
                
                try:
                    departure_result = self.sf.query_all(departure_query)
                    for dep_record in departure_result['records']:
                        departure_counts[dep_record['KaptioTravel__Itinerary__c']] = dep_record['DepartureCount']
                except Exception as e:
                    logger.warning(f"Could not get departure counts: {e}")
            
            # Add departure counts to itinerary records
            for record in itinerary_records:
                record['Departure_Count'] = departure_counts.get(record['Itinerary_Id'], 1)
            
            itinerary_df = pd.DataFrame(itinerary_records)
            
            # Merge with callout data
            detailed_df = longest_calculations.merge(
                itinerary_df, 
                left_on='Id', 
                right_on='CalloutRequest_Id', 
                how='left'
            )
            
            # Calculate complexity indicators
            if not detailed_df.empty:
                # Fill missing values
                detailed_df['Days'] = detailed_df['Days'].fillna(0)
                detailed_df['Pax'] = detailed_df['Pax'].fillna(0)
                detailed_df['Departure_Count'] = detailed_df['Departure_Count'].fillna(1)
                
                # Complexity score (weighted calculation)
                detailed_df['Complexity_Score'] = (
                    detailed_df['Days'] * 0.3 +
                    detailed_df['Pax'] * 0.2 +
                    detailed_df['Departure_Count'] * 0.5
                )
            
            logger.info(f"Successfully retrieved detailed data for {len(detailed_df)} longest calculations")
            return detailed_df[['Id', 'Duration_Minutes', 'User_Name', 'CreatedDate', 'KaptioTravel__State__c', 
                             'Itinerary_Name', 'Days', 'Pax', 'Departure_Count', 'Status', 'Complexity_Score']]
            
        except Exception as e:
            logger.error(f"Error getting detailed itinerary data: {e}")
            # Return basic longest calculations if detailed query fails
            return longest_calculations[['Id', 'Duration_Minutes', 'User_Name', 'CreatedDate', 'KaptioTravel__State__c']]
    
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
        
        # Continue with dashboard content
        self._render_dashboard_content(df, metrics, timeout_threshold, duration_threshold)
    
    def _render_dashboard_content(self, df, metrics, timeout_threshold=5.0, duration_threshold=30):
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
        
        # Top 20 Longest Calculating Itineraries
        st.subheader("🐌 Top 20 Longest Calculating Itineraries")
        
        with st.spinner("Loading detailed itinerary characteristics..."):
            detailed_itinerary_df = self.get_detailed_itinerary_data(df, limit=20)
        
        if not detailed_itinerary_df.empty and len(detailed_itinerary_df) > 0:
            # Display the detailed table
            display_columns = {
                'Duration_Minutes': 'Duration (min)',
                'Itinerary_Name': 'Itinerary',
                'Days': 'Days',
                'Pax': 'Pax',
                'Departure_Count': 'Departures',
                'Complexity_Score': 'Complexity',
                'User_Name': 'User',
                'Status': 'Status',
                'CreatedDate': 'Date'
            }
            
            # Format the data for display
            display_df = detailed_itinerary_df.copy()
            
            # Format duration with color coding
            if 'Duration_Minutes' in display_df.columns:
                display_df['Duration_Minutes'] = display_df['Duration_Minutes'].round(1)
            if 'Complexity_Score' in display_df.columns:
                display_df['Complexity_Score'] = display_df['Complexity_Score'].round(1)
            if 'CreatedDate' in display_df.columns:
                display_df['CreatedDate'] = pd.to_datetime(display_df['CreatedDate']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Rename columns for display
            display_df = display_df.rename(columns=display_columns)
            
            # Select only the columns we want to show
            show_cols = [col for col in display_columns.values() if col in display_df.columns]
            st.dataframe(
                display_df[show_cols], 
                use_container_width=True,
                hide_index=True
            )
            
            # Analysis insights
            if len(detailed_itinerary_df) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'Days' in detailed_itinerary_df.columns:
                        avg_days = detailed_itinerary_df['Days'].mean()
                        st.metric("Avg Days", f"{avg_days:.1f}")
                    else:
                        st.metric("Avg Days", "N/A")
                
                with col2:
                    if 'Departure_Count' in detailed_itinerary_df.columns:
                        avg_departures = detailed_itinerary_df['Departure_Count'].mean()
                        st.metric("Avg Departures", f"{avg_departures:.1f}")
                    else:
                        st.metric("Avg Departures", "N/A")
                
                with col3:
                    if 'Complexity_Score' in detailed_itinerary_df.columns:
                        avg_complexity = detailed_itinerary_df['Complexity_Score'].mean()
                        st.metric("Avg Complexity", f"{avg_complexity:.1f}")
                    else:
                        st.metric("Avg Complexity", "N/A")
                
                # Correlation analysis
                required_cols = ['Duration_Minutes', 'Days', 'Departure_Count', 'Complexity_Score']
                has_required_cols = all(col in detailed_itinerary_df.columns for col in required_cols)
                
                if len(detailed_itinerary_df) >= 5 and has_required_cols:  # Need minimum data points for correlation
                    st.markdown("**📊 Complexity vs Duration Analysis:**")
                    
                    # Create scatter plot
                    try:
                        fig = px.scatter(
                            detailed_itinerary_df,
                            x='Complexity_Score',
                            y='Duration_Minutes',
                            size='Departure_Count',
                            color='Days',
                            hover_data=['Itinerary_Name', 'User_Name'] + (['Pax'] if 'Pax' in detailed_itinerary_df.columns else []),
                            title="Calculation Duration vs Itinerary Complexity",
                            labels={
                                'Complexity_Score': 'Complexity Score',
                                'Duration_Minutes': 'Duration (Minutes)',
                                'Days': 'Trip Days'
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation if we have numeric data
                        correlation = detailed_itinerary_df[required_cols].corr()['Duration_Minutes']
                        
                        st.markdown("**🔗 Correlation with Calculation Time:**")
                        st.markdown(f"- **Days**: {correlation['Days']:.3f}")
                        st.markdown(f"- **Departure Options**: {correlation['Departure_Count']:.3f}")  
                        st.markdown(f"- **Overall Complexity**: {correlation['Complexity_Score']:.3f}")
                        
                        st.info("📝 **Note**: Data retrieved from actual Salesforce Itinerary records linked to the longest calculating CalloutRequests.")
                        
                    except Exception as e:
                        st.warning(f"Could not generate correlation analysis: {e}")
                else:
                    st.info("Need at least 5 records with complete data for correlation analysis")
        else:
            st.warning("⚠️ No detailed itinerary data available. This could be because:")
            st.markdown("- No CalloutRequests are linked to Itinerary records")
            st.markdown("- No long-running calculations found in the selected time period") 
            st.markdown("- Insufficient permissions to access itinerary data")
        
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