#!/usr/bin/env python3
"""
Kaptio Travel Analytics Script
=============================

This script analyzes Salesforce data to provide insights into Kaptio Travel system
performance, user behavior, and operational efficiency based on the feedback analysis.

Author: Data Analytics Team
Date: 2025-01-14
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from simple_salesforce import Salesforce, SalesforceLogin
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kaptio_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SalesforceConfig:
    """Configuration for Salesforce connection"""
    username: str
    password: str
    security_token: str
    domain: str = 'login'  # Use 'test' for sandbox
    api_version: str = '58.0'

@dataclass
class AnalyticsConfig:
    """Configuration for analytics parameters"""
    lookback_days: int = 30
    critical_duration_minutes: int = 60
    timeout_threshold_minutes: int = 30
    alert_email: Optional[str] = None
    export_path: str = './exports'

class KaptioAnalytics:
    """Main analytics class for Kaptio Travel data analysis"""
    
    def __init__(self, sf_config: SalesforceConfig, analytics_config: AnalyticsConfig):
        self.sf_config = sf_config
        self.analytics_config = analytics_config
        self.sf = None
        self.data_cache = {}
        
    def connect_salesforce(self) -> None:
        """Establish connection to Salesforce"""
        try:
            logger.info("Connecting to Salesforce...")
            session_id, instance = SalesforceLogin(
                username=self.sf_config.username,
                password=self.sf_config.password,
                security_token=self.sf_config.security_token,
                domain=self.sf_config.domain
            )
            self.sf = Salesforce(instance=instance, session_id=session_id)
            logger.info("Successfully connected to Salesforce")
        except Exception as e:
            logger.error(f"Failed to connect to Salesforce: {e}")
            raise
    
    def get_callout_requests(self, days_back: int = None) -> pd.DataFrame:
        """Extract CalloutRequest data for performance analysis"""
        days_back = days_back or self.analytics_config.lookback_days
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            Id, CreatedDate, CreatedById, LastModifiedDate, LastModifiedById,
            KaptioTravel__State__c, KaptioTravel__CompletionDateTime__c,
            KaptioTravel__ErrorDetails__c, KaptioTravel__ExternalId__c,
            KaptioTravel__InitializationDateTime__c, KaptioTravel__InitiatedBy__c,
            KaptioTravel__Itinerary__c, KaptioTravel__LastBatchReceivedDateTime__c,
            KaptioTravel__Outdated__c,
            KaptioTravel__InitiatedBy__r.Name, KaptioTravel__InitiatedBy__r.Profile.Name,
            KaptioTravel__Itinerary__r.Name, KaptioTravel__Itinerary__r.KaptioTravel__Group_Size__c
        FROM KaptioTravel__CalloutRequest__c 
        WHERE CreatedDate >= {start_date}T00:00:00Z
        ORDER BY CreatedDate DESC
        """
        
        logger.info("Fetching CalloutRequest data...")
        result = self.sf.query_all(query)
        
        if result['totalSize'] == 0:
            logger.warning("No CalloutRequest records found")
            return pd.DataFrame()
        
        records = []
        for record in result['records']:
            # Clean the record data
            clean_record = {k: v for k, v in record.items() if not k.startswith('attributes')}
            
            # Flatten nested objects
            if clean_record.get('KaptioTravel__InitiatedBy__r'):
                clean_record['InitiatedBy_Name'] = clean_record['KaptioTravel__InitiatedBy__r']['Name']
                clean_record['InitiatedBy_Profile'] = clean_record['KaptioTravel__InitiatedBy__r']['Profile']['Name'] if clean_record['KaptioTravel__InitiatedBy__r'].get('Profile') else None
                del clean_record['KaptioTravel__InitiatedBy__r']
            
            if clean_record.get('KaptioTravel__Itinerary__r'):
                clean_record['Itinerary_Name'] = clean_record['KaptioTravel__Itinerary__r']['Name']
                clean_record['Itinerary_GroupSize'] = clean_record['KaptioTravel__Itinerary__r']['KaptioTravel__Group_Size__c']
                del clean_record['KaptioTravel__Itinerary__r']
            
            records.append(clean_record)
        
        df = pd.DataFrame(records)
        
        # Convert datetime columns
        datetime_cols = ['CreatedDate', 'LastModifiedDate', 'KaptioTravel__CompletionDateTime__c',
                        'KaptioTravel__InitializationDateTime__c', 'KaptioTravel__LastBatchReceivedDateTime__c']
        
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # Calculate performance metrics
        df['Duration_Minutes'] = (df['KaptioTravel__CompletionDateTime__c'] - 
                                 df['KaptioTravel__InitializationDateTime__c']).dt.total_seconds() / 60
        
        df['TimeToFirstResponse_Minutes'] = (df['KaptioTravel__LastBatchReceivedDateTime__c'] - 
                                            df['KaptioTravel__InitializationDateTime__c']).dt.total_seconds() / 60
        
        # Add derived fields
        df['IsTimeout'] = df['KaptioTravel__ErrorDetails__c'].str.contains('timeout|Timeout|TIMEOUT', na=False)
        df['IsLongRunning'] = df['Duration_Minutes'] > self.analytics_config.critical_duration_minutes
        df['IsFailed'] = df['KaptioTravel__State__c'].isin(['Failed', 'Error'])
        df['IsSuccessful'] = df['KaptioTravel__State__c'] == 'Completed'
        
        logger.info(f"Fetched {len(df)} CalloutRequest records")
        return df
    
    def get_itinerary_data(self, days_back: int = None) -> pd.DataFrame:
        """Extract Itinerary data for business analysis"""
        days_back = days_back or self.analytics_config.lookback_days
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            Id, Name, CreatedDate, CreatedById, LastModifiedDate, LastModifiedById,
            KaptioTravel__Status__c, KaptioTravel__RecalculationStatus__c,
            KaptioTravel__RecalculationErrors__c, KaptioTravel__Start_Date__c,
            KaptioTravel__End_Date__c, KaptioTravel__Group_Size__c,
            KaptioTravel__Account__r.Name, CreatedBy.Name, CreatedBy.Profile.Name,
            KaptioTravel__Channel__r.Name
        FROM KaptioTravel__Itinerary__c 
        WHERE LastModifiedDate >= {start_date}T00:00:00Z
        ORDER BY LastModifiedDate DESC
        LIMIT 10000
        """
        
        logger.info("Fetching Itinerary data...")
        result = self.sf.query_all(query)
        
        if result['totalSize'] == 0:
            return pd.DataFrame()
        
        records = []
        for record in result['records']:
            clean_record = {k: v for k, v in record.items() if not k.startswith('attributes')}
            
            # Flatten nested objects
            if clean_record.get('KaptioTravel__Account__r'):
                clean_record['Account_Name'] = clean_record['KaptioTravel__Account__r']['Name']
                del clean_record['KaptioTravel__Account__r']
            
            if clean_record.get('CreatedBy'):
                clean_record['CreatedBy_Name'] = clean_record['CreatedBy']['Name']
                clean_record['CreatedBy_Profile'] = clean_record['CreatedBy']['Profile']['Name'] if clean_record['CreatedBy'].get('Profile') else None
                del clean_record['CreatedBy']
            
            if clean_record.get('KaptioTravel__Channel__r'):
                clean_record['Channel_Name'] = clean_record['KaptioTravel__Channel__r']['Name']
                del clean_record['KaptioTravel__Channel__r']
            
            records.append(clean_record)
        
        df = pd.DataFrame(records)
        
        # Convert datetime columns
        datetime_cols = ['CreatedDate', 'LastModifiedDate', 'KaptioTravel__Start_Date__c', 'KaptioTravel__End_Date__c']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # Add business logic fields
        df['HasRecalculationErrors'] = df['KaptioTravel__RecalculationErrors__c'].notna()
        df['IsInProgress'] = df['KaptioTravel__RecalculationStatus__c'].str.contains('Progress', na=False)
        
        logger.info(f"Fetched {len(df)} Itinerary records")
        return df
    
    def get_async_requests(self, days_back: int = None) -> pd.DataFrame:
        """Extract AsyncRequest data for background job analysis"""
        days_back = days_back or self.analytics_config.lookback_days
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            Id, CreatedDate, CreatedById, LastModifiedDate,
            KaptioTravel__AsyncType__c, KaptioTravel__Error__c,
            KaptioTravel__ErrorMessage__c, KaptioTravel__Params__c,
            CreatedBy.Name, CreatedBy.Profile.Name
        FROM KaptioTravel__AsyncRequest__c 
        WHERE CreatedDate >= {start_date}T00:00:00Z
        ORDER BY CreatedDate DESC
        LIMIT 5000
        """
        
        logger.info("Fetching AsyncRequest data...")
        result = self.sf.query_all(query)
        
        if result['totalSize'] == 0:
            return pd.DataFrame()
        
        records = []
        for record in result['records']:
            clean_record = {k: v for k, v in record.items() if not k.startswith('attributes')}
            
            if clean_record.get('CreatedBy'):
                clean_record['CreatedBy_Name'] = clean_record['CreatedBy']['Name']
                clean_record['CreatedBy_Profile'] = clean_record['CreatedBy']['Profile']['Name'] if clean_record['CreatedBy'].get('Profile') else None
                del clean_record['CreatedBy']
            
            records.append(clean_record)
        
        df = pd.DataFrame(records)
        
        # Convert datetime columns
        datetime_cols = ['CreatedDate', 'LastModifiedDate']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        logger.info(f"Fetched {len(df)} AsyncRequest records")
        return df
    
    def get_user_activity(self, days_back: int = None) -> pd.DataFrame:
        """Extract User activity data"""
        days_back = days_back or self.analytics_config.lookback_days
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            Id, Name, Email, Profile.Name, UserRole.Name, 
            IsActive, LastLoginDate, CreatedDate,
            TimeZoneSidKey, LocaleSidKey, LanguageLocaleKey
        FROM User 
        WHERE IsActive = true 
        AND LastLoginDate >= {start_date}T00:00:00Z
        ORDER BY LastLoginDate DESC
        LIMIT 1000
        """
        
        logger.info("Fetching User activity data...")
        result = self.sf.query_all(query)
        
        if result['totalSize'] == 0:
            return pd.DataFrame()
        
        records = []
        for record in result['records']:
            clean_record = {k: v for k, v in record.items() if not k.startswith('attributes')}
            
            if clean_record.get('Profile'):
                clean_record['Profile_Name'] = clean_record['Profile']['Name']
                del clean_record['Profile']
            
            if clean_record.get('UserRole'):
                clean_record['UserRole_Name'] = clean_record['UserRole']['Name']
                del clean_record['UserRole']
            
            records.append(clean_record)
        
        df = pd.DataFrame(records)
        
        # Convert datetime columns
        datetime_cols = ['LastLoginDate', 'CreatedDate']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        logger.info(f"Fetched {len(df)} User records")
        return df
    
    def analyze_performance_metrics(self, callout_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance metrics from CalloutRequest data"""
        if callout_df.empty:
            return {}
        
        # Filter out records without completion time for duration analysis
        completed_df = callout_df[callout_df['Duration_Minutes'].notna()]
        
        metrics = {
            'total_requests': len(callout_df),
            'completed_requests': len(completed_df),
            'failed_requests': len(callout_df[callout_df['IsFailed']]),
            'timeout_requests': len(callout_df[callout_df['IsTimeout']]),
            'long_running_requests': len(callout_df[callout_df['IsLongRunning']]),
            
            'success_rate': len(callout_df[callout_df['IsSuccessful']]) / len(callout_df) * 100 if len(callout_df) > 0 else 0,
            'timeout_rate': len(callout_df[callout_df['IsTimeout']]) / len(callout_df) * 100 if len(callout_df) > 0 else 0,
            
            'avg_duration_minutes': completed_df['Duration_Minutes'].mean() if not completed_df.empty else 0,
            'median_duration_minutes': completed_df['Duration_Minutes'].median() if not completed_df.empty else 0,
            'p95_duration_minutes': completed_df['Duration_Minutes'].quantile(0.95) if not completed_df.empty else 0,
            'max_duration_minutes': completed_df['Duration_Minutes'].max() if not completed_df.empty else 0,
            
            'avg_time_to_first_response': callout_df['TimeToFirstResponse_Minutes'].mean(),
            'requests_over_1_hour': len(completed_df[completed_df['Duration_Minutes'] > 60]),
            'requests_over_30_minutes': len(completed_df[completed_df['Duration_Minutes'] > 30]),
        }
        
        # Daily trends
        callout_df['Date'] = callout_df['CreatedDate'].dt.date
        daily_stats = callout_df.groupby('Date').agg({
            'Id': 'count',
            'IsFailed': 'sum',
            'IsTimeout': 'sum',
            'Duration_Minutes': 'mean'
        }).reset_index()
        daily_stats.columns = ['Date', 'Daily_Requests', 'Daily_Failures', 'Daily_Timeouts', 'Avg_Duration']
        
        metrics['daily_trends'] = daily_stats.to_dict('records')
        
        # User-based analysis
        user_stats = callout_df.groupby('InitiatedBy_Name').agg({
            'Id': 'count',
            'IsFailed': 'sum',
            'IsTimeout': 'sum',
            'Duration_Minutes': 'mean'
        }).reset_index()
        user_stats.columns = ['User', 'Total_Requests', 'Failures', 'Timeouts', 'Avg_Duration']
        user_stats = user_stats.sort_values('Total_Requests', ascending=False).head(10)
        
        metrics['top_users'] = user_stats.to_dict('records')
        
        return metrics
    
    def analyze_user_behavior(self, user_df: pd.DataFrame, callout_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        if user_df.empty:
            return {}
        
        # Active users analysis
        user_df['Days_Since_Login'] = (datetime.now(user_df['LastLoginDate'].dt.tz) - user_df['LastLoginDate']).dt.days
        
        behavior_metrics = {
            'total_active_users': len(user_df),
            'users_logged_in_last_7_days': len(user_df[user_df['Days_Since_Login'] <= 7]),
            'users_logged_in_last_24_hours': len(user_df[user_df['Days_Since_Login'] <= 1]),
            
            'profile_distribution': user_df['Profile_Name'].value_counts().to_dict(),
            'timezone_distribution': user_df['TimeZoneSidKey'].value_counts().head(10).to_dict(),
        }
        
        # User engagement with pricing system
        if not callout_df.empty:
            user_engagement = callout_df.groupby('InitiatedBy_Name').agg({
                'Id': 'count',
                'IsFailed': 'sum',
                'IsSuccessful': 'sum',
                'Duration_Minutes': 'mean'
            }).reset_index()
            
            user_engagement['Success_Rate'] = user_engagement['IsSuccessful'] / user_engagement['Id'] * 100
            user_engagement['Failure_Rate'] = user_engagement['IsFailed'] / user_engagement['Id'] * 100
            
            behavior_metrics['user_engagement'] = user_engagement.to_dict('records')
            
            # Identify problem users (high failure rates)
            problem_users = user_engagement[
                (user_engagement['Id'] >= 5) & 
                (user_engagement['Failure_Rate'] > 20)
            ].sort_values('Failure_Rate', ascending=False)
            
            behavior_metrics['problem_users'] = problem_users.to_dict('records')
        
        return behavior_metrics
    
    def generate_alerts(self, performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on performance thresholds"""
        alerts = []
        
        # Performance alerts
        if performance_metrics.get('timeout_rate', 0) > 10:
            alerts.append({
                'type': 'HIGH_TIMEOUT_RATE',
                'severity': 'HIGH',
                'message': f"Timeout rate is {performance_metrics['timeout_rate']:.1f}% (threshold: 10%)",
                'timestamp': datetime.now().isoformat()
            })
        
        if performance_metrics.get('avg_duration_minutes', 0) > 30:
            alerts.append({
                'type': 'SLOW_PERFORMANCE',
                'severity': 'MEDIUM',
                'message': f"Average calculation time is {performance_metrics['avg_duration_minutes']:.1f} minutes (threshold: 30 min)",
                'timestamp': datetime.now().isoformat()
            })
        
        if performance_metrics.get('requests_over_1_hour', 0) > 0:
            alerts.append({
                'type': 'CRITICAL_DURATION',
                'severity': 'CRITICAL',
                'message': f"{performance_metrics['requests_over_1_hour']} calculations took over 1 hour",
                'timestamp': datetime.now().isoformat()
            })
        
        if performance_metrics.get('success_rate', 100) < 80:
            alerts.append({
                'type': 'LOW_SUCCESS_RATE',
                'severity': 'HIGH',
                'message': f"Success rate is {performance_metrics['success_rate']:.1f}% (threshold: 80%)",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def create_performance_dashboard(self, callout_df: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        """Create Streamlit dashboard for performance metrics"""
        st.title("🚀 Kaptio  Performance Analytics for Intrepid Travel")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Success Rate", 
                f"{metrics.get('success_rate', 0):.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Avg Duration", 
                f"{metrics.get('avg_duration_minutes', 0):.1f} min",
                delta=None
            )
        
        with col3:
            st.metric(
                "Timeout Rate", 
                f"{metrics.get('timeout_rate', 0):.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Total Requests", 
                f"{metrics.get('total_requests', 0):,}",
                delta=None
            )
        
        # Performance trends
        if not callout_df.empty and 'daily_trends' in metrics:
            st.subheader("📊 Daily Performance Trends")
            
            daily_df = pd.DataFrame(metrics['daily_trends'])
            daily_df['Date'] = pd.to_datetime(daily_df['Date'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Daily Requests', 'Average Duration', 'Daily Failures', 'Daily Timeouts'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Daily requests
            fig.add_trace(
                go.Scatter(x=daily_df['Date'], y=daily_df['Daily_Requests'], name='Requests'),
                row=1, col=1
            )
            
            # Average duration
            fig.add_trace(
                go.Scatter(x=daily_df['Date'], y=daily_df['Avg_Duration'], name='Avg Duration (min)'),
                row=1, col=2
            )
            
            # Daily failures
            fig.add_trace(
                go.Scatter(x=daily_df['Date'], y=daily_df['Daily_Failures'], name='Failures'),
                row=2, col=1
            )
            
            # Daily timeouts
            fig.add_trace(
                go.Scatter(x=daily_df['Date'], y=daily_df['Daily_Timeouts'], name='Timeouts'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Duration distribution
        if not callout_df.empty:
            st.subheader("⏱️ Calculation Duration Distribution")
            
            completed_df = callout_df[callout_df['Duration_Minutes'].notna()]
            if not completed_df.empty:
                fig = px.histogram(
                    completed_df, 
                    x='Duration_Minutes', 
                    nbins=50,
                    title="Distribution of Calculation Times",
                    labels={'Duration_Minutes': 'Duration (Minutes)', 'count': 'Number of Requests'}
                )
                fig.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="30 min threshold")
                fig.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="1 hour threshold")
                st.plotly_chart(fig, use_container_width=True)
        
        # User performance analysis
        if 'top_users' in metrics and metrics['top_users']:
            st.subheader("👤 Top Users by Activity")
            
            users_df = pd.DataFrame(metrics['top_users'])
            fig = px.bar(
                users_df.head(10), 
                x='User', 
                y='Total_Requests',
                title="Most Active Users",
                color='Failures',
                color_continuous_scale='Reds'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    def export_data(self) -> None:
        """Export analyzed data to files"""
        export_dir = self.analytics_config.export_path
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export cached data
        for data_type, df in self.data_cache.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{data_type}_{timestamp}.csv"
                filepath = os.path.join(export_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Exported {data_type} data to {filepath}")
    
    def send_alert_email(self, alerts: List[Dict[str, Any]]) -> None:
        """Send email alerts for critical issues"""
        if not alerts or not self.analytics_config.alert_email:
            return
        
        critical_alerts = [a for a in alerts if a['severity'] in ['CRITICAL', 'HIGH']]
        if not critical_alerts:
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = 'kaptio-analytics@company.com'
            msg['To'] = self.analytics_config.alert_email
            msg['Subject'] = f'Kaptio Travel - {len(critical_alerts)} Critical Alerts'
            
            body = "Critical performance alerts detected:\n\n"
            for alert in critical_alerts:
                body += f"• {alert['type']}: {alert['message']}\n"
            
            msg.attach(MimeText(body, 'plain'))
            
            # Note: Configure SMTP server details as needed
            logger.info(f"Would send email alert with {len(critical_alerts)} critical alerts")
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        logger.info("Starting Kaptio Travel analytics pipeline...")
        
        # Connect to Salesforce
        if not self.sf:
            self.connect_salesforce()
        
        # Extract data
        logger.info("Extracting data from Salesforce...")
        callout_df = self.get_callout_requests()
        itinerary_df = self.get_itinerary_data()
        async_df = self.get_async_requests()
        user_df = self.get_user_activity()
        
        # Cache data for export
        self.data_cache = {
            'callout_requests': callout_df,
            'itineraries': itinerary_df,
            'async_requests': async_df,
            'users': user_df
        }
        
        # Analyze performance
        logger.info("Analyzing performance metrics...")
        performance_metrics = self.analyze_performance_metrics(callout_df)
        
        # Analyze user behavior
        logger.info("Analyzing user behavior...")
        behavior_metrics = self.analyze_user_behavior(user_df, callout_df)
        
        # Generate alerts
        alerts = self.generate_alerts(performance_metrics)
        
        # Send critical alerts
        if alerts:
            self.send_alert_email(alerts)
        
        # Prepare results
        results = {
            'performance_metrics': performance_metrics,
            'behavior_metrics': behavior_metrics,
            'alerts': alerts,
            'data_summary': {
                'callout_requests': len(callout_df),
                'itineraries': len(itinerary_df),
                'async_requests': len(async_df),
                'active_users': len(user_df)
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Analysis complete!")
        return results

def main():
    """Main function to run the analytics"""
    
    # Configuration - update these with your Salesforce credentials
    sf_config = SalesforceConfig(
        username=os.getenv('SF_USERNAME', 'your.email@company.com'),
        password=os.getenv('SF_PASSWORD', 'your_password'),
        security_token=os.getenv('SF_SECURITY_TOKEN', 'your_security_token'),
        domain=os.getenv('SF_DOMAIN', 'login')  # 'test' for sandbox
    )
    
    analytics_config = AnalyticsConfig(
        lookback_days=30,
        critical_duration_minutes=60,
        timeout_threshold_minutes=30,
        alert_email=os.getenv('ALERT_EMAIL'),
        export_path='./exports'
    )
    
    # Initialize analytics
    analytics = KaptioAnalytics(sf_config, analytics_config)
    
    # Check if running in Streamlit
    if len(sys.argv) > 1 and sys.argv[1] == 'streamlit':
        # Streamlit interface
        st.set_page_config(
            page_title="Kaptio Travel Analytics",
            page_icon="🚀",
            layout="wide"
        )
        
        # Run analysis
        with st.spinner('Connecting to Salesforce and analyzing data...'):
            results = analytics.run_analysis()
        
        # Display dashboard
        analytics.create_performance_dashboard(
            analytics.data_cache.get('callout_requests', pd.DataFrame()),
            results['performance_metrics']
        )
        
        # Show alerts
        if results['alerts']:
            st.subheader("🚨 Active Alerts")
            for alert in results['alerts']:
                if alert['severity'] == 'CRITICAL':
                    st.error(f"**{alert['type']}**: {alert['message']}")
                elif alert['severity'] == 'HIGH':
                    st.warning(f"**{alert['type']}**: {alert['message']}")
                else:
                    st.info(f"**{alert['type']}**: {alert['message']}")
        
        # Export button
        if st.button('Export Data'):
            analytics.export_data()
            st.success('Data exported successfully!')
    
    else:
        # Command line interface
        results = analytics.run_analysis()
        
        # Print summary
        print("\n" + "="*50)
        print("KAPTIO TRAVEL ANALYTICS SUMMARY")
        print("="*50)
        
        perf = results['performance_metrics']
        print(f"Total Requests: {perf.get('total_requests', 0):,}")
        print(f"Success Rate: {perf.get('success_rate', 0):.1f}%")
        print(f"Average Duration: {perf.get('avg_duration_minutes', 0):.1f} minutes")
        print(f"Timeout Rate: {perf.get('timeout_rate', 0):.1f}%")
        print(f"Requests >1 hour: {perf.get('requests_over_1_hour', 0)}")
        
        if results['alerts']:
            print(f"\nACTIVE ALERTS: {len(results['alerts'])}")
            for alert in results['alerts']:
                print(f"  • {alert['severity']}: {alert['message']}")
        
        # Export data
        analytics.export_data()
        
        # Save results
        with open(f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main() 