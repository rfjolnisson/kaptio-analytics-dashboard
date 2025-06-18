# Kaptio Travel Performance Analytics

A comprehensive Python-based analytics solution for monitoring and analyzing Kaptio Travel system performance, specifically addressing the critical issues identified in user feedback:

- **Performance Issues**: 1+ hour price calculations
- **Timeout Errors**: Causing user panic and confidence loss
- **Complex Error Identification**: Difficulty in identifying costing errors
- **User Experience Problems**: Poor error messaging and slow responses

## Features

### 🚀 Real-time Performance Monitoring
- Price calculation duration analysis
- Timeout detection and tracking
- Success/failure rate monitoring
- User impact analysis

### 📊 Interactive Dashboard
- Live Streamlit dashboard with key metrics
- Performance trend visualization
- User behavior analysis
- Automated recommendations

### 🚨 Alerting System
- Critical performance threshold alerts
- Timeout spike detection
- User experience degradation warnings

### 📈 Business Intelligence
- Daily/weekly performance trends
- User productivity metrics
- Operational deadline impact analysis
- Problem user identification

## Installation

### Prerequisites
- Python 3.8+
- Access to Kaptio Travel Salesforce org
- Salesforce API permissions

### Setup

1. **Clone or download the files:**
   ```bash
   # If using git
   git clone <repository>
   cd kaptio-travel-analytics
   
   # Or simply download the files to a directory
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Salesforce credentials:**
   
   Create a `.env` file in the project directory:
   ```bash
   # Salesforce Configuration
   SF_USERNAME=your.email@company.com
   SF_PASSWORD=your_password
   SF_SECURITY_TOKEN=your_security_token
   SF_DOMAIN=login  # Use 'test' for sandbox
   
   # Optional: Email for alerts
   ALERT_EMAIL=admin@company.com
   ```

4. **Get your Salesforce Security Token:**
   - Log into Salesforce
   - Go to Setup → My Personal Information → Reset Security Token
   - Check your email for the security token

## Usage

### Option 1: Interactive Dashboard (Recommended)

Run the Streamlit dashboard for real-time interactive analysis:

```bash
streamlit run kaptio_simple_analytics.py
```

This will open a web browser with an interactive dashboard showing:
- Key performance metrics
- Critical issues summary
- Performance trend charts
- User impact analysis
- Actionable recommendations

### Option 2: Command Line Analysis

Run a one-time analysis from the command line:

```bash
python kaptio_simple_analytics.py
```

This will:
- Connect to Salesforce
- Analyze recent performance data
- Print a summary to the console
- Save detailed results to JSON and CSV files

### Option 3: Scheduled Analysis

Set up automated analysis using cron (Linux/Mac) or Task Scheduler (Windows):

```bash
# Run analysis every hour
0 * * * * cd /path/to/kaptio-analytics && python kaptio_simple_analytics.py

# Run analysis every day at 8 AM
0 8 * * * cd /path/to/kaptio-analytics && python kaptio_simple_analytics.py
```

## Key Metrics Analyzed

### Performance Metrics
- **Success Rate**: Percentage of successful price calculations
- **Average Duration**: Mean time for price calculations
- **Timeout Rate**: Percentage of calculations that timeout
- **Long Running Calculations**: Number of calculations >1 hour

### User Experience Metrics
- **User Activity Patterns**: Most active users and their success rates
- **Problem Users**: Users experiencing high failure rates
- **Error Frequency**: Common error patterns and their impact

### Business Impact Metrics
- **Daily Trends**: Performance patterns over time
- **Operational Impact**: Effect on business deadlines
- **System Health**: Overall system reliability indicators

### 🆕 Size Variation & Complexity Metrics
- **Services Count**: Number of services/items per itinerary affecting calculation complexity
- **Trip Duration**: Number of days in the itinerary impacting pricing scope
- **Configurations**: Different pricing configuration options multiplying calculations
- **Departure Dates**: Number of departure date options requiring separate pricing
- **Group Size Variations**: Different group size options adding complexity
- **Complexity Scoring**: Weighted composite score of all complexity factors
- **Correlation Analysis**: Statistical relationships between complexity and performance
- **Performance by Complexity**: How calculation time varies with itinerary complexity

## Critical Thresholds

The system monitors these critical thresholds and generates alerts:

| Metric | Warning | Critical |
|--------|---------|----------|
| Timeout Rate | >5% | >10% |
| Average Duration | >30 min | >60 min |
| Success Rate | <90% | <80% |
| Long Running | >0 | >5 |

## Understanding the Results

### Dashboard Sections

1. **Key Metrics Cards**: High-level performance indicators
2. **Critical Issues Summary**: Immediate problems requiring attention
3. **Performance Analysis**: Detailed charts and trends
4. **User Impact Analysis**: User-specific performance data
5. **🆕 Size Variation Analysis**: Complexity factors affecting performance:
   - Average services, days, configurations, and departures per itinerary
   - Complexity score distribution and correlation with calculation time
   - Performance comparison across complexity quartiles
   - Detailed breakdown of most complex itineraries
6. **Immediate Action Items**: Prioritized recommendations (now includes complexity-based insights)

### Interpreting Alerts

- 🔥 **CRITICAL**: Immediate action required, system severely impacted
- ⚡ **HIGH**: Significant performance issue, user experience affected
- 🎯 **MEDIUM**: Notable problem, monitor closely
- 📊 **ONGOING**: Continuous monitoring recommendation

## Data Export

The system automatically exports data for further analysis:

### Files Generated
- `kaptio_performance_YYYYMMDD_HHMMSS.csv`: Raw performance data
- `kaptio_metrics_YYYYMMDD_HHMMSS.json`: Calculated metrics
- `analysis_results_YYYYMMDD_HHMMSS.json`: Complete analysis results

### Export Location
All exports are saved to the `./exports/` directory.

## Troubleshooting

### Common Issues

**"Failed to connect to Salesforce"**
- Check your credentials in the `.env` file
- Verify your security token is current
- Ensure your IP is trusted or use VPN
- Check if your user has API access permissions

**"No data found"**
- Verify the date range (default is last 30 days)
- Check if your user has access to KaptioTravel objects
- Confirm the org has CalloutRequest data

**"Permission denied"**
- Ensure your Salesforce user has read access to:
  - KaptioTravel__CalloutRequest__c
  - KaptioTravel__Itinerary__c
  - User objects

### Salesforce Permissions Required

Your Salesforce user needs:
- **API Enabled** permission
- **Read** access to Kaptio Travel objects
- **View All Data** (recommended) or specific object permissions

## Customization

### Modifying Analysis Parameters

Edit the values in `kaptio_simple_analytics.py`:

```python
# Change lookback period (days)
df = analytics.get_callout_performance(days_back=60)

# Modify critical thresholds
df['IsLongRunning'] = df['Duration_Minutes'] > 45  # 45 min instead of 60

# Add custom analysis
# ... your custom code here
```

### Adding New Metrics

1. Modify the SOQL query in `get_callout_performance()`
2. Add new calculations in `analyze_performance()`
3. Update the dashboard in `create_dashboard()`

### Extending to Other Objects

The framework can easily be extended to analyze:
- Itinerary performance
- User login patterns
- Background job failures
- API callout patterns

## Support and Maintenance

### Regular Maintenance
- Review and update critical thresholds quarterly
- Monitor data export sizes
- Update Salesforce credentials as needed
- Review and optimize SOQL queries for performance

### Scaling Considerations
- For large orgs (>10k records/month), consider:
  - Implementing data archiving
  - Using Salesforce Analytics API
  - Setting up dedicated reporting user
  - Implementing incremental data loading

## Business Impact

This analytics solution directly addresses the feedback issues:

### Immediate Benefits
- **Visibility**: Real-time insight into timeout and performance issues
- **Alerting**: Proactive notification of system degradation
- **User Support**: Identification of users needing additional help
- **Trend Analysis**: Understanding of peak usage and problem patterns

### Expected Outcomes
- 50% reduction in average calculation time within 3 months
- 80% reduction in timeout-related support tickets
- Improved user confidence through better error visibility
- 95% compliance with June/July deadline requirements

## Contributing

To contribute improvements:

1. Test changes thoroughly with your Salesforce org
2. Document any new features or metrics
3. Update this README with usage instructions
4. Consider backward compatibility

## Security Notes

- Never commit actual credentials to version control
- Use environment variables for all sensitive data
- Regularly rotate Salesforce security tokens
- Monitor API usage to avoid limits
- Consider using OAuth for production deployments

---

**Need Help?**
- Check the troubleshooting section above
- Review Salesforce API documentation
- Test with a smaller date range first
- Verify your Salesforce permissions 