# Daily Tracker Dashboard 📊

A Streamlit dashboard for tracking and visualizing hourly ride data with real-time analytics and comparisons.

## Features

- 📈 Real-time ride tracking and analytics
- 📊 Interactive hourly distribution charts
- 📋 Detailed breakdown by area and hour
- 🔄 Day-over-day and week-over-week comparisons
- 💾 CSV export functionality
- 🎨 Beautiful conditional formatting with blue gradients
- ⚡ Auto-refresh capability

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd daily-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run tracker_timestamp_moved.py
```

2. Enter your Rabbit API Token in the sidebar

3. Select a date to view analytics

4. Use the refresh button to update data

## Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- Plotly 5.17.0+
- Requests 2.31.0+
- OpenPyXL 3.1.0+

## Configuration

The app requires:
- Valid Rabbit API authentication token
- Access to `https://dashboard.rabbit-api.app/export` endpoint

## Features Breakdown

### Key Metrics
- Total rides today with comparison to yesterday
- Daily growth percentage
- Week-over-week comparison
- Peak hour identification

### Detailed Hourly Breakdown
- Rides by area for each hour (0-23)
- Total rides per hour across all areas
- Comparison with previous day
- Comparison with same day last week
- Blue gradient visualization (darker = more rides)
- CSV export option

### Hourly Ride Distribution Chart
- Line chart showing ride distribution across 24 hours
- Comparison lines for yesterday and last week
- Interactive hover information
- Average rides per hour
- Busiest hours highlighted

## Data Refresh

- Data is cached for 5 minutes (300 seconds)
- Use refresh buttons to force data reload
- Timestamp shows last update time

## License

MIT License

## Support

For issues or questions, please open an issue on GitHub.
