# 🏠 Multi-Listing Calendar Dashboard

A Streamlit application for viewing and filtering booking blocks across multiple Airbnb listings by ingesting iCal (.ics) feeds.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **📅 Month Calendar View** — Full month grid showing which listings are blocked each day with color-coded tags
- **📋 List View** — Sortable, filterable table of all booking blocks with CSV export
- **📊 Summary View** — Occupancy statistics and bar charts by listing
- **🔄 Auto-Refresh** — Cached data with 10-minute TTL and manual refresh option
- **⚡ Resilient** — Graceful error handling if any feed fails to load

## Screenshots

| Month Calendar | List View | Summary |
|----------------|-----------|---------|
| Color-coded daily blocks | Exportable table | Occupancy charts |

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/airbnb-calendar-dashboard.git
   cd airbnb-calendar-dashboard
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Configuration

### Adding or Modifying Listings

Edit the `ICAL_FEEDS` dictionary in `app.py`:

```python
ICAL_FEEDS = {
    "Listing Name": {
        "url": "https://www.airbnb.com/calendar/ical/YOUR_ID.ics?t=YOUR_TOKEN",
        "id": "YOUR_ID",
        "color": "#FF5A5F"  # Hex color for calendar display
    },
    # Add more listings...
}
```

### Finding Your Airbnb iCal URL

1. Go to your [Airbnb Hosting Calendar](https://www.airbnb.com/hosting/calendar)
2. Select the listing
3. Click **Availability** → **Connect calendars** → **Connect to another website**
4. Copy the export URL provided

### Cache Settings

Adjust the cache TTL (time-to-live) by modifying:

```python
CACHE_TTL = 600  # seconds (default: 10 minutes)
```

## Usage

### Sidebar Controls

| Control | Description |
|---------|-------------|
| **Refresh Feeds** | Clears cache and reloads all iCal data |
| **Select Listings** | Multi-select filter for which listings to display |
| **Date Range Mode** | Choose "1 week", "1 month", or "custom" range |
| **Feed Status** | Shows load status for each listing feed |

### Views

#### Month Calendar
- Navigate between months using Previous/Next buttons
- "Today" button returns to current month
- Each day cell shows color-coded chips for blocked listings
- Legend at top shows listing colors

#### List View
- Displays all booking blocks within selected date range
- Columns: Listing, Start Date, End Date, Nights, Summary
- Click column headers to sort
- **Export to CSV** button downloads filtered data

#### Summary View
- Total blocked nights by listing
- Average occupancy rate calculation
- Bar chart visualization
- Detailed breakdown table

## Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation and analysis |
| `requests` | HTTP requests for fetching iCal feeds |
| `icalendar` | Robust iCal (.ics) parsing |

### Why icalendar?

The `icalendar` package was chosen for .ics parsing because:
- Most mature and widely-used Python library for iCal handling
- Proper support for all-day events (common in Airbnb)
- Handles timezone conversion correctly
- Gracefully handles malformed events

### iCal Date Handling

- **DTEND is exclusive**: A booking from Jan 1 to Jan 3 blocks the nights of Jan 1 and Jan 2
- **All-day events**: Converted to date-level blocks (not datetime)
- **Deduplication**: Overlapping events from the same listing are merged

### Data Schema

The normalized DataFrame contains:

| Column | Type | Description |
|--------|------|-------------|
| `listing_name` | str | Display name of the listing |
| `listing_id` | str | Airbnb listing ID |
| `event_uid` | str | Unique event identifier |
| `start_date` | date | First blocked night |
| `end_date` | date | Day after last blocked night (exclusive) |
| `nights_blocked` | int | Number of nights blocked |
| `summary` | str | Event description (e.g., "Reserved", "Airbnb (Not available)") |
| `source_url` | str | Original iCal feed URL |

## Syncing with Other Platforms

This dashboard reads Airbnb calendars. For syncing calendars across platforms:

- **Export from Airbnb**: Use the iCal URL in this app's configuration
- **Import to Airbnb**: Add external calendar URLs ending in `.ics` via Airbnb's calendar settings
- **Supported platforms**: VRBO, Booking.com, Tripadvisor, Google Calendar, Apple Calendar

See [Airbnb's calendar sync documentation](https://www.airbnb.com/help/article/99) for details.

## Troubleshooting

### Feed fails to load
- Check that the iCal URL is correct and accessible
- Airbnb tokens can expire; regenerate the URL from your hosting dashboard
- The app continues to function with partial data if some feeds fail

### Calendar not updating
- Airbnb calendars sync externally every ~3 hours
- Click "Refresh Feeds" to clear the local cache
- Check the Feed Status section in the sidebar

### Performance issues
- Reduce the number of listings if loading is slow
- Increase `CACHE_TTL` for less frequent network requests

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- iCal parsing by [icalendar](https://icalendar.readthedocs.io/)
- Airbnb calendar documentation referenced for iCal format handling
