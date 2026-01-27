"""
Multi-Listing Calendar Dashboard
================================
A Streamlit app for viewing and filtering booking blocks across multiple Airbnb listings
by ingesting iCal (.ics) feeds.

Uses the 'icalendar' package for robust .ics parsing - it's the most mature and widely-used
Python library for iCal handling, with proper support for all-day events, timezones, and
edge cases common in Airbnb calendar exports.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
from icalendar import Calendar
from datetime import date, datetime, timedelta
import calendar
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

ICAL_FEEDS = {
    "Lanesville": {
        "url": "https://www.airbnb.com/calendar/ical/52260561.ics?t=5b901a38421d4320ba225a8d1f1c2c80",
        "id": "52260561",
        "color": "#FF5A5F",  # Airbnb red
    },
    "Milla": {
        "url": "https://www.airbnb.com/calendar/ical/907278153520205895.ics?t=577b74f67d454da8a5cbe85de23549a7",
        "id": "907278153520205895",
        "color": "#00A699",  # Teal
    },
    "West Kill": {
        "url": "https://www.airbnb.com/calendar/ical/818696761794452121.ics?t=0b0f20612dc74ab8b19c85b454a191b7",
        "id": "818696761794452121",
        "color": "#FC642D",  # Orange
    },
    "Miller Road": {
        "url": "https://www.airbnb.com/calendar/ical/2105131.ics?t=e157f31bee7341a59209c4971602f816",
        "id": "2105131",
        "color": "#484848",  # Dark gray
    }
}

CACHE_TTL = 600  # 10 minutes

# =============================================================================
# DATA FETCHING & PARSING
# =============================================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ical(url: str, timeout: int = 30) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch iCal data from URL with timeout and error handling.
    
    Returns:
        tuple: (ical_text, error_message) - one will be None
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text, None
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)}"


def parse_date_value(dt_value) -> date:
    """
    Convert an iCal date/datetime value to a Python date.
    Handles both DATE and DATE-TIME types correctly.
    """
    if hasattr(dt_value, 'dt'):
        dt_value = dt_value.dt
    
    if isinstance(dt_value, datetime):
        return dt_value.date()
    elif isinstance(dt_value, date):
        return dt_value
    else:
        return datetime.fromisoformat(str(dt_value)).date()


def parse_ical(ical_text: str, listing_name: str, listing_id: str, source_url: str) -> pd.DataFrame:
    """
    Parse iCal text into a normalized DataFrame of booking blocks.
    """
    events = []
    seen_uids = set()
    
    try:
        cal = Calendar.from_ical(ical_text)
    except Exception as e:
        st.warning(f"Failed to parse iCal for {listing_name}: {e}")
        return pd.DataFrame()
    
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        
        try:
            uid = str(component.get('uid', ''))
            dtstart = component.get('dtstart')
            dtend = component.get('dtend')
            
            if not dtstart:
                continue
            
            start_date = parse_date_value(dtstart)
            
            if dtend:
                end_date = parse_date_value(dtend)
            else:
                end_date = start_date + timedelta(days=1)
            
            nights_blocked = (end_date - start_date).days
            
            if nights_blocked <= 0:
                continue
            
            summary = str(component.get('summary', 'Blocked'))
            
            dedup_key = f"{start_date}_{end_date}"
            if dedup_key in seen_uids:
                continue
            seen_uids.add(dedup_key)
            
            events.append({
                'listing_name': listing_name,
                'listing_id': listing_id,
                'event_uid': uid,
                'start_date': start_date,
                'end_date': end_date,
                'nights_blocked': nights_blocked,
                'summary': summary,
                'source_url': source_url
            })
            
        except Exception:
            continue
    
    if not events:
        return pd.DataFrame(columns=[
            'listing_name', 'listing_id', 'event_uid', 'start_date',
            'end_date', 'nights_blocked', 'summary', 'source_url'
        ])
    
    df = pd.DataFrame(events)
    df['start_date'] = pd.to_datetime(df['start_date']).dt.date
    df['end_date'] = pd.to_datetime(df['end_date']).dt.date
    
    return df


def load_all_feeds() -> tuple[pd.DataFrame, dict]:
    """
    Load and parse all iCal feeds, handling errors gracefully.
    """
    all_events = []
    status = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, meta) in enumerate(ICAL_FEEDS.items()):
        status_text.text(f"Loading {name}...")
        
        ical_text, error = fetch_ical(meta['url'])
        
        if error:
            status[name] = (False, error)
        else:
            df = parse_ical(ical_text, name, meta['id'], meta['url'])
            if df.empty:
                status[name] = (True, f"Loaded (0 events)")
            else:
                all_events.append(df)
                status[name] = (True, f"Loaded ({len(df)} events)")
        
        progress_bar.progress((i + 1) / len(ICAL_FEEDS))
    
    progress_bar.empty()
    status_text.empty()
    
    if all_events:
        combined_df = pd.concat(all_events, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=[
            'listing_name', 'listing_id', 'event_uid', 'start_date',
            'end_date', 'nights_blocked', 'summary', 'source_url'
        ])
    
    return combined_df, status


# =============================================================================
# CALENDAR GRID UTILITIES
# =============================================================================

def build_month_grid(year: int, month: int) -> list[list[date]]:
    """
    Build a month grid structure for calendar rendering.
    Week starts on Sunday.
    """
    cal = calendar.Calendar(firstweekday=6)  # Sunday start
    return cal.monthdatescalendar(year, month)


def get_bookings_for_date(target_date: date, events_df: pd.DataFrame) -> list[dict]:
    """
    Get list of bookings that include a specific date.
    """
    if events_df.empty:
        return []
    
    mask = (events_df['start_date'] <= target_date) & (target_date < events_df['end_date'])
    return events_df[mask].to_dict('records')


def filter_events_by_date_range(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """
    Filter events that overlap with the given date range.
    """
    if df.empty:
        return df
    
    mask = (df['start_date'] < end) & (df['end_date'] > start)
    return df[mask].copy()


# =============================================================================
# CALENDAR HTML GENERATION
# =============================================================================

def generate_calendar_html(year: int, month: int, events_df: pd.DataFrame, selected_listings: list[str]) -> str:
    """Generate complete HTML for the calendar with embedded CSS."""
    
    # Filter events to selected listings
    if selected_listings:
        filtered_df = events_df[events_df['listing_name'].isin(selected_listings)]
    else:
        filtered_df = events_df
    
    grid = build_month_grid(year, month)
    today = date.today()
    
    # Build complete HTML document
    html = f'''<!DOCTYPE html>
<html>
<head>
<style>
* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: transparent;
    padding: 10px;
}}

.calendar-container {{
    max-width: 100%;
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

.legend-container {{
    display: flex;
    gap: 24px;
    padding: 16px 20px;
    background: #F7F7F7;
    flex-wrap: wrap;
    border-bottom: 1px solid #EBEBEB;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.legend-color {{
    width: 14px;
    height: 14px;
    border-radius: 50%;
    flex-shrink: 0;
}}

.legend-text {{
    font-size: 14px;
    font-weight: 500;
    color: #222222;
}}

.calendar-header {{
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    background: #FAFAFA;
    border-bottom: 1px solid #EBEBEB;
}}

.calendar-header-cell {{
    text-align: center;
    padding: 14px 8px;
    font-size: 12px;
    font-weight: 600;
    color: #717171;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.calendar-grid {{
    display: grid;
    grid-template-columns: repeat(7, 1fr);
}}

.calendar-cell {{
    min-height: 120px;
    border-right: 1px solid #EBEBEB;
    border-bottom: 1px solid #EBEBEB;
    padding: 10px;
    background: #FFFFFF;
    vertical-align: top;
    position: relative;
}}

.calendar-cell:nth-child(7n) {{
    border-right: none;
}}

.calendar-cell.outside-month {{
    background: #FAFAFA;
}}

.calendar-cell.today {{
    background: #FEF7F6;
}}

.day-number {{
    font-size: 14px;
    font-weight: 500;
    color: #222222;
    margin-bottom: 8px;
    height: 28px;
    display: flex;
    align-items: center;
}}

.day-number.outside {{
    color: #B0B0B0;
}}

.day-number.today-marker {{
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 2px solid #FF5A5F;
    color: #FF5A5F;
    font-weight: 600;
}}

.bookings-container {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.booking-bar {{
    display: flex;
    align-items: center;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 500;
    color: white;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 26px;
    border-radius: 4px;
    cursor: default;
}}

.booking-bar.is-start {{
    border-top-left-radius: 13px;
    border-bottom-left-radius: 13px;
    margin-right: -10px;
    padding-right: 14px;
}}

.booking-bar.is-end {{
    border-top-right-radius: 13px;
    border-bottom-right-radius: 13px;
    margin-left: -10px;
    padding-left: 14px;
}}

.booking-bar.is-middle {{
    border-radius: 0;
    margin-left: -10px;
    margin-right: -10px;
    padding-left: 14px;
    padding-right: 14px;
}}

.booking-bar.is-single {{
    border-radius: 13px;
    margin-left: 0;
    margin-right: 0;
}}

.booking-avatar {{
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: rgba(255,255,255,0.25);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 6px;
    font-size: 10px;
    font-weight: 600;
    flex-shrink: 0;
}}

.booking-text {{
    overflow: hidden;
    text-overflow: ellipsis;
}}
</style>
</head>
<body>
<div class="calendar-container">
    <div class="legend-container">
'''
    
    # Add legend items
    for name in selected_listings:
        meta = ICAL_FEEDS.get(name, {})
        color = meta.get('color', '#888888')
        html += f'''        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <span class="legend-text">{name}</span>
        </div>
'''
    
    html += '''    </div>
    <div class="calendar-header">
'''
    
    # Add day headers
    for day in ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']:
        html += f'        <div class="calendar-header-cell">{day}</div>\n'
    
    html += '''    </div>
    <div class="calendar-grid">
'''
    
    # Add calendar cells
    for week in grid:
        for day_idx, day in enumerate(week):
            is_current_month = day.month == month
            is_today = day == today
            
            cell_classes = ['calendar-cell']
            if not is_current_month:
                cell_classes.append('outside-month')
            if is_today:
                cell_classes.append('today')
            
            html += f'        <div class="{" ".join(cell_classes)}">\n'
            
            # Day number
            day_classes = ['day-number']
            if not is_current_month:
                day_classes.append('outside')
            if is_today:
                day_classes.append('today-marker')
            
            html += f'            <div class="{" ".join(day_classes)}">{day.day}</div>\n'
            
            # Get bookings for this day
            bookings = get_bookings_for_date(day, filtered_df)
            
            if bookings:
                html += '            <div class="bookings-container">\n'
                
                # Sort bookings by listing name for consistent ordering
                bookings_sorted = sorted(bookings, key=lambda x: x['listing_name'])
                
                for booking in bookings_sorted:
                    listing_name = booking['listing_name']
                    meta = ICAL_FEEDS.get(listing_name, {})
                    color = meta.get('color', '#888888')
                    
                    start = booking['start_date']
                    end = booking['end_date']
                    last_night = end - timedelta(days=1)
                    nights = booking['nights_blocked']
                    summary = booking.get('summary', 'Reserved')
                    
                    # Determine bar position
                    is_start = day == start
                    is_end = day == last_night
                    
                    bar_classes = ['booking-bar']
                    if is_start and is_end:
                        bar_classes.append('is-single')
                    elif is_start:
                        bar_classes.append('is-start')
                    elif is_end:
                        bar_classes.append('is-end')
                    else:
                        bar_classes.append('is-middle')
                    
                    # Show label on start day or first day of week for continuing bookings
                    is_week_start = day_idx == 0
                    show_label = is_start or (is_week_start and not is_start)
                    
                    tooltip = f"{listing_name}: {summary} ({nights} night{'s' if nights != 1 else ''})"
                    
                    if show_label:
                        label_html = f'<span class="booking-avatar">{listing_name[0]}</span><span class="booking-text">{listing_name}</span>'
                    else:
                        label_html = ''
                    
                    html += f'                <div class="{" ".join(bar_classes)}" style="background-color: {color};" title="{tooltip}">{label_html}</div>\n'
                
                html += '            </div>\n'
            
            html += '        </div>\n'
    
    html += '''    </div>
</div>
</body>
</html>'''
    
    return html


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_list_view(events_df: pd.DataFrame, start_date: date, end_date: date, selected_listings: list[str]):
    """Render filtered table view with export option."""
    
    filtered_df = filter_events_by_date_range(events_df, start_date, end_date)
    
    if selected_listings:
        filtered_df = filtered_df[filtered_df['listing_name'].isin(selected_listings)]
    
    if filtered_df.empty:
        st.info("No bookings found for the selected filters.")
        return
    
    filtered_df = filtered_df.sort_values('start_date')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Blocks", len(filtered_df))
    with col2:
        st.metric("Total Nights", filtered_df['nights_blocked'].sum())
    with col3:
        st.metric("Listings", filtered_df['listing_name'].nunique())
    
    display_df = filtered_df[[
        'listing_name', 'start_date', 'end_date', 'nights_blocked', 'summary'
    ]].copy()
    display_df.columns = ['Listing', 'Start Date', 'End Date', 'Nights', 'Summary']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Start Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
            "End Date": st.column_config.DateColumn(format="MMM DD, YYYY"),
        }
    )
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Export to CSV",
        data=csv,
        file_name=f"airbnb_blocks_{start_date}_{end_date}.csv",
        mime="text/csv"
    )


def render_summary_view(events_df: pd.DataFrame, start_date: date, end_date: date, selected_listings: list[str]):
    """Render summary statistics and bar chart."""
    
    filtered_df = filter_events_by_date_range(events_df, start_date, end_date)
    
    if selected_listings:
        filtered_df = filtered_df[filtered_df['listing_name'].isin(selected_listings)]
    
    st.markdown(f"### Summary: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}")
    
    if filtered_df.empty:
        st.info("No bookings found for the selected filters.")
        return
    
    summary_data = []
    
    for listing in filtered_df['listing_name'].unique():
        listing_events = filtered_df[filtered_df['listing_name'] == listing]
        
        blocked_dates = set()
        for _, event in listing_events.iterrows():
            event_start = max(event['start_date'], start_date)
            event_end = min(event['end_date'], end_date)
            
            current = event_start
            while current < event_end:
                blocked_dates.add(current)
                current += timedelta(days=1)
        
        summary_data.append({
            'Listing': listing,
            'Blocked Nights': len(blocked_dates),
            'Number of Blocks': len(listing_events)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    total_nights_in_range = (end_date - start_date).days
    total_blocked = summary_df['Blocked Nights'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Date Range (nights)", total_nights_in_range)
    with col2:
        st.metric("Total Blocked Nights", total_blocked)
    with col3:
        avg_occupancy = (total_blocked / (total_nights_in_range * len(summary_df)) * 100) if summary_df.shape[0] > 0 else 0
        st.metric("Avg Occupancy Rate", f"{avg_occupancy:.1f}%")
    
    st.markdown("#### Blocked Nights by Listing")
    chart_df = summary_df.set_index('Listing')['Blocked Nights']
    st.bar_chart(chart_df)
    
    st.markdown("#### Breakdown")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Multi-Listing Calendar Dashboard",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 Multi-Listing Calendar Dashboard")
    st.markdown("View and filter booking blocks across multiple Airbnb listings")
    
    # Initialize session state
    if 'events_df' not in st.session_state:
        st.session_state.events_df = None
    if 'feed_status' not in st.session_state:
        st.session_state.feed_status = None
    if 'calendar_month' not in st.session_state:
        st.session_state.calendar_month = date.today().month
    if 'calendar_year' not in st.session_state:
        st.session_state.calendar_year = date.today().year
    
    today = date.today()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Refresh Feeds", use_container_width=True):
            st.cache_data.clear()
            st.session_state.events_df = None
            st.session_state.feed_status = None
            st.rerun()
        
        st.divider()
        
        st.subheader("📋 Listings")
        all_listings = list(ICAL_FEEDS.keys())
        selected_listings = st.multiselect(
            "Select Listings",
            options=all_listings,
            default=all_listings,
            help="Filter to show only selected listings"
        )
        
        st.divider()
        
        st.subheader("📅 Date Range")
        date_mode = st.radio(
            "Range Mode",
            options=["1 week", "1 month", "custom"],
            horizontal=True
        )
        
        if date_mode == "1 week":
            start_date = today
            end_date = today + timedelta(days=7)
            st.info(f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}")
        elif date_mode == "1 month":
            start_date = today
            end_date = today + timedelta(days=30)
            st.info(f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=today)
            with col2:
                end_date = st.date_input("End Date", value=today + timedelta(days=30))
            
            if start_date >= end_date:
                st.error("End date must be after start date")
                return
        
        st.divider()
        
        st.subheader("📡 Feed Status")
        if st.session_state.feed_status:
            for name, (success, msg) in st.session_state.feed_status.items():
                if success:
                    st.success(f"✅ {name}: {msg}")
                else:
                    st.error(f"❌ {name}: {msg}")
    
    # Load data if not loaded
    if st.session_state.events_df is None:
        with st.spinner("Loading calendar feeds..."):
            events_df, status = load_all_feeds()
            st.session_state.events_df = events_df
            st.session_state.feed_status = status
            st.rerun()
    
    events_df = st.session_state.events_df
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📅 Month Calendar", "📋 List View", "📊 Summary"])
    
    with tab1:
        # Month navigation
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("◀ Prev", use_container_width=True):
                if st.session_state.calendar_month == 1:
                    st.session_state.calendar_month = 12
                    st.session_state.calendar_year -= 1
                else:
                    st.session_state.calendar_month -= 1
                st.rerun()
        
        with col2:
            if st.button("Today", use_container_width=True):
                st.session_state.calendar_month = today.month
                st.session_state.calendar_year = today.year
                st.rerun()
        
        with col3:
            month_name = calendar.month_name[st.session_state.calendar_month]
            st.markdown(f"<h2 style='text-align: center; margin: 0;'>{month_name} {st.session_state.calendar_year}</h2>", unsafe_allow_html=True)
        
        with col5:
            if st.button("Next ▶", use_container_width=True):
                if st.session_state.calendar_month == 12:
                    st.session_state.calendar_month = 1
                    st.session_state.calendar_year += 1
                else:
                    st.session_state.calendar_month += 1
                st.rerun()
        
        # Render calendar using components.html for proper HTML rendering
        calendar_html = generate_calendar_html(
            st.session_state.calendar_year, 
            st.session_state.calendar_month, 
            events_df, 
            selected_listings
        )
        
        # Calculate dynamic height based on number of weeks
        grid = build_month_grid(st.session_state.calendar_year, st.session_state.calendar_month)
        num_weeks = len(grid)
        calendar_height = 80 + 50 + (num_weeks * 130)  # legend + header + cells
        
        components.html(calendar_html, height=calendar_height, scrolling=False)
    
    with tab2:
        render_list_view(events_df, start_date, end_date, selected_listings)
    
    with tab3:
        render_summary_view(events_df, start_date, end_date, selected_listings)
    
    # Footer
    st.divider()
    st.caption(
        "Data refreshes automatically every 10 minutes. "
        "Click 'Refresh Feeds' to manually update. "
        "Note: Airbnb calendars typically sync every 3 hours."
    )


if __name__ == "__main__":
    main()
