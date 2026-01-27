"""
Airbnb Multi-Listing Calendar Manager with CSV Enrichment
Streamlit app that ingests Airbnb iCal feeds and enriches with CSV reservation data.
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from icalendar import Calendar
import re
import json
import os
import tempfile
import shutil
from typing import Optional, Dict, List, Tuple, Any
import calendar as cal_module
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

ICAL_FEEDS = {
    "lanesville": "https://www.airbnb.com/calendar/ical/52260561.ics?t=5b901a38421d4320ba225a8d1f1c2c80",
    "milla": "https://www.airbnb.com/calendar/ical/907278153520205895.ics?t=577b74f67d454da8a5cbe85de23549a7",
    "westkill": "https://www.airbnb.com/calendar/ical/818696761794452121.ics?t=0b0f20612dc74ab8b19c85b454a191b7",
    "millerroad": "https://www.airbnb.com/calendar/ical/2105131.ics?t=e157f31bee7341a59209c4971602f816",
}

LISTING_DISPLAY_NAMES = {
    "lanesville": "Lanesville (Kaaterskill Cabin)",
    "milla": "Milla (Tiny Cabin Under Stars)",
    "westkill": "West Kill (Brewery/Falls)",
    "millerroad": "Miller Road (Phoenicia/Hunter)",
}

# Weighted keyword classifier scoring
CLASSIFIER_WEIGHTS = {
    "lanesville": {
        "kaaterskill": 3,
        "firepit": 2,
        "sauna": 2,
        "hottub": 1,
        "cabin": 1,
    },
    "milla": {
        "tiny": 3,
        "under": 2,
        "stars": 2,
        "sauna": 2,
        "hottub": 1,
        "cabin": 1,
    },
    "westkill": {
        "west": 3,
        "kill": 3,
        "brewery": 3,
        "kaaterskill": 2,
        "falls": 2,
        "modern": 1,
        "cabin": 1,
    },
    "millerroad": {
        "phoenicia": 3,
        "hunter": 3,
        "slopes": 2,
        "modern": 2,
        "cabin": 1,
    },
}

# Compound token mapping
COMPOUND_TOKENS = {
    ("hot", "tub"): "hottub",
    ("fire", "pit"): "firepit",
}

DEFAULT_ALIAS_STORE_PATH = "listing_aliases.json"

# ============================================================================
# CANONICALIZATION & CLASSIFICATION
# ============================================================================

def canonicalize_listing(text: str) -> str:
    """Canonicalize listing text for matching."""
    if not text or pd.isna(text):
        return ""
    text = str(text).casefold().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_with_compounds(text: str) -> List[str]:
    """Tokenize text and detect compound words."""
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = (tokens[i], tokens[i + 1])
            if pair in COMPOUND_TOKENS:
                result.append(COMPOUND_TOKENS[pair])
                i += 2
                continue
        result.append(tokens[i])
        i += 1
    return result


def classify_listing(listing_norm: str) -> Dict[str, Any]:
    """Classify a normalized listing string into a listing key."""
    if not listing_norm:
        return {
            "listing_key": "unknown",
            "confidence": 0.0,
            "max_score": 0,
            "second_score": 0,
            "matched_terms": [],
            "scores": {},
        }
    
    tokens = tokenize_with_compounds(listing_norm)
    token_set = set(tokens)
    
    scores = {}
    term_matches = {}
    
    for key, weights in CLASSIFIER_WEIGHTS.items():
        score = 0
        matches = []
        for term, weight in weights.items():
            if term in token_set:
                score += weight
                matches.append(term)
        scores[key] = score
        term_matches[key] = matches
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_key, max_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    
    # Decision rule: max_score >= 4 AND (max_score - second_score) >= 2
    if max_score < 4 or (max_score - second_score) < 2:
        listing_key = "unknown"
        confidence = 0.0
    else:
        listing_key = max_key
        confidence = min(1.0, max_score / 10.0) * min(1.0, (max_score - second_score) / 4.0)
    
    return {
        "listing_key": listing_key,
        "confidence": confidence,
        "max_score": max_score,
        "second_score": second_score,
        "matched_terms": term_matches.get(max_key, []),
        "scores": scores,
    }


def is_strong_classification(result: Dict[str, Any]) -> bool:
    """Check if classification is strong enough for auto-learning."""
    return result["max_score"] >= 6 and (result["max_score"] - result["second_score"]) >= 3

# ============================================================================
# ALIAS STORE
# ============================================================================

def get_default_alias_store() -> Dict:
    """Return empty alias store structure."""
    return {
        "version": 1,
        "updated_at": datetime.now().isoformat(),
        "aliases": {},
    }


def load_alias_store(path: str) -> Dict:
    """Load alias store from JSON file."""
    if not os.path.exists(path):
        return get_default_alias_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "aliases" not in data:
            st.warning(f"Alias store at {path} has invalid format. Using empty store.")
            return get_default_alias_store()
        return data
    except (json.JSONDecodeError, IOError) as e:
        st.warning(f"Could not load alias store: {e}. Using empty store.")
        return get_default_alias_store()


def save_alias_store(store: Dict, path: str) -> bool:
    """Save alias store atomically."""
    try:
        store["updated_at"] = datetime.now().isoformat()
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        shutil.move(tmp_path, path)
        return True
    except (IOError, OSError) as e:
        st.error(f"Could not save alias store: {e}")
        return False


def add_alias(store: Dict, listing_norm: str, listing_key: str, source: str, 
              confidence: float, raw_example: str) -> Dict:
    """Add or update an alias in the store."""
    now = datetime.now().isoformat()
    if listing_norm in store["aliases"]:
        entry = store["aliases"][listing_norm]
        entry["last_seen"] = now
        entry["listing_key"] = listing_key
        entry["source"] = source
        entry["confidence"] = confidence
        if raw_example and raw_example not in entry.get("raw_examples", []):
            entry.setdefault("raw_examples", []).append(raw_example)
            entry["raw_examples"] = entry["raw_examples"][-5:]
    else:
        store["aliases"][listing_norm] = {
            "listing_key": listing_key,
            "source": source,
            "confidence": confidence,
            "first_seen": now,
            "last_seen": now,
            "raw_examples": [raw_example] if raw_example else [],
        }
    return store

# ============================================================================
# ICAL FETCHING & PARSING
# ============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_ical(url: str, listing_key: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch iCal data from URL with caching."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text, None
    except requests.RequestException as e:
        return None, f"Failed to fetch {listing_key}: {str(e)}"


def parse_ical_to_df(ical_text: str, listing_key: str, listing_name: str, 
                     source_url: str) -> pd.DataFrame:
    """Parse iCal text into a DataFrame of blocked date ranges."""
    events = []
    try:
        cal = Calendar.from_ical(ical_text)
        for component in cal.walk():
            if component.name == "VEVENT":
                dtstart = component.get("dtstart")
                dtend = component.get("dtend")
                summary = str(component.get("summary", "Blocked"))
                uid = str(component.get("uid", ""))
                
                if dtstart:
                    start_date = dtstart.dt
                    if isinstance(start_date, datetime):
                        start_date = start_date.date()
                    
                    if dtend:
                        end_date = dtend.dt
                        if isinstance(end_date, datetime):
                            end_date = end_date.date()
                    else:
                        end_date = start_date + timedelta(days=1)
                    
                    # DTEND is exclusive in iCal
                    nights_blocked = (end_date - start_date).days
                    
                    events.append({
                        "listing_key": listing_key,
                        "listing_name": listing_name,
                        "event_uid": uid,
                        "start_date": start_date,
                        "end_date": end_date,
                        "nights_blocked": nights_blocked,
                        "summary": summary,
                        "source_url": source_url,
                    })
    except Exception as e:
        st.warning(f"Error parsing iCal for {listing_key}: {e}")
    
    return pd.DataFrame(events)


def fetch_all_ical_feeds() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch and parse all iCal feeds, returning combined DataFrame and errors."""
    all_events = []
    errors = {}
    
    for listing_key, url in ICAL_FEEDS.items():
        ical_text, error = fetch_ical(url, listing_key)
        if error:
            errors[listing_key] = error
        elif ical_text:
            listing_name = LISTING_DISPLAY_NAMES.get(listing_key, listing_key)
            df = parse_ical_to_df(ical_text, listing_key, listing_name, url)
            if not df.empty:
                all_events.append(df)
    
    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=[
            "listing_key", "listing_name", "event_uid", "start_date", 
            "end_date", "nights_blocked", "summary", "source_url"
        ])
    
    return combined, errors

# ============================================================================
# CSV LOADING & NORMALIZATION
# ============================================================================

def parse_currency(value: Any) -> float:
    """Parse currency string to float."""
    if pd.isna(value):
        return 0.0
    s = str(value).strip()
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def parse_date_flexible(value: Any) -> Optional[date]:
    """Parse date from various formats."""
    if pd.isna(value):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    
    s = str(value).strip()
    formats = [
        "%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y",
        "%m-%d-%Y", "%m-%d-%y", "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def get_file_hash(content: bytes) -> str:
    """Generate hash of file content for caching."""
    return hashlib.md5(content).hexdigest()


@st.cache_data(show_spinner=False)
def load_reservations_csv(content: bytes, _hash: str, alias_store_path: str) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Load and normalize reservations CSV."""
    warnings = []
    alias_store = load_alias_store(alias_store_path)
    
    try:
        from io import BytesIO
        df = pd.read_csv(BytesIO(content))
    except Exception as e:
        return pd.DataFrame(), [f"Failed to parse CSV: {e}"], alias_store
    
    # Column mapping
    column_map = {
        "Confirmation code": "reservation_id",
        "Status": "status",
        "Guest name": "guest_name",
        "Contact": "contact",
        "# of adults": "adults",
        "# of children": "children",
        "# of infants": "infants",
        "Start date": "checkin_date",
        "End date": "checkout_date",
        "# of nights": "nights",
        "Booked": "booked_on",
        "Listing": "listing_raw",
        "Earnings": "earnings",
    }
    
    # Rename columns
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    # Parse numeric columns
    for col in ["adults", "children", "infants", "nights"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    
    # Parse dates
    if "checkin_date" in df.columns:
        df["checkin_date"] = df["checkin_date"].apply(parse_date_flexible)
    if "checkout_date" in df.columns:
        df["checkout_date"] = df["checkout_date"].apply(parse_date_flexible)
    if "booked_on" in df.columns:
        df["booked_on"] = df["booked_on"].apply(parse_date_flexible)
    
    # Parse earnings
    if "earnings" in df.columns:
        df["earnings"] = df["earnings"].apply(parse_currency)
    
    # Classify listings
    if "listing_raw" in df.columns:
        df["listing_norm"] = df["listing_raw"].apply(canonicalize_listing)
        
        listing_keys = []
        listing_sources = []
        new_aliases = []
        
        for idx, row in df.iterrows():
            listing_norm = row["listing_norm"]
            listing_raw = row.get("listing_raw", "")
            
            # Check alias store first
            if listing_norm in alias_store["aliases"]:
                key = alias_store["aliases"][listing_norm]["listing_key"]
                source = "alias"
            else:
                # Use classifier
                result = classify_listing(listing_norm)
                key = result["listing_key"]
                source = "classifier"
                
                # Auto-learn strong classifications
                if key != "unknown" and is_strong_classification(result):
                    new_aliases.append((listing_norm, key, result["confidence"], listing_raw))
            
            listing_keys.append(key)
            listing_sources.append(source)
        
        df["listing_key"] = listing_keys
        df["listing_source"] = listing_sources
        
        # Add new auto-learned aliases
        for norm, key, conf, raw in new_aliases:
            alias_store = add_alias(alias_store, norm, key, "auto", conf, raw)
    
    # Compute nights mismatch
    if "checkin_date" in df.columns and "checkout_date" in df.columns and "nights" in df.columns:
        def compute_nights(row):
            if row["checkin_date"] and row["checkout_date"]:
                return (row["checkout_date"] - row["checkin_date"]).days
            return None
        df["computed_nights"] = df.apply(compute_nights, axis=1)
        df["nights_mismatch"] = df.apply(
            lambda r: r["computed_nights"] != r["nights"] if r["computed_nights"] is not None else False,
            axis=1
        )
    
    # Drop rows with invalid dates
    valid_mask = df["checkin_date"].notna() & df["checkout_date"].notna()
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        warnings.append(f"Dropped {invalid_count} rows with invalid dates")
    df = df[valid_mask].copy()
    
    return df, warnings, alias_store

# ============================================================================
# ENRICHMENT & MATCHING
# ============================================================================

def enrich_ical_events(ical_df: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich iCal events with reservation data."""
    if ical_df.empty:
        return ical_df
    
    enriched = ical_df.copy()
    
    # Initialize enrichment columns
    enrichment_cols = [
        "reservation_id", "status", "guest_name", "contact", 
        "adults", "children", "infants", "earnings", "booked_on",
        "match_confidence", "match_type"
    ]
    for col in enrichment_cols:
        enriched[col] = None
    enriched["match_confidence"] = "none"
    enriched["match_type"] = ""
    
    if res_df.empty:
        return enriched
    
    for idx, event in enriched.iterrows():
        listing_key = event["listing_key"]
        event_start = event["start_date"]
        event_end = event["end_date"]
        
        # Filter reservations for this listing
        listing_res = res_df[res_df["listing_key"] == listing_key]
        if listing_res.empty:
            continue
        
        # Try exact date match first
        exact_match = listing_res[
            (listing_res["checkin_date"] == event_start) & 
            (listing_res["checkout_date"] == event_end)
        ]
        
        if not exact_match.empty:
            match = exact_match.iloc[0]
            for col in ["reservation_id", "status", "guest_name", "contact", 
                       "adults", "children", "infants", "earnings", "booked_on"]:
                if col in match.index:
                    enriched.at[idx, col] = match[col]
            enriched.at[idx, "match_confidence"] = "high"
            enriched.at[idx, "match_type"] = "exact"
            continue
        
        # Try overlap match
        best_overlap = 0
        best_match = None
        
        for _, res in listing_res.iterrows():
            res_start = res["checkin_date"]
            res_end = res["checkout_date"]
            
            # Calculate overlap
            overlap_start = max(event_start, res_start)
            overlap_end = min(event_end, res_end)
            
            if overlap_start < overlap_end:
                overlap_nights = (overlap_end - overlap_start).days
                res_nights = (res_end - res_start).days
                
                # Require overlap >= 2 nights OR >= 50% of reservation
                if overlap_nights >= 2 or (res_nights > 0 and overlap_nights / res_nights >= 0.5):
                    if overlap_nights > best_overlap:
                        best_overlap = overlap_nights
                        best_match = res
        
        if best_match is not None:
            for col in ["reservation_id", "status", "guest_name", "contact", 
                       "adults", "children", "infants", "earnings", "booked_on"]:
                if col in best_match.index:
                    enriched.at[idx, col] = best_match[col]
            enriched.at[idx, "match_confidence"] = "medium"
            enriched.at[idx, "match_type"] = f"overlap ({best_overlap} nights)"
    
    return enriched


def audit_mismatches(ical_df: pd.DataFrame, res_df: pd.DataFrame, 
                     start_date: date, end_date: date, 
                     listing_filter: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit mismatches between iCal and CSV data."""
    # Filter by date range and listings
    ical_filtered = ical_df[
        (ical_df["start_date"] < end_date) & 
        (ical_df["end_date"] > start_date) &
        (ical_df["listing_key"].isin(listing_filter))
    ]
    
    # iCal events without CSV match
    unmatched_ical = ical_filtered[ical_filtered["match_confidence"] == "none"].copy()
    
    # CSV reservations without iCal match
    res_filtered = res_df[
        (res_df["checkin_date"] < end_date) & 
        (res_df["checkout_date"] > start_date) &
        (res_df["listing_key"].isin(listing_filter))
    ]
    
    unmatched_res = []
    for _, res in res_filtered.iterrows():
        listing_key = res["listing_key"]
        res_start = res["checkin_date"]
        res_end = res["checkout_date"]
        
        # Check if any iCal event overlaps with this reservation
        ical_listing = ical_filtered[ical_filtered["listing_key"] == listing_key]
        has_overlap = False
        
        for _, event in ical_listing.iterrows():
            event_start = event["start_date"]
            event_end = event["end_date"]
            
            overlap_start = max(event_start, res_start)
            overlap_end = min(event_end, res_end)
            
            if overlap_start < overlap_end:
                has_overlap = True
                break
        
        if not has_overlap:
            unmatched_res.append(res)
    
    unmatched_res_df = pd.DataFrame(unmatched_res) if unmatched_res else pd.DataFrame()
    
    return unmatched_ical, unmatched_res_df

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_month_calendar(enriched_df: pd.DataFrame, year: int, month: int, 
                         listing_filter: List[str]):
    """Render a month calendar grid with events."""
    cal = cal_module.Calendar(firstweekday=6)  # Sunday first
    month_days = cal.monthdayscalendar(year, month)
    month_name = cal_module.month_name[month]
    
    st.subheader(f"{month_name} {year}")
    
    # Create header
    cols = st.columns(7)
    for i, day_name in enumerate(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]):
        cols[i].markdown(f"**{day_name}**")
    
    # Color map for listings
    colors = {
        "lanesville": "#FF6B6B",
        "milla": "#4ECDC4",
        "westkill": "#45B7D1",
        "millerroad": "#96CEB4",
        "unknown": "#95A5A6",
    }
    
    # Render weeks
    for week in month_days:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day == 0:
                cols[i].write("")
            else:
                current_date = date(year, month, day)
                
                # Find events for this day
                day_events = enriched_df[
                    (enriched_df["start_date"] <= current_date) & 
                    (enriched_df["end_date"] > current_date) &
                    (enriched_df["listing_key"].isin(listing_filter))
                ]
                
                with cols[i]:
                    st.write(f"**{day}**")
                    for _, event in day_events.iterrows():
                        listing_key = event["listing_key"]
                        color = colors.get(listing_key, "#95A5A6")
                        
                        if event.get("match_confidence") != "none" and pd.notna(event.get("guest_name")):
                            guest = str(event["guest_name"])[:15]
                            earnings = event.get("earnings", 0)
                            label = f"{guest}"
                            if earnings and earnings > 0:
                                label += f" (${earnings:.0f})"
                        else:
                            label = "Blocked"
                        
                        st.markdown(
                            f'<span style="background-color:{color};color:white;'
                            f'padding:2px 6px;border-radius:3px;font-size:11px;">'
                            f'{listing_key[:4]}: {label}</span>',
                            unsafe_allow_html=True
                        )


def render_list_view(enriched_df: pd.DataFrame, start_date: date, end_date: date,
                    listing_filter: List[str], show_unmatched_only: bool = False):
    """Render list/table view of events."""
    filtered = enriched_df[
        (enriched_df["start_date"] < end_date) & 
        (enriched_df["end_date"] > start_date) &
        (enriched_df["listing_key"].isin(listing_filter))
    ].copy()
    
    if show_unmatched_only:
        filtered = filtered[filtered["match_confidence"] == "none"]
    
    if filtered.empty:
        st.info("No events found for the selected criteria.")
        return
    
    # Prepare display columns
    display_cols = [
        "listing_name", "start_date", "end_date", "nights_blocked",
        "guest_name", "status", "earnings", "match_confidence"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    filtered_display = filtered[display_cols].sort_values(["start_date", "listing_name"])
    
    st.dataframe(filtered_display, use_container_width=True, hide_index=True)
    
    # Export button
    csv_data = filtered.to_csv(index=False)
    st.download_button(
        "📥 Export to CSV",
        csv_data,
        "calendar_events.csv",
        "text/csv",
    )


def render_summary(enriched_df: pd.DataFrame, res_df: pd.DataFrame,
                  start_date: date, end_date: date, listing_filter: List[str],
                  csv_enabled: bool):
    """Render summary statistics."""
    filtered = enriched_df[
        (enriched_df["start_date"] < end_date) & 
        (enriched_df["end_date"] > start_date) &
        (enriched_df["listing_key"].isin(listing_filter))
    ]
    
    st.subheader("Blocked Nights by Listing")
    nights_summary = filtered.groupby("listing_name")["nights_blocked"].sum().reset_index()
    nights_summary.columns = ["Listing", "Blocked Nights"]
    st.dataframe(nights_summary, use_container_width=True, hide_index=True)
    
    if csv_enabled and not res_df.empty:
        st.subheader("Earnings by Listing (Matched Reservations)")
        matched = filtered[filtered["match_confidence"] != "none"]
        if not matched.empty:
            earnings_summary = matched.groupby("listing_name")["earnings"].sum().reset_index()
            earnings_summary.columns = ["Listing", "Total Earnings"]
            earnings_summary["Total Earnings"] = earnings_summary["Total Earnings"].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00"
            )
            st.dataframe(earnings_summary, use_container_width=True, hide_index=True)
        
        # Mismatches
        unmatched_ical, unmatched_res = audit_mismatches(
            enriched_df, res_df, start_date, end_date, listing_filter
        )
        
        if not unmatched_ical.empty:
            st.subheader(f"⚠️ iCal Blocks Without CSV Match ({len(unmatched_ical)})")
            display_cols = ["listing_name", "start_date", "end_date", "nights_blocked", "summary"]
            display_cols = [c for c in display_cols if c in unmatched_ical.columns]
            st.dataframe(unmatched_ical[display_cols], use_container_width=True, hide_index=True)
            
            csv_data = unmatched_ical.to_csv(index=False)
            st.download_button(
                "📥 Export Unmatched iCal",
                csv_data,
                "unmatched_ical.csv",
                "text/csv",
            )
        
        if not unmatched_res.empty:
            st.subheader(f"⚠️ CSV Reservations Without iCal Match ({len(unmatched_res)})")
            display_cols = ["listing_key", "reservation_id", "guest_name", "checkin_date", 
                          "checkout_date", "status", "earnings"]
            display_cols = [c for c in display_cols if c in unmatched_res.columns]
            st.dataframe(unmatched_res[display_cols], use_container_width=True, hide_index=True)


def render_diagnostics(res_df: pd.DataFrame, alias_store: Dict, 
                      feed_errors: Dict[str, str]):
    """Render diagnostics view."""
    st.subheader("Feed Status")
    for key in ICAL_FEEDS:
        if key in feed_errors:
            st.error(f"❌ {LISTING_DISPLAY_NAMES.get(key, key)}: {feed_errors[key]}")
        else:
            st.success(f"✅ {LISTING_DISPLAY_NAMES.get(key, key)}: OK")
    
    st.subheader("Alias Store Stats")
    alias_count = len(alias_store.get("aliases", {}))
    auto_count = sum(1 for a in alias_store.get("aliases", {}).values() if a.get("source") == "auto")
    user_count = sum(1 for a in alias_store.get("aliases", {}).values() if a.get("source") == "user")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Aliases", alias_count)
    col2.metric("Auto-learned", auto_count)
    col3.metric("User-defined", user_count)
    
    if not res_df.empty:
        st.subheader("Unknown Listings")
        unknown = res_df[res_df["listing_key"] == "unknown"]
        if not unknown.empty:
            unknown_counts = unknown.groupby("listing_raw").size().reset_index(name="Count")
            unknown_counts = unknown_counts.sort_values("Count", ascending=False)
            
            st.dataframe(unknown_counts, use_container_width=True, hide_index=True)
            
            # Show classifier scores for each unknown
            st.write("**Classification Scores:**")
            for listing_raw in unknown["listing_raw"].unique()[:5]:
                listing_norm = canonicalize_listing(listing_raw)
                result = classify_listing(listing_norm)
                st.write(f"- **{listing_raw[:50]}...**")
                st.write(f"  Scores: {result['scores']}")
                st.write(f"  Max: {result['max_score']}, 2nd: {result['second_score']}")
        else:
            st.success("All listings successfully classified!")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Airbnb Calendar Manager",
        page_icon="🏠",
        layout="wide",
    )
    
    st.title("🏠 Airbnb Multi-Listing Calendar")
    
    # Initialize session state
    if "alias_store" not in st.session_state:
        st.session_state.alias_store = None
    if "res_df" not in st.session_state:
        st.session_state.res_df = pd.DataFrame()
    if "csv_warnings" not in st.session_state:
        st.session_state.csv_warnings = []
    
    # Sidebar
    with st.sidebar:
        st.header("📅 Date Range")
        
        date_mode = st.radio(
            "Select range",
            ["1 Week", "1 Month", "Custom"],
            horizontal=True,
        )
        
        today = date.today()
        if date_mode == "1 Week":
            start_date = today
            end_date = today + timedelta(days=7)
        elif date_mode == "1 Month":
            start_date = today
            end_date = today + timedelta(days=30)
        else:
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start", today)
            end_date = col2.date_input("End", today + timedelta(days=30))
        
        st.write(f"**{start_date} to {end_date}**")
        
        st.header("🏡 Listings")
        listing_options = list(ICAL_FEEDS.keys())
        listing_filter = st.multiselect(
            "Filter listings",
            options=listing_options,
            default=listing_options,
            format_func=lambda x: LISTING_DISPLAY_NAMES.get(x, x),
        )
        
        if not listing_filter:
            listing_filter = listing_options
        
        # Refresh button
        if st.button("🔄 Refresh Feeds"):
            st.cache_data.clear()
            st.rerun()
        
        st.header("📊 CSV Enrichment")
        csv_enabled = st.checkbox("Enable reservations CSV")
        
        alias_store_path = st.text_input(
            "Alias store path",
            value=DEFAULT_ALIAS_STORE_PATH,
        )
        
        res_df = pd.DataFrame()
        csv_warnings = []
        alias_store = load_alias_store(alias_store_path)
        
        if csv_enabled:
            csv_source = st.radio(
                "CSV Source",
                ["Upload", "Server Path"],
                horizontal=True,
            )
            
            csv_content = None
            
            if csv_source == "Upload":
                uploaded_files = st.file_uploader(
                    "Upload CSV files",
                    type=["csv"],
                    accept_multiple_files=True,
                )
                if uploaded_files:
                    all_dfs = []
                    for uf in uploaded_files:
                        content = uf.read()
                        file_hash = get_file_hash(content)
                        df, warns, alias_store = load_reservations_csv(content, file_hash, alias_store_path)
                        if not df.empty:
                            all_dfs.append(df)
                        csv_warnings.extend(warns)
                    if all_dfs:
                        res_df = pd.concat(all_dfs, ignore_index=True)
            
            elif csv_source == "Server Path":
                csv_path = st.text_input("CSV file path", "/mnt/project/")
                if csv_path and os.path.exists(csv_path):
                    if os.path.isdir(csv_path):
                        csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
                        selected_files = st.multiselect("Select CSV files", csv_files)
                        all_dfs = []
                        for f in selected_files:
                            full_path = os.path.join(csv_path, f)
                            with open(full_path, "rb") as fp:
                                content = fp.read()
                            file_hash = get_file_hash(content)
                            df, warns, alias_store = load_reservations_csv(content, file_hash, alias_store_path)
                            if not df.empty:
                                all_dfs.append(df)
                            csv_warnings.extend(warns)
                        if all_dfs:
                            res_df = pd.concat(all_dfs, ignore_index=True)
                    else:
                        with open(csv_path, "rb") as f:
                            content = f.read()
                        file_hash = get_file_hash(content)
                        res_df, csv_warnings, alias_store = load_reservations_csv(content, file_hash, alias_store_path)
            
            if not res_df.empty:
                st.success(f"Loaded {len(res_df)} reservations")
                
                # Show unknown listings resolver
                unknown = res_df[res_df["listing_key"] == "unknown"]
                if not unknown.empty:
                    with st.expander(f"⚠️ Resolve Unknown Listings ({len(unknown)} rows)"):
                        unknown_listings = unknown["listing_raw"].unique()
                        
                        new_mappings = {}
                        for listing_raw in unknown_listings:
                            listing_norm = canonicalize_listing(listing_raw)
                            selected = st.selectbox(
                                f'"{listing_raw[:40]}..."',
                                options=["unknown"] + list(ICAL_FEEDS.keys()),
                                key=f"resolve_{listing_norm}",
                                format_func=lambda x: LISTING_DISPLAY_NAMES.get(x, x),
                            )
                            if selected != "unknown":
                                new_mappings[listing_norm] = (selected, listing_raw)
                        
                        if new_mappings:
                            if st.button("💾 Save Mappings"):
                                for norm, (key, raw) in new_mappings.items():
                                    alias_store = add_alias(alias_store, norm, key, "user", 1.0, raw)
                                if save_alias_store(alias_store, alias_store_path):
                                    st.success("Mappings saved!")
                                    st.cache_data.clear()
                                    st.rerun()
            
            if csv_warnings:
                for warn in csv_warnings:
                    st.warning(warn)
        
        # Alias store admin
        st.header("🔧 Alias Store Admin")
        col1, col2 = st.columns(2)
        if col1.button("Reload Alias File"):
            alias_store = load_alias_store(alias_store_path)
            st.success("Reloaded")
        
        with col2:
            confirm_reset = st.checkbox("Confirm reset")
            if st.button("Reset Alias File", disabled=not confirm_reset):
                alias_store = get_default_alias_store()
                if save_alias_store(alias_store, alias_store_path):
                    st.success("Reset complete")
                    st.cache_data.clear()
                    st.rerun()
    
    # Main content
    # Fetch iCal feeds
    with st.spinner("Fetching calendar feeds..."):
        ical_df, feed_errors = fetch_all_ical_feeds()
    
    # Enrich with CSV if enabled
    if csv_enabled and not res_df.empty:
        enriched_df = enrich_ical_events(ical_df, res_df)
        # Save updated alias store
        save_alias_store(alias_store, alias_store_path)
    else:
        enriched_df = ical_df.copy()
        if not enriched_df.empty:
            enriched_df["match_confidence"] = "none"
            enriched_df["guest_name"] = None
            enriched_df["earnings"] = None
    
    # Show feed errors
    if feed_errors:
        with st.expander(f"⚠️ {len(feed_errors)} Feed Errors"):
            for key, error in feed_errors.items():
                st.error(f"{LISTING_DISPLAY_NAMES.get(key, key)}: {error}")
    
    # Tabs
    tabs = st.tabs(["📅 Month Calendar", "📋 List View", "📊 Summary", "🔍 Diagnostics"])
    
    with tabs[0]:
        # Month calendar view
        if enriched_df.empty:
            st.info("No calendar events found.")
        else:
            # Navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            
            view_year = col2.number_input("Year", min_value=2020, max_value=2030, 
                                         value=start_date.year, key="cal_year")
            view_month = col2.number_input("Month", min_value=1, max_value=12, 
                                          value=start_date.month, key="cal_month")
            
            render_month_calendar(enriched_df, int(view_year), int(view_month), listing_filter)
    
    with tabs[1]:
        show_unmatched = st.checkbox("Show only unmatched blocks", key="list_unmatched")
        render_list_view(enriched_df, start_date, end_date, listing_filter, show_unmatched)
    
    with tabs[2]:
        render_summary(enriched_df, res_df, start_date, end_date, listing_filter, csv_enabled)
    
    with tabs[3]:
        render_diagnostics(res_df, alias_store, feed_errors)


if __name__ == "__main__":
    main()
