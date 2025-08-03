# Doppler Shift Calculator for Marching Band

A web application for calculating the doppler shift effect in marching band formations on a football field. This tool helps analyze how sound frequency changes as performers move across the field relative to an observer position.

## Features

- **Interactive Interface**: User-friendly web interface with intuitive controls
- **Observer Position Options**: Choose between custom positioning or Lucas Oil Box preset
- **Visual Output**: Interactive football field layout and doppler shift analysis plots
- **Real-time Calculations**: Instant results with Plotly-powered visualizations
- **Comprehensive Analysis**: Detailed doppler shift calculations and measurements

## What It Does

This application calculates the doppler shift effect that occurs when sound sources (like marching band performers) move across a football field relative to a stationary observer. The doppler shift is the change in frequency of a wave (sound) as the source moves toward or away from the observer.

### Key Calculations:
- **Position Tracking**: Maps performer start and end positions on a standard football field
- **Observer Positioning**: Supports custom observer placement or preset Lucas Oil Box position
- **Doppler Analysis**: Calculates frequency shifts based on relative motion
- **Visualization**: Displays field layout and frequency change over time

## Technical Details

- **Framework**: Streamlit
- **Python Version**: 3.12+
- **Key Libraries**: NumPy, Matplotlib, Pandas, Plotly
- **Plotting**: Interactive Plotly visualizations
- **Field Standards**: NCAA football field specifications

## Local Development

### Setup

```bash
# Create virtual environment with Python 3.12+
python3.12 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Run the Application

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Usage

1. **Configure Parameters**:
   - Musical parameters (tempo, counts, frequency)
   - Observer position on the field (custom or Lucas Oil Box preset)
   - Start and end positions for the sound source
2. **Calculate**: Click "Calculate Doppler Shift" to generate results
3. **View Results**: 
   - Interactive football field layout with positions
   - Doppler shift vs count plot
   - Comprehensive analysis data

## Deployment

This application can be deployed to various cloud platforms including Streamlit Cloud, Heroku, Railway, or Render. See the respective platform documentation for deployment instructions. 