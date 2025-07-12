
# ------- Imports -------
import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_option_menu import option_menu
from streamlit_folium import folium_static
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from functools import lru_cache
import joblib
from geopy.distance import geodesic

# ------- Geolocator Setup -------
geolocator = Nominatim(user_agent="earthquake_map")

@lru_cache(maxsize=None)
def get_location_name(lat, lon):
    """Retrieve a human-readable location name from latitude and longitude."""
    try:
        location = geolocator.reverse((lat, lon), language='en', timeout=5)
        if location and 'address' in location.raw:
            address = location.raw['address']
            # Extract relevant location info, fallback to "Unknown" if missing
            city = address.get('city', address.get('town', address.get('village', 'Unknown')))
            country = address.get('country', 'Unknown')
            return f"{city}, {country}" if city != "Unknown" else country  # Return 'City, Country' if possible
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Geocoding error: {e}")
    return "Unknown Location"


# ------- MongoDB Setup -------
MONGO_URI = "mongodb://localhost:27017/"  # Update this with your actual MongoDB URI

try:
    client = MongoClient(MONGO_URI)
    db = client["Quake"]  # Replace with your database name
    collection = db["quakes"]  # Replace with your collection name
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")

# ------- Sidebar Navigation -------
with st.sidebar:
    menu = option_menu(
        'Earthquake Prediction Dashboard',
        ['Home', 'Earthquake Map', 'Trends','Legends','PredModel'],
        icons=['house', 'globe', 'bar-chart-line'],
        default_index=0
    )

# ------- Data Loading Function -------
@st.cache_data
def load_data():
    """Load earthquake data from MongoDB into a DataFrame."""
    try:
        data = list(collection.find())  # Fetch data from MongoDB
        
        if len(data) == 0:
            st.warning("No data found in the MongoDB collection!")
        
        df = pd.DataFrame(data)
        
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)  # Drop the MongoDB ObjectID field
        
        return df
    except Exception as e:
        st.error(f"Error loading data from MongoDB: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if loading fails

# Load data into the DataFrame
df = load_data()

# =============================
# End of Header Section
# =============================

model_filename = "quake_model.pkl"
model_reg = joblib.load(model_filename)
# st.write(f"Model loaded from {model_filename}")


if menu == "Home":
    st.title("🌍 Earthquake Analysis Dashboard")

    # Basic Statistics
    total_earthquakes = df.shape[0]
    unique_years = df['Year'].nunique()  # Count of unique years
    average_yearly_earthquakes = total_earthquakes / unique_years if unique_years > 0 else 0

    # Calculate average Magnitude and maximum Depth directly
    average_magnitude = df['Magnitude'].mean() if not df['Magnitude'].empty else 0
    max_depth = df['Depth'].max() if not df['Depth'].empty else 0

    # Count unique types of earthquakes
    unique_types_count = df['Type'].nunique()

    # Display metrics in a box
    st.markdown(
    """
    <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
        <div style='background-color: #697565; padding: 20px; border-radius: 10px; margin: 10px; flex: 1; min-width: 200px; color: #000000;'>
            <h5>Total Earthquakes</h5>
            <h3>""" + str(total_earthquakes) + """</h3>
        </div>
        <div style='background-color: #697565; padding: 20px; border-radius: 10px; margin: 10px; flex: 1; min-width: 200px; color: #000000;'>
            <h5>Average Magnitude</h5>
            <h3>""" + str(round(average_magnitude, 2)) + """</h3>
        </div>
        <div style='background-color: #697565; padding: 20px; border-radius: 10px; margin: 10px; flex: 1; min-width: 200px; color: #000000;'>
            <h5>Max Depth (km)</h5>
            <h3>""" + str(max_depth) + """</h3>
        </div>
        <div style='background-color: #697565; padding: 20px; border-radius: 10px; margin: 10px; flex: 1; min-width: 200px; color: #000000;'>
            <h5>Types of Earthquakes</h5>
            <h3>""" + str(unique_types_count) + """</h3>
        </div>
        <div style='background-color: #697565; padding: 20px; border-radius: 10px; margin: 10px; flex: 1; min-width: 200px; color: #000000;'>
            <h5>Average Yearly Earthquakes</h5>
            <h3>""" + str(round(average_yearly_earthquakes, 2)) + """</h3>
        </div>
    </div>
    """, unsafe_allow_html=True
    )

    # Trend Over Time
    st.subheader("Yearly Earthquake Counts")
    yearly_counts = df['Year'].value_counts().sort_index()

    # Create a line chart with orange color using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_counts.index, 
        y=yearly_counts.values, 
        mode='lines+markers', 
        line=dict(color='orange'),  
        name='Yearly Counts'
    ))

    fig.update_layout(
        title="Yearly Earthquake Counts",
        xaxis_title="Year",
        yaxis_title="Number of Earthquakes",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Download button for the data
    st.subheader("Download Data")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='earthquake_data.csv',
        mime='text/csv',
    )



  


elif menu == "Earthquake Map":
    st.sidebar.subheader("Filter by Year")

    # Ensure 'Year' column exists in DataFrame
    if 'Year' in df.columns:
        year_options = df['Year'].unique()
        selected_year = st.sidebar.selectbox("Select Year", sorted(year_options, reverse=True))

        # Cache filtered DataFrame
        @st.cache_data
        def load_data(year):
            return df[df['Year'] == year]

        filtered_df = load_data(selected_year)

        # Page Header
        st.subheader(f"Data Summary for the Year: {selected_year}")

        # Global Earthquake Map
        st.subheader("🗺️ Global Earthquake Map")
        
        # Cache map creation
        @st.cache_resource
        def create_map(data):
            m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter", prefer_canvas=True)
            marker_cluster = MarkerCluster(options={'spiderfyOnMaxZoom': False}).add_to(m)

            # Add markers to map
            for _, row in data.iterrows():
                location_name = get_location_name(row['Latitude'], row['Longitude'])
                folium.CircleMarker(
                    location=(row['Latitude'], row['Longitude']),
                    radius=min(row['Magnitude'] * 2, 10),
                    color='orange',
                    fill=True,
                    fill_color='orange',
                    fill_opacity=0.6,
                    popup=f"<b>Magnitude:</b> {row['Magnitude']}<br><b>Location:</b> {location_name}"
                ).add_to(marker_cluster)

            return m

        # Generate and display map
        m = create_map(filtered_df)
        st.components.v1.html(m._repr_html_(), height=500, width=800)

        # Calculate Min and Max Magnitude
        min_magnitude = filtered_df['Magnitude'].min() if 'Magnitude' in filtered_df.columns else 0
        max_magnitude = filtered_df['Magnitude'].max() if 'Magnitude' in filtered_df.columns else 0

        # Magnitude Scale HTML
        st.markdown(f"""
        <div style='background-color: #222; padding: 10px; border-radius: 10px; margin-top: 20px;'>
            <h4 style='color: white; text-align: center;'>Magnitude Scale</h4>
            <div style='text-align: left;'>
                <div style='width: 100%; height: 10px; background-color: orange;'></div>
                <div style='font-size: 12px; color: white;'>
                    <h5>Min: {min_magnitude}</h5> 
                    <h5>Max: {max_magnitude}</h5>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key Performance Indicators
        st.markdown("<h2 style='text-align: center;'>Key Performance Indicators</h2>", unsafe_allow_html=True)

        with st.container():
            col1, col2, col3 = st.columns(3)

            # Total Earthquakes KPI
            total_earthquakes = filtered_df.shape[0]
            with col1:
                st.markdown(
                    f"""
                    <div style='border: 2px solid orange; border-radius: 10px; padding: 20px; text-align: center;'>
                        <h3>Total Earthquakes</h3>
                        <h2>{total_earthquakes}</h2>
                    </div>
                    """, unsafe_allow_html=True
                )

            # Max Magnitude KPI
            with col2:
                st.markdown(
                    f"""
                    <div style='border: 2px solid orange; border-radius: 10px; padding: 20px; text-align: center;'>
                        <h3>Max Magnitude</h3>
                        <h2>{max_magnitude}</h2>
                    </div>
                    """, unsafe_allow_html=True
                )

            # Avg Magnitude KPI
            with col3:
                avg_magnitude = round(filtered_df['Magnitude'].mean(), 2) if 'Magnitude' in filtered_df.columns else "N/A"
                st.markdown(
                    f"""
                    <div style='border: 2px solid orange; border-radius: 10px; padding: 20px; text-align: center;'>
                        <h3>Avg Magnitude</h3>
                        <h2>{avg_magnitude}</h2>
                    </div>
                    """, unsafe_allow_html=True
                )

        # Footer
        st.markdown("---")
        st.markdown("<h5 style='text-align: center;'>Earthquake Analytics Dashboard | Powered by Streamlit & MongoDB</h5>", unsafe_allow_html=True)





elif menu == "Trends":
    st.subheader("📈 Earthquake Trends by Year ")

    # Ensure that data is loaded correctly
    if df.empty:
        st.warning("No earthquake data found!")
    else:
        # Convert columns to appropriate types
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
        df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
        df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

        # Drop rows with missing values in critical columns
        df = df.dropna(subset=['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Year'])

        # Year range selection
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        year_range = st.slider("Select Year Range", min_value=min_year, max_value=max_year,
                               value=(min_year, min_year + 5), step=1)

        

        # Filter data by selected year range
        year_data = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

        if not year_data.empty:
            # Top 50 earthquakes by magnitude
            top_earthquakes = year_data.nlargest(50, 'Magnitude')

            # Most and least severe earthquakes
            most_severe = top_earthquakes.loc[top_earthquakes['Magnitude'].idxmax()]
            least_severe = top_earthquakes.loc[top_earthquakes['Magnitude'].idxmin()]

            # Side-by-side columns for details
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Most Severe Earthquake")
                st.write(f"- **Magnitude:** {most_severe['Magnitude']}")
                st.write(f"- **Date:** {most_severe['Date']}")
                st.write(f"- **Depth (km):** {most_severe['Depth']}")
                st.write(f"- **Location:** {get_location_name(most_severe['Latitude'], most_severe['Longitude'])}")

            with col2:
                st.markdown("### Least Severe Earthquake")
                st.write(f"- **Magnitude:** {least_severe['Magnitude']}")
                st.write(f"- **Date:** {least_severe['Date']}")
                st.write(f"- **Depth (km):** {least_severe['Depth']}")
                st.write(f"- **Location:** {get_location_name(least_severe['Latitude'], least_severe['Longitude'])}")

            # Legend for red and blue dots
            st.markdown("""
                <div style="display: flex; gap: 15px; margin-top: 15px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 15px; height: 15px; background-color: red; border-radius: 50%;"></div>
                        <p style="margin: 0 0 0 8px;">Most Severe Earthquake</p>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 15px; height: 15px; background-color: blue; border-radius: 50%;"></div>
                        <p style="margin: 0 0 0 8px;">Least Severe Earthquake</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Map with most/least severe earthquakes and cluster for others
            with st.spinner('Generating map...'):
                avg_lat = top_earthquakes['Latitude'].mean()
                avg_lon = top_earthquakes['Longitude'].mean()
                quake_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=3, tiles='CartoDB dark_matter')

                # Most and least severe markers
                folium.CircleMarker(
                    location=(most_severe['Latitude'], most_severe['Longitude']),
                    radius=12, color='red', fill=True, fill_opacity=1,
                    popup=f"Most Severe Earthquake: {most_severe['Magnitude']}<br>Date: {most_severe['Date']}"
                ).add_to(quake_map)

                folium.CircleMarker(
                    location=(least_severe['Latitude'], least_severe['Longitude']),
                    radius=12, color='blue', fill=True, fill_opacity=1,
                    popup=f"Least Severe Earthquake: {least_severe['Magnitude']}<br>Date: {least_severe['Date']}"
                ).add_to(quake_map)

                # Add clustered markers for remaining earthquakes
                marker_cluster = MarkerCluster().add_to(quake_map)
                for _, quake in top_earthquakes.iterrows():
                    if quake['Magnitude'] in [most_severe['Magnitude'], least_severe['Magnitude']]:
                        continue
                    folium.CircleMarker(
                        location=(quake['Latitude'], quake['Longitude']),
                        radius=5, color='orange', fill=True, fill_opacity=0.8,
                        popup=f"Magnitude: {quake['Magnitude']}<br>Date: {quake['Date']}"
                    ).add_to(marker_cluster)

                folium_static(quake_map)

        else:
            st.warning("No earthquake data found for the selected year range.")

        # Trend Analysis: Max and Average Magnitude
        yearly_stats = df.groupby("Year").agg(
            Total_Earthquakes=("ID", "count"),
            Max_Magnitude=("Magnitude", "max"),
            Avg_Magnitude=("Magnitude", "mean")
        ).reset_index()

        # Plot trends: Max and Avg Magnitude with orange color
        st.markdown("### Max and Average Earthquake Magnitude by Year")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=yearly_stats["Year"], y=yearly_stats["Max_Magnitude"],
                                  mode="lines+markers", name="Max Magnitude", line=dict(color='orange')))
        fig2.add_trace(go.Scatter(x=yearly_stats["Year"], y=yearly_stats["Avg_Magnitude"],
                                  mode="lines+markers", name="Avg Magnitude", line=dict(color='darkorange')))
        fig2.update_layout( 
                           xaxis_title="Year", yaxis_title="Magnitude")
        st.plotly_chart(fig2, use_container_width=True)

        # Bar chart: Number of Earthquakes by Year with orange color
        st.markdown("### Number of Earthquakes by Year")
        fig1 = px.bar(yearly_stats, x="Year", y="Total_Earthquakes",
                      labels={"Total_Earthquakes": "Number of Earthquakes"}, color='Total_Earthquakes',
                      color_discrete_sequence=['orange'])  # Set bar color to orange
        st.plotly_chart(fig1, use_container_width=True)

        
        # Histogram of Earthquake Magnitudes
        st.markdown("### Histogram of Earthquake Magnitudes")
        fig_hist = px.histogram(df, x='Magnitude',  
                                labels={'Magnitude': 'Magnitude'}, 
                                color_discrete_sequence=['orange'])  # Set histogram color to orange
        st.plotly_chart(fig_hist, use_container_width=True)


        # Earthquake Type Distribution as an Interactive Pie Chart
        st.markdown("### Distribution of Earthquake Types (Interactive Pie Chart)")
        type_counts = df['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']

        # Create the pie chart using Plotly
        fig_pie = px.pie(type_counts, values='Count', names='Type', 
                        title='Distribution of Earthquake Types', 
                        color_discrete_sequence=px.colors.sequential.Oryel)
        st.plotly_chart(fig_pie, use_container_width=True)






elif menu == "Legends":
    st.subheader("📊 Earthquake Legends")

    # Load the dataset from MongoDB
    df = load_data()

    # Convert necessary columns to numeric and handle missing values
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year  # Extract year from 'Date' column

    # Drop rows with missing values to ensure plotting is smooth
    df = df.dropna(subset=['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Year'])

    # Convert 'Year' to integer for consistency
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())

    # Year range selection
    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, min_year + 5),  # Default to first 5 years
        step=1  # Keeping the step as an integer
    )

    # Filter data based on the selected year range
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

    if filtered_df.empty:
        st.warning("No earthquake data found for the selected year range.")
    else:
        # Set figure style for dark mode
        plt.style.use('dark_background')

        # Create a color palette for orange shades
        orange_palette = plt.cm.Oranges(np.linspace(0.2, 1, filtered_df['Magnitude'].nunique()))

        # Latitude vs Magnitude
        st.markdown("### Latitude vs. Earthquake Magnitude")
        st.write("This scatter plot shows the relationship between latitude and earthquake magnitude.")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        scatter1 = ax1.scatter(
            filtered_df['Latitude'], filtered_df['Magnitude'],
            c=filtered_df['Magnitude'], cmap='Oranges', edgecolor='black'
        )
        ax1.set_xlabel('Latitude', color='white')
        ax1.set_ylabel('Magnitude', color='white')
        ax1.grid(True)

        # Customize the legend
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Magnitude", loc='upper left')
        legend1.get_frame().set_facecolor('none')  # Remove legend background
        for text in legend1.get_texts():
            text.set_color('white')  # Change legend text color to white

        # Render the first plot in Streamlit
        st.pyplot(fig1)

        # Longitude vs Magnitude with trend line
        st.markdown("### Longitude vs. Earthquake Magnitude")
        st.write("This plot illustrates how earthquake magnitudes vary with longitude, including a trend line.")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        scatter2 = ax2.scatter(
            filtered_df['Longitude'], filtered_df['Magnitude'],
            c=filtered_df['Magnitude'], cmap='Oranges', edgecolor='black'
        )
        ax2.set_xlabel('Longitude', color='white')
        ax2.set_ylabel('Magnitude', color='white')
        ax2.grid(True)

        # Add trend line
        z = np.polyfit(filtered_df['Longitude'], filtered_df['Magnitude'], 1)
        p = np.poly1d(z)
        ax2.plot(filtered_df['Longitude'], p(filtered_df['Longitude']), color='orange', linestyle='--', linewidth=2)

        # Customize the legend
        legend2 = ax2.legend(*scatter2.legend_elements(), title="Magnitude", loc='upper left')
        legend2.get_frame().set_facecolor('none')  # Remove legend background
        for text in legend2.get_texts():
            text.set_color('white')  # Change legend text color to white

        # Render the second plot in Streamlit
        st.pyplot(fig2)

        # Depth vs Magnitude with size for depth
        st.markdown("### Depth vs. Earthquake Magnitude")
        st.write("This scatter plot visualizes the relationship between the depth of earthquakes and their magnitudes, with point size representing depth.")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        scatter3 = ax3.scatter(
            filtered_df['Depth'], filtered_df['Magnitude'],
            s=filtered_df['Depth'] * 10, c=filtered_df['Magnitude'], cmap='Oranges', edgecolor='black', alpha=0.7
        )
        ax3.set_xlabel('Depth (km)', color='white')
        ax3.set_ylabel('Magnitude', color='white')
        ax3.grid(True)

        # Customize the legend
        legend3 = ax3.legend(*scatter3.legend_elements(), title="Magnitude", loc='upper left')
        legend3.get_frame().set_facecolor('none')  # Remove legend background
        for text in legend3.get_texts():
            text.set_color('white')  # Change legend text color to white

        # Render the third plot in Streamlit
        st.pyplot(fig3)

    st.markdown("---")  # Optional: Adds a horizontal line for better visual separation




    


if menu == "PredModel":
    st.subheader("Earthquake Magnitude Prediction")
    
    # Get user input for latitude, longitude, and depth
    latitude = st.number_input("Enter Latitude:", format="%.6f")
    longitude = st.number_input("Enter Longitude:", format="%.6f")
    depth = st.number_input("Enter Depth (km):", format="%.1f")

    if st.button("Predict Magnitude"):
        # Function to make a prediction using the trained model
        def make_prediction(lat, lon, depth):
            input_data = np.array([[lat, lon, depth]])
            prediction = model_reg.predict(input_data)
            return prediction[0]

        # Make a prediction using the model
        predicted_magnitude = make_prediction(latitude, longitude, depth)
        st.write(f"Predicted Magnitude: {predicted_magnitude:.2f}")

        # Function to find the nearest earthquake
        def find_nearest_earthquake(lat, lon):
            nearest_quake = None
            min_distance = float('inf')

            # Fetch all earthquake records from the MongoDB collection
            earthquakes = collection.find({"Type": "Earthquake"})

            for quake in earthquakes:
                quake_location = (quake['Latitude'], quake['Longitude'])
                user_location = (lat, lon)

                # Calculate distance using geodesic
                distance = geodesic(user_location, quake_location).kilometers

                # Update nearest earthquake if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    nearest_quake = quake

            return nearest_quake

        # Find the nearest earthquake
        nearest_quake = find_nearest_earthquake(latitude, longitude)

        if nearest_quake:
            # Create a four-column layout for earthquake details
            col1, col2, col3, col4 = st.columns(4)
            st.markdown("### Nearest Earthquake Details:")
            with col1:
                st.write(f"- **Date:** {nearest_quake['Date']}")
            with col2:
                st.write(f"- **Magnitude:** {nearest_quake['Magnitude']}")
            with col3:
                st.write(f"- **Depth:** {nearest_quake['Depth']} km")
            with col4:
                st.write(f"- **Location:** ({nearest_quake['Latitude']}, {nearest_quake['Longitude']})")

            # Create a dark-themed Folium map centered at the user's input location
            quake_map = folium.Map(location=[latitude, longitude], zoom_start=5, tiles='CartoDB dark_matter')

            # Add the nearest earthquake location to the map
            folium.Marker(
                location=[nearest_quake['Latitude'], nearest_quake['Longitude']],
                popup=(f"Nearest Earthquake<br>"
                       f"Date: {nearest_quake['Date']}<br>"
                       f"Magnitude: {nearest_quake['Magnitude']}"),
                icon=folium.Icon(color='red')
            ).add_to(quake_map)

            # Display the map
            folium_static(quake_map)
        else:
            st.warning("No earthquake data found.")





client.close()
