import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from deep_translator import GoogleTranslator

# CSV file paths for storing data
GRIPPER_FILE = "grippers.csv"
TACTILE_FILE = "tactile_sensors.csv"
PROXIMITY_FILE = "proximity_sensors.csv"


# Translator function
def translate_text(text, lang='en'):
    if lang == 'de':
        return GoogleTranslator(source='en', target='de').translate(text)
    return text
# Function to save uploaded image
def save_uploaded_image(image, name):
    # Ensure the 'images/' directory exists
    if not os.path.exists("images"):
        os.makedirs("images")  # Create directory if it doesn't exist

    # Define the image path
    image_path = f"images/{name}.png"

    # Save the uploaded image to the specified path
    with open(image_path, "wb") as f:
        f.write(image.getbuffer())

    return image_path

# Function to load or create initial data
def load_data(file_path, default_data):
    """
    Load the CSV file or create it with default data.
    Add missing columns for backward compatibility.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Check for missing columns
        required_columns = list(default_data[0].keys())  # Get column names from default data
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Add missing column with default value

        # Save updated DataFrame back to file
        df.to_csv(file_path, index=False)
    else:
        df = pd.DataFrame(default_data)
        df.to_csv(file_path, index=False)

    return df



# Initial data for each category
gripper_data = [
    {'Name': 'Vacuum Gripper', 'Cost': 6, 'ISO Compliance': 8, 'Safety': 9, 'Performance': 7, 'Image': None, 'Research URL': None},
    {'Name': 'Soft Robotic Gripper', 'Cost': 7, 'ISO Compliance': 9, 'Safety': 8, 'Performance': 9, 'Image': None, 'Research URL': None}
]

tactile_data = [
    {'Name': 'Capacitive Sensor', 'Cost': 5, 'ISO Compliance': 8, 'Safety': 8, 'Performance': 9, 'Image': None, 'Research URL': None},
    {'Name': 'Piezoresistive Sensor', 'Cost': 6, 'ISO Compliance': 9, 'Safety': 9, 'Performance': 8, 'Image': None, 'Research URL': None}
]

proximity_data = [
    {'Name': 'Ultrasonic Sensor', 'Cost': 4, 'ISO Compliance': 7, 'Safety': 9, 'Performance': 8, 'Image': None, 'Research URL': None},
    {'Name': 'LIDAR Sensor', 'Cost': 8, 'ISO Compliance': 9, 'Safety': 8, 'Performance': 10, 'Image': None, 'Research URL': None}
]


# Load data
grippers = load_data(GRIPPER_FILE, gripper_data)
tactile_sensors = load_data(TACTILE_FILE, tactile_data)
proximity_sensors = load_data(PROXIMITY_FILE, proximity_data)


# Weighted scoring function
def calculate_scores(df, weights):
    for factor, weight in weights.items():
        df[f'{factor} Weighted'] = df[factor] * weight
    df['Total Score'] = df[[f'{factor} Weighted' for factor in weights]].sum(axis=1)

    # Recalculate Rank based on the updated Total Score
    df['Rank'] = df['Total Score'].rank(ascending=False)
    return df


# Function to add new entry
def add_entry(file_path, df, entry):
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(file_path, index=False)
    return df


# Function to plot visualizations
def plot_visualizations(df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    melted_df = df.melt(id_vars=['Name'], value_vars=['Cost', 'ISO Compliance', 'Safety', 'Performance'],
                        var_name='Factors', value_name='Scores')
    sns.barplot(data=melted_df, x='Factors', y='Scores', hue='Name', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


# Streamlit app
st.title(translate_text("Weighted Model Analysis for Grippers and Sensors"))

# Language Selection
selected_language = st.selectbox("Select Language / Sprache auswählen", options=["en", "de"],
                                 format_func=lambda x: "English" if x == "en" else "Deutsch")


# Sliders for weight adjustments
def weight_sliders(prefix):
    cost = st.slider(translate_text("Weight for Cost", selected_language), 0.0, 1.0, 0.25, key=f"{prefix}_cost")
    iso = st.slider(translate_text("Weight for ISO Compliance", selected_language), 0.0, 1.0, 0.25, key=f"{prefix}_iso")
    safety = st.slider(translate_text("Weight for Safety", selected_language), 0.0, 1.0, 0.25, key=f"{prefix}_safety")
    performance = st.slider(translate_text("Weight for Performance", selected_language), 0.0, 1.0, 0.25,
                            key=f"{prefix}_performance")
    return {"Cost": cost, "ISO Compliance": iso, "Safety": safety, "Performance": performance}


# Tabs for different categories
tab1, tab2, tab3 = st.tabs([
    translate_text("Grippers", selected_language),
    translate_text("Tactile Sensors", selected_language),
    translate_text("Proximity Sensors", selected_language)
])

# Gripper tab
# Tab for Grippers
with tab1:
    st.header(translate_text("Grippers", selected_language))

    # Slider for adjusting weights dynamically
    gripper_weights = weight_sliders("grippers")
    grippers = calculate_scores(grippers, gripper_weights)
    st.write(grippers[['Name', 'Total Score', 'Rank']])  # Display updated gripper table
    plot_visualizations(grippers, translate_text("Gripper Analysis", selected_language))  # Visualization

    # Add New Gripper Section
    st.subheader(translate_text("Add New Gripper", selected_language))
    name = st.text_input(translate_text("Name", selected_language), key="add_gripper_name")

    # Cost-related sub-parameters
    st.write(translate_text("Enter values for cost-related parameters (1-10):", selected_language))
    material_cost = st.number_input(translate_text("Material Cost", selected_language), min_value=1, max_value=10,
                                    value=5, key="gripper_material_cost")
    manufacturing_cost = st.number_input(translate_text("Manufacturing Cost", selected_language), min_value=1,
                                         max_value=10, value=5, key="gripper_manufacturing_cost")
    maintenance_cost = st.number_input(translate_text("Maintenance Cost", selected_language), min_value=1, max_value=10,
                                       value=5, key="gripper_maintenance_cost")
    energy_consumption = st.number_input(translate_text("Energy Consumption", selected_language), min_value=1,
                                         max_value=10, value=5, key="gripper_energy_cost")


    # Calculate overall cost weight dynamically
    def calculate_cost_weight(material, manufacturing, maintenance, energy):
        weights = {"Material Cost": 0.4, "Manufacturing Cost": 0.3, "Maintenance Cost": 0.2, "Energy Consumption": 0.1}
        return round(material * weights["Material Cost"] + manufacturing * weights["Manufacturing Cost"] +
                     maintenance * weights["Maintenance Cost"] + energy * weights["Energy Consumption"], 2)


    cost_weight = calculate_cost_weight(material_cost, manufacturing_cost, maintenance_cost, energy_consumption)


    # ISO-related sub-parameters
    st.write(translate_text("Enter values for ISO-related parameters (1-10):", selected_language))
    compliance_standards = st.number_input(translate_text("Compliance Standards", selected_language), min_value=1, max_value=10,
                                    value=5, key="gripper_Compliance_Standards")
    material_safety = st.number_input(translate_text("Material Safety", selected_language), min_value=1,
                                         max_value=10, value=5, key="material_safety")

    accuracy_standards = st.number_input(translate_text("Accuracy Standards", selected_language), min_value=1,
                                         max_value=10, value=5, key="gripper_accuracy_standards")


    # Calculate overall ISO weight dynamically
    def calculate_iso_weight(Icompliance, Isafety, Iaccuracy):
        weights = {"Compliance Standards": 0.4, "Material Safety": 0.3, "Accuracy Standards": 0.3}
        return round(Icompliance * weights["Compliance Standards"] + Isafety * weights["Material Safety"] +
                     Iaccuracy * weights["Accuracy Standards"], 2)


    iso_weight = calculate_iso_weight(compliance_standards, material_safety, accuracy_standards)




    # Safety-related sub-parameters
    st.write(translate_text("Enter values for Safety-related parameters (1-10):", selected_language))
    impact_resistance = st.number_input(translate_text("Impact Resistance", selected_language), min_value=1,
                                           max_value=10,
                                           value=5, key="gripper_Impact_Resistance")
    fail_safe = st.number_input(translate_text("Fail-Safe Mechanisms", selected_language), min_value=1,
                                      max_value=10, value=5, key="gripper_fail_safe")

    force_limitation = st.number_input(translate_text("Force Limitation", selected_language), min_value=1,
                                         max_value=10, value=5, key="gripper_force_limitation")


    # Calculate overall safety weight dynamically
    def calculate_safety_weight(gimpact, gfail_safe, gforce_limitation):
        weights = {"Impact Resistance": 0.4, "Fail-Safe Mechanisms": 0.3, "Force Limitation": 0.3}
        return round(gimpact * weights["Impact Resistance"] + gfail_safe * weights["Fail-Safe Mechanisms"] +
                     gforce_limitation * weights["Force Limitation"], 2)


    safety_weight = calculate_safety_weight(impact_resistance, fail_safe, force_limitation)

    # performance-related sub-parameters
    st.write(translate_text("Enter values for Performance-related parameters (1-10):", selected_language))
    grip_precision = st.number_input(translate_text("Precision", selected_language), min_value=1,
                                        max_value=10,
                                        value=5, key="gripper_precision")
    grip_response_time = st.number_input(translate_text("Response Time", selected_language), min_value=1,
                                max_value=10, value=5, key="gripper_response_time")

    grip_endurance = st.number_input(translate_text("Endurance", selected_language), min_value=1,
                                       max_value=10, value=5, key="gripper_endurance")


    # Calculate overall performance weight dynamically
    def calculate_performance_weight(gprecision, gresponse, gendurance):
        weights = {"Precision": 0.4, "Response Time": 0.3, "Endurance": 0.3}
        return round(gprecision * weights["Precision"] + gresponse * weights["Response Time"] +
                     gendurance * weights["Endurance"], 2)


    performance_weight = calculate_performance_weight(grip_precision, grip_response_time, grip_endurance)



    # Image and Research Paper URL inputs
    image = st.file_uploader(translate_text("Upload Image", selected_language), type=['jpg', 'jpeg', 'png'],
                             key="gripper_image")
    research_url = st.text_input(translate_text("Research Paper URL", selected_language), key="gripper_url")



    # Add Button
    if st.button(translate_text("Add Gripper", selected_language), key="add_gripper"):
        if name:
            image_path = None
            if image:
                # Save uploaded image
                image_path = save_uploaded_image(image, name)

                # New entry for grippers
            new_entry = {
                "Name": name,
                "Cost": cost_weight,  # Calculated cost weight
                "ISO Compliance": iso_weight,
                "Safety": safety_weight,
                "Performance": performance_weight,
                "Image": image_path,
                "Research URL": research_url
            }
            grippers = add_entry(GRIPPER_FILE, grippers, new_entry)
            st.success(translate_text(f"Gripper '{name}' added successfully!", selected_language))

    # Delete Gripper Section
    st.subheader(translate_text("Delete Gripper", selected_language))
    gripper_names = grippers['Name'].tolist()
    gripper_to_delete = st.selectbox(translate_text("Select Gripper to Delete", selected_language), gripper_names,
                                     key="delete_gripper_selectbox")
    if st.button(translate_text("Delete Gripper", selected_language), key="delete_gripper_button"):
        if gripper_to_delete:
            grippers = grippers[grippers['Name'] != gripper_to_delete]
            grippers.to_csv(GRIPPER_FILE, index=False)
            st.success(translate_text(f"Gripper '{gripper_to_delete}' deleted successfully!", selected_language))

    # Rename Gripper Section
    st.subheader(translate_text("Rename Gripper", selected_language))
    gripper_to_rename = st.selectbox(translate_text("Select Gripper to Rename", selected_language), gripper_names,
                                     key="rename_gripper_selectbox")
    new_gripper_name = st.text_input(translate_text("Enter New Name for the Gripper", selected_language),
                                     key="rename_gripper_textinput")
    if st.button(translate_text("Rename Gripper", selected_language), key="rename_gripper_button"):
        if new_gripper_name and gripper_to_rename:
            grippers.loc[grippers['Name'] == gripper_to_rename, 'Name'] = new_gripper_name
            grippers.to_csv(GRIPPER_FILE, index=False)
            st.success(translate_text(f"Gripper '{gripper_to_rename}' renamed to '{new_gripper_name}' successfully!",
                                      selected_language))

    # Display Gripper List with Images and URLs
    st.subheader(translate_text("Gripper List", selected_language))
    for _, row in grippers.iterrows():
        st.write(f"**{row['Name']}**")
        if 'Image' in row and pd.notna(row['Image']):  # Check if Image column exists and is not empty
            st.image(row['Image'], caption=row['Name'], use_container_width=True)
        if 'Research URL' in row and pd.notna(row['Research URL']):  # Check if Research URL exists and is not empty
            st.write(f"[Research Paper]({row['Research URL']})")
        st.write("---")

# Tactile Sensors tab
# Tab for Tactile Sensors
# Tactile Sensors Tab
with tab2:
    st.header(translate_text("Tactile Sensors", selected_language))

    # Slider for adjusting weights dynamically
    tactile_weights = weight_sliders("tactile")
    tactile_sensors = calculate_scores(tactile_sensors, tactile_weights)
    st.write(tactile_sensors[['Name', 'Total Score', 'Rank']])  # Display updated tactile sensor table
    plot_visualizations(tactile_sensors, translate_text("Tactile Sensors Analysis", selected_language))  # Visualization

    # Add New Tactile Sensor Section
    st.subheader(translate_text("Add New Tactile Sensor", selected_language))
    name = st.text_input(translate_text("Name", selected_language), key="add_tactile_name")

    # ISO-related sub-parameters
    st.write(translate_text("Enter values for ISO-related parameters (1-10):", selected_language))
    sensitivity = st.number_input(translate_text("Sensitivity", selected_language), min_value=1, max_value=10, value=5,
                                  key="tactile_sensitivity_input")
    durability = st.number_input(translate_text("Durability", selected_language), min_value=1, max_value=10, value=5,
                                 key="tactile_durability_input")
    ease_of_integration = st.number_input(translate_text("Ease of Integration", selected_language), min_value=1,
                                          max_value=10, value=5, key="tactile_integration_input")


    # Calculate overall ISO Compliance weight dynamically
    def calculate_iso_weight(sensitivity, durability, integration):
        weights = {"Sensitivity": 0.5, "Durability": 0.3, "Ease of Integration": 0.2}
        return round(sensitivity * weights["Sensitivity"] + durability * weights["Durability"] + integration * weights[
            "Ease of Integration"], 2)


    iso_weight = calculate_iso_weight(sensitivity, durability, ease_of_integration)

    # Image and Research Paper URL inputs
    image = st.file_uploader(translate_text("Upload Image", selected_language), type=['jpg', 'jpeg', 'png'],
                             key="tactile_image_input")
    research_url = st.text_input(translate_text("Research Paper URL", selected_language), key="tactile_url_input")

    # Main factors
    cost = st.number_input(translate_text("Cost", selected_language), min_value=1, max_value=10, value=5,
                           key="tactile_cost_input")
    safety = st.number_input(translate_text("Safety", selected_language), min_value=1, max_value=10, value=8,
                             key="tactile_safety_input")
    performance = st.number_input(translate_text("Performance", selected_language), min_value=1, max_value=10, value=7,
                                  key="tactile_performance_input")

    # Add Button
    if st.button(translate_text("Add Tactile Sensor", selected_language), key="add_tactile_button"):
        if name:
            image_path = None
            if image:
                # Save uploaded image
                image_path =  save_uploaded_image(image, name)

            # New entry for tactile sensors
            new_entry = {
                "Name": name,
                "Cost": cost,
                "ISO Compliance": iso_weight,  # Calculated ISO weight
                "Safety": safety,
                "Performance": performance,
                "Image": image_path,
                "Research URL": research_url
            }
            tactile_sensors = add_entry(TACTILE_FILE, tactile_sensors, new_entry)
            st.success(translate_text(f"Tactile Sensor '{name}' added successfully!", selected_language))

    # Delete Tactile Sensor Section
    st.subheader(translate_text("Delete Tactile Sensor", selected_language))
    tactile_names = tactile_sensors['Name'].tolist()
    tactile_to_delete = st.selectbox(translate_text("Select Tactile Sensor to Delete", selected_language),
                                     tactile_names, key="delete_tactile_selectbox")
    if st.button(translate_text("Delete Tactile Sensor", selected_language), key="delete_tactile_button"):
        if tactile_to_delete:
            tactile_sensors = tactile_sensors[tactile_sensors['Name'] != tactile_to_delete]
            tactile_sensors.to_csv(TACTILE_FILE, index=False)
            st.success(translate_text(f"Tactile Sensor '{tactile_to_delete}' deleted successfully!", selected_language))

    # Rename Tactile Sensor Section
    st.subheader(translate_text("Rename Tactile Sensor", selected_language))
    tactile_to_rename = st.selectbox(translate_text("Select Tactile Sensor to Rename", selected_language),
                                     tactile_names, key="rename_tactile_selectbox")
    new_tactile_name = st.text_input(translate_text("Enter New Name for the Tactile Sensor", selected_language),
                                     key="rename_tactile_textinput")
    if st.button(translate_text("Rename Tactile Sensor", selected_language), key="rename_tactile_button"):
        if new_tactile_name and tactile_to_rename:
            tactile_sensors.loc[tactile_sensors['Name'] == tactile_to_rename, 'Name'] = new_tactile_name
            tactile_sensors.to_csv(TACTILE_FILE, index=False)
            st.success(
                translate_text(f"Tactile Sensor '{tactile_to_rename}' renamed to '{new_tactile_name}' successfully!",
                               selected_language))

    # Display Tactile Sensors with Images and URLs
    st.subheader(translate_text("Tactile Sensor List", selected_language))
    for _, row in tactile_sensors.iterrows():
        st.write(f"**{row['Name']}**")
        if 'Image' in row and pd.notna(row['Image']):  # Check if Image column exists and is not empty
            st.image(row['Image'], caption=row['Name'], use_container_width=True)
        if 'Research URL' in row and pd.notna(row['Research URL']):  # Check if Research URL exists and is not empty
            st.write(f"[Research Paper]({row['Research URL']})")
        st.write("---")

# Proximity Sensors tab
# Tab for Proximity Sensors
with tab3:
    st.header(translate_text("Proximity Sensors", selected_language))

    # Slider for adjusting weights dynamically
    proximity_weights = weight_sliders("proximity")
    proximity_sensors = calculate_scores(proximity_sensors, proximity_weights)
    st.write(proximity_sensors[['Name', 'Total Score', 'Rank']])  # Display updated proximity sensor table
    plot_visualizations(proximity_sensors,
                        translate_text("Proximity Sensors Analysis", selected_language))  # Visualization

    # Add New Proximity Sensor Section
    st.subheader(translate_text("Add New Proximity Sensor", selected_language))
    name = st.text_input(translate_text("Name", selected_language), key="add_proximity_name")

    # Performance-related sub-parameters
    st.write(translate_text("Enter values for performance-related parameters (1-10):", selected_language))
    range = st.number_input(translate_text("Range (Detection Distance)", selected_language), min_value=1, max_value=10,
                            value=5, key="proximity_range")
    accuracy = st.number_input(translate_text("Accuracy", selected_language), min_value=1, max_value=10, value=5,
                               key="proximity_accuracy")
    response_time = st.number_input(translate_text("Response Time", selected_language), min_value=1, max_value=10,
                                    value=5, key="proximity_response_time")


    # Calculate overall Performance weight dynamically
    def calculate_performance_weight(range, accuracy, response_time):
        weights = {"Range": 0.4, "Accuracy": 0.4, "Response Time": 0.2}
        return round(
            range * weights["Range"] + accuracy * weights["Accuracy"] + response_time * weights["Response Time"], 2)


    performance_weight = calculate_performance_weight(range, accuracy, response_time)

    # Image and Research Paper URL inputs
    image = st.file_uploader(translate_text("Upload Image", selected_language), type=['jpg', 'jpeg', 'png'],
                             key="proximity_image")
    research_url = st.text_input(translate_text("Research Paper URL", selected_language), key="proximity_url")

    # Main factors
    cost = st.number_input(translate_text("Cost", selected_language), min_value=1, max_value=10, value=5,
                           key="proximity_cost_input")
    iso = st.number_input(translate_text("ISO Compliance", selected_language), min_value=1, max_value=10, value=7,
                          key="proximity_iso_input")
    safety = st.number_input(translate_text("Safety", selected_language), min_value=1, max_value=10, value=8,
                             key="proximity_safety_input")

    # Add Button
    if st.button(translate_text("Add Proximity Sensor", selected_language), key="add_proximity"):
        if name:
            image_path = None
            if image:
                # Save uploaded image
                image_path = save_uploaded_image(image, name)

            # New entry for proximity sensors
            new_entry = {
                "Name": name,
                "Cost": cost,
                "ISO Compliance": iso,
                "Safety": safety,
                "Performance": performance_weight,  # Calculated Performance weight
                "Image": image_path,
                "Research URL": research_url
            }
            proximity_sensors = add_entry(PROXIMITY_FILE, proximity_sensors, new_entry)
            st.success(translate_text(f"Proximity Sensor '{name}' added successfully!", selected_language))

    # Delete Proximity Sensor Section
    st.subheader(translate_text("Delete Proximity Sensor", selected_language))
    proximity_names = proximity_sensors['Name'].tolist()
    proximity_to_delete = st.selectbox(translate_text("Select Proximity Sensor to Delete", selected_language),
                                       proximity_names, key="delete_proximity_selectbox")
    if st.button(translate_text("Delete Proximity Sensor", selected_language), key="delete_proximity_button"):
        if proximity_to_delete:
            proximity_sensors = proximity_sensors[proximity_sensors['Name'] != proximity_to_delete]
            proximity_sensors.to_csv(PROXIMITY_FILE, index=False)
            st.success(
                translate_text(f"Proximity Sensor '{proximity_to_delete}' deleted successfully!", selected_language))

    # Rename Proximity Sensor Section
    st.subheader(translate_text("Rename Proximity Sensor", selected_language))
    proximity_to_rename = st.selectbox(translate_text("Select Proximity Sensor to Rename", selected_language),
                                       proximity_names, key="rename_proximity_selectbox")
    new_proximity_name = st.text_input(translate_text("Enter New Name for the Proximity Sensor", selected_language),
                                       key="rename_proximity_textinput")
    if st.button(translate_text("Rename Proximity Sensor", selected_language), key="rename_proximity_button"):
        if new_proximity_name and proximity_to_rename:
            proximity_sensors.loc[proximity_sensors['Name'] == proximity_to_rename, 'Name'] = new_proximity_name
            proximity_sensors.to_csv(PROXIMITY_FILE, index=False)
            st.success(translate_text(
                f"Proximity Sensor '{proximity_to_rename}' renamed to '{new_proximity_name}' successfully!",
                selected_language))

    # Display Proximity Sensors with Images and URLs
    st.subheader(translate_text("Proximity Sensor List", selected_language))
    for _, row in proximity_sensors.iterrows():
        st.write(f"**{row['Name']}**")
        if 'Image' in row and pd.notna(row['Image']):  # Check if Image column exists and is not empty
            st.image(row['Image'], caption=row['Name'], use_container_width=True)
        if 'Research URL' in row and pd.notna(row['Research URL']):  # Check if Research URL exists and is not empty
            st.write(f"[Research Paper]({row['Research URL']})")
        st.write("---")

