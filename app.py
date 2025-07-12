import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import io
import os
import tensorflow as tf
import numpy as np
import uuid

# --- FUNCTIONS: Load Models and Draw Predictions ---

@st.cache_resource
def load_all_models(classifier_path, roboflow_api_key):
    """
    Loads the local Keras classifier from the repository and initializes the Roboflow client.
    This function is cached to run only once when the app starts.
    """
    # 1. Load Keras Classifier Model from local path
    try:
        # Check if the file exists in the repository
        if not os.path.exists(classifier_path):
            st.error(f"Classifier model file not found at path: '{classifier_path}'. Make sure it's in your GitHub repo.")
            return None, None
            
        # Load the model
        classifier_model = tf.keras.models.load_model(classifier_path)
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None, None
    
    # 2. Initialize Roboflow Detection Client
    detector_client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=roboflow_api_key
    )
    return classifier_model, detector_client

def draw_predictions(image, predictions):
    """Draws bounding boxes and labels on an image based on Roboflow predictions."""
    draw = ImageDraw.Draw(image)
    detection_count = 0
    if isinstance(predictions, list) and len(predictions) > 0:
        # Access the correct prediction structure from Roboflow Workflow API
        inner_predictions = predictions[0].get('predictions', {}).get('predictions', [])
        detection_count = len(inner_predictions)
        
        for pred in inner_predictions:
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1, x2, y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2
            
            label = f"{pred['class']} ({pred['confidence']:.0%})"
            color = "red"
            
            try:
                font = ImageFont.load_default(size=15)
            except IOError:
                font = None

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Draw a text background for better visibility
            text_bbox = draw.textbbox((0,0), label, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            text_width = text_bbox[2] - text_bbox[0]
            text_bg_rect = [x1, y1 - text_height - 5, x1 + text_width + 4, y1 - 5]
            draw.rectangle(text_bg_rect, fill=color)
            draw.text((x1 + 2, y1 - text_height - 3), label, fill="white", font=font)
            
    return image, detection_count

# ==============================================================================
# --- MAIN STREAMLIT APP UI & LOGIC ---
# ==============================================================================

st.set_page_config(layout="wide", page_title="Smart Car Damage Assessment")
st.title("🚗 Smart Car Damage Assessment")
st.markdown("An end-to-end AI pipeline to classify and locate vehicle damage.")

# --- Configuration for Streamlit Cloud ---
# The model file path is relative to the root of the GitHub repository
CLASSIFIER_FILENAME = "damage_classifier.keras" 

# Use st.secrets to securely access keys and IDs from Streamlit Cloud settings
try:
    ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKFLOW_ID = st.secrets["ROBOFLOW_WORKFLOW_ID"]
    ROBOFLOW_WORKSPACE = st.secrets["ROBOFLOW_WORKSPACE"]
except (KeyError, FileNotFoundError):
    st.error("Roboflow secrets are not set. Please configure them in your Streamlit Cloud app settings.")
    st.stop()


# --- Load all models at startup ---
classifier_model, detector_client = load_all_models(CLASSIFIER_FILENAME, ROBOFLOW_API_KEY)

if classifier_model is None or detector_client is None:
    st.error("A critical component failed to load. The application cannot proceed.")
    st.stop()

# --- Main App Workflow ---
uploaded_file = st.file_uploader("Upload an image of a car to begin analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # Create a centered layout for the initial view
    col1, col2, col3 = st.columns([1, 1.5, 1]) 
    with col2:
        st.image(original_image, caption="Your Uploaded Image", use_container_width=True)
        analyze_button = st.button("🔍 Analyze Image", use_container_width=True)

    if analyze_button:
        st.markdown("---") # Separator line
        col_res1, col_res2 = st.columns(2) # New columns for results
        
        with col_res1:
            st.subheader("Analysis Results")
            # --- Stage 1: Classification ---
            with st.spinner("Stage 1: Performing initial damage classification..."):
                img_for_classifier = original_image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_for_classifier)
                img_array_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_batch = np.expand_dims(img_array_preprocessed, 0)
                
                prediction = classifier_model.predict(img_batch)
                score = prediction[0][0]
                is_damaged_by_classifier = score >= 0.5
            
            # --- Stage 2: Localization (if needed) & Final Verdict Logic ---
            if is_damaged_by_classifier:
                with st.spinner("Stage 2: Classifier detected potential damage. Locating specifics..."):
                    # Save the uploaded image to a temporary in-memory buffer
                    buffered = io.BytesIO()
                    original_image.save(buffered, format="JPEG")
                    img_bytes = buffered.getvalue() # Get the bytes of the image
                    
                    try:
                        # Call Roboflow API using image bytes
                        result = detector_client.run_workflow(
                            workspace_name=ROBOFLOW_WORKSPACE,
                            workflow_id=ROBOFLOW_WORKFLOW_ID,
                            images={"image": img_bytes} # Sending bytes directly
                        )
                        image_with_boxes, detection_count = draw_predictions(original_image.copy(), result)
                        
                        if detection_count > 0:
                            verdict_html = f'<div style="background-color:#FFF3CD; color: #31333F; padding:10px; border-radius:5px;">⚠️ <b>FINAL VERDICT: DAMAGED</b><br>The system identified {detection_count} specific point(s) of damage.</div>'
                            st.markdown(verdict_html, unsafe_allow_html=True)
                        else:
                            verdict_html = '<div style="background-color:#D4EDDA; color: #31333F; padding:10px; border-radius:5px;">✅ <b>FINAL VERDICT: LIKELY WHOLE</b><br>Initial analysis suggested potential damage, but a detailed inspection found no specific damage points.</div>'
                            st.markdown(verdict_html, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error during damage localization: {e}")

            else:
                verdict_html = f'<div style="background-color:#D4EDDA; color: #31333F; padding:10px; border-radius:5px;">✅ <b>FINAL VERDICT: WHOLE</b><br>The classification model is confident the vehicle is undamaged (Confidence: {1-score:.2%}).</div>'
                st.markdown(verdict_html, unsafe_allow_html=True)
                st.balloons()
        
        with col_res2:
            st.subheader("Final Visual Result")
            # Display the final image result in the second column
            if 'image_with_boxes' in locals() and detection_count > 0:
                st.image(image_with_boxes, caption="Detected Damages", use_container_width=True)
            else:
                st.image(original_image, caption="No specific damages were localized.", use_container_width=True)

            # Display raw JSON only if detection was run
            if is_damaged_by_classifier and 'result' in locals():
                 with st.expander("Show Raw Detection Data (JSON)"):
                    st.json(result)
