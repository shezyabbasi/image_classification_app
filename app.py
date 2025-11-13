import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import json
import csv
from datetime import datetime
import base64

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .upload-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .result-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 1rem 0;
        text-align: center;
    }
    .dog-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 2rem;
        font-weight: bold;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(245,87,108,0.4);
    }
    .cat-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 2rem;
        font-weight: bold;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(79,172,254,0.4);
    }
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 30px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        transition: width 0.5s ease;
    }
    .title-container {
        text-align: center;
        color: white;
        padding: 2rem 0;
    }
    h1 {
        color: white !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .history-item {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="title-container">
        <h1>🐾 Cat vs Dog Classifier</h1>
        <p class="subtitle">Upload images and let AI identify if they're cats or dogs!</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.0

# Example images
EXAMPLE_IMAGES = {
    "Cat 1": "attached_assets/stock_images/cute_tabby_cat_portr_16f778e9.jpg",
    "Cat 2": "attached_assets/stock_images/cute_tabby_cat_portr_76703c22.jpg",
    "Dog 1": "attached_assets/stock_images/golden_retriever_dog_dd7bb9db.jpg",
    "Dog 2": "attached_assets/stock_images/golden_retriever_dog_f78bd7da.jpg"
}

@st.cache_resource
def load_model():
    """Load and compile the pre-trained model"""
    imagenet_model = keras.applications.MobileNetV2(weights='imagenet')
    return imagenet_model

def apply_preprocessing(image, brightness=1.0, contrast=1.0, sharpness=1.0, blur=False):
    """Apply preprocessing filters to the image"""
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Sharpness
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    # Blur
    if blur:
        image = image.filter(ImageFilter.BLUR)
    
    return image

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def predict_cat_or_dog(image, model):
    """Predict if the image is a cat or dog"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(predictions, top=20)[0]
    
    cat_synsets = ['n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
                   'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165',
                   'n02129604', 'n02130308']
    
    cat_confidence = 0.0
    dog_confidence = 0.0
    
    for class_name, description, confidence in decoded_predictions:
        if class_name in cat_synsets:
            cat_confidence += confidence
        elif class_name.startswith('n0208') or class_name.startswith('n0209') or \
             class_name.startswith('n0210') or class_name.startswith('n0211'):
            dog_confidence += confidence
        else:
            desc_lower = description.lower()
            if any(keyword in desc_lower for keyword in ['cat', 'lynx', 'leopard', 'cheetah', 
                                                          'jaguar', 'lion', 'tiger']):
                cat_confidence += confidence
            elif any(keyword in desc_lower for keyword in ['dog', 'puppy', 'hound', 'terrier',
                                                           'retriever', 'shepherd', 'spaniel',
                                                           'bulldog', 'poodle', 'collie', 'husky',
                                                           'pug', 'beagle', 'corgi', 'dachshund',
                                                           'chihuahua', 'pomeranian', 'rottweiler',
                                                           'dalmatian', 'boxer', 'schnauzer',
                                                           'setter', 'pointer', 'mastiff', 'akita',
                                                           'shiba', 'malamute', 'samoyed']):
                dog_confidence += confidence
    
    total = cat_confidence + dog_confidence
    is_low_confidence = total <= 0.05
    
    raw_cat_confidence = cat_confidence
    raw_dog_confidence = dog_confidence
    
    if dog_confidence > cat_confidence:
        prediction = "Dog"
        confidence = raw_dog_confidence
    elif cat_confidence > dog_confidence:
        prediction = "Cat"
        confidence = raw_cat_confidence
    else:
        prediction = "Cat"
        confidence = raw_cat_confidence
    
    if total > 0:
        normalized_cat = cat_confidence / total
        normalized_dog = dog_confidence / total
    else:
        normalized_cat = 0.5
        normalized_dog = 0.5
    
    return prediction, confidence, normalized_cat, normalized_dog, is_low_confidence

def image_to_base64(image):
    """Convert PIL Image to base64 string for storage"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def add_to_history(image, prediction, confidence, cat_conf, dog_conf, is_low_conf, filename=""):
    """Add a prediction to the history"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'prediction': prediction,
        'confidence': float(confidence),
        'cat_confidence': float(cat_conf),
        'dog_confidence': float(dog_conf),
        'is_low_confidence': bool(is_low_conf),
        'image_thumbnail': image_to_base64(image.resize((100, 100)))
    }
    st.session_state.prediction_history.append(history_entry)

def export_to_csv():
    """Export prediction history to CSV format"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Cat Confidence', 'Dog Confidence', 'Low Confidence'])
    
    for entry in st.session_state.prediction_history:
        writer.writerow([
            entry['timestamp'],
            entry['filename'],
            entry['prediction'],
            f"{entry['confidence']*100:.2f}%",
            f"{entry['cat_confidence']*100:.2f}%",
            f"{entry['dog_confidence']*100:.2f}%",
            'Yes' if entry['is_low_confidence'] else 'No'
        ])
    
    return output.getvalue()

def export_to_json():
    """Export prediction history to JSON format"""
    export_data = []
    for entry in st.session_state.prediction_history:
        export_data.append({
            'timestamp': entry['timestamp'],
            'filename': entry['filename'],
            'prediction': entry['prediction'],
            'confidence': entry['confidence'],
            'cat_confidence': entry['cat_confidence'],
            'dog_confidence': entry['dog_confidence'],
            'is_low_confidence': entry['is_low_confidence']
        })
    return json.dumps(export_data, indent=2)

# Load model with spinner
with st.spinner('🔄 Loading AI model...'):
    model = load_model()
    st.session_state.model = model
    st.session_state.model_loaded = True

# Sidebar for settings and history
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Confidence threshold
    st.markdown("### Confidence Threshold")
    st.session_state.confidence_threshold = st.slider(
        "Minimum confidence to show predictions",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter predictions below this confidence level"
    )
    
    # Preprocessing options
    st.markdown("### 🎨 Image Preprocessing")
    enable_preprocessing = st.checkbox("Enable preprocessing", value=False)
    
    brightness = 1.0
    contrast = 1.0
    sharpness = 1.0
    apply_blur = False
    
    if enable_preprocessing:
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
        sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        apply_blur = st.checkbox("Apply Blur", value=False)
    
    st.markdown("---")
    
    # History section
    st.markdown("## 📊 Prediction History")
    
    if len(st.session_state.prediction_history) > 0:
        st.markdown(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
        
        # Export buttons
        st.markdown("### Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_to_csv()
            st.download_button(
                label="📥 CSV",
                data=csv_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = export_to_json()
            st.download_button(
                label="📥 JSON",
                data=json_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Recent Predictions")
        
        # Show last 5 predictions
        for entry in reversed(st.session_state.prediction_history[-5:]):
            with st.container():
                st.markdown(f"""
                <div class="history-item">
                    <small>{entry['timestamp']}</small><br>
                    <strong>{entry['prediction']}</strong> ({entry['confidence']*100:.1f}%)
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No predictions yet. Upload images to get started!")

# Main content
tab1, tab2 = st.tabs(["📤 Upload Images", "🖼️ Example Images"])

with tab1:
    st.markdown("### Upload Your Images")
    
    # Batch upload
    uploaded_files = st.file_uploader(
        "Choose one or more images (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="You can upload multiple images at once for batch processing"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        
        # Process all images
        if st.button("🚀 Classify All Images", type="primary"):
            progress_bar = st.progress(0)
            results_container = st.container()
            
            with results_container:
                cols = st.columns(min(3, len(uploaded_files)))
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    col_idx = idx % 3
                    
                    with cols[col_idx]:
                        image = Image.open(uploaded_file)
                        
                        # Apply preprocessing if enabled
                        if enable_preprocessing:
                            image = apply_preprocessing(image, brightness, contrast, sharpness, apply_blur)
                        
                        st.image(image, caption=uploaded_file.name, use_container_width=True)
                        
                        with st.spinner('Analyzing...'):
                            prediction, confidence, cat_conf, dog_conf, is_low_confidence = predict_cat_or_dog(image, model)
                        
                        # Check confidence threshold
                        if confidence >= st.session_state.confidence_threshold:
                            # Add to history
                            add_to_history(image, prediction, confidence, cat_conf, dog_conf, is_low_confidence, uploaded_file.name)
                            
                            # Display result
                            if prediction == "Dog":
                                st.markdown(f'<div class="dog-badge">🐶 {prediction}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="cat-badge">🐱 {prediction}</div>', unsafe_allow_html=True)
                            
                            st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                            
                            if is_low_confidence:
                                st.warning("⚠️ Low confidence")
                            
                            # Mini confidence bars
                            st.markdown(f"🐶 Dog: {dog_conf*100:.0f}%")
                            st.progress(float(dog_conf))
                            st.markdown(f"🐱 Cat: {cat_conf*100:.0f}%")
                            st.progress(float(cat_conf))
                        else:
                            st.error(f"Confidence {confidence*100:.1f}% below threshold {st.session_state.confidence_threshold*100:.0f}%")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"✅ Processed {len(uploaded_files)} images!")

with tab2:
    st.markdown("### Try Example Images")
    st.info("Click on an example image below to classify it instantly!")
    
    cols = st.columns(4)
    
    for idx, (name, path) in enumerate(EXAMPLE_IMAGES.items()):
        with cols[idx % 4]:
            try:
                example_image = Image.open(path)
                st.image(example_image, caption=name, use_container_width=True)
                
                if st.button(f"Classify {name}", key=f"example_{idx}"):
                    with st.spinner('Analyzing...'):
                        # Apply preprocessing if enabled
                        processed_example = example_image.copy()
                        if enable_preprocessing:
                            processed_example = apply_preprocessing(processed_example, brightness, contrast, sharpness, apply_blur)
                        
                        prediction, confidence, cat_conf, dog_conf, is_low_confidence = predict_cat_or_dog(processed_example, model)
                    
                    # Add to history
                    add_to_history(processed_example, prediction, confidence, cat_conf, dog_conf, is_low_confidence, name)
                    
                    # Display result
                    if prediction == "Dog":
                        st.markdown(f'<div class="dog-badge">🐶 {prediction}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="cat-badge">🐱 {prediction}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Confidence:** {confidence*100:.1f}%")
                    
                    if is_low_confidence:
                        st.warning("⚠️ Low confidence")
            except Exception as e:
                st.error(f"Could not load example image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: white; padding: 1rem;">
        <p>Powered by TensorFlow & MobileNetV2 | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
