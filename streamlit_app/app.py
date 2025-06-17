import sys
import os
import io
import zipfile
from datetime import datetime

# Adds the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure page
st.set_page_config(
    page_title="DCGAN Image Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .download-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the DCGAN generator model"""
    try:
        from src.models.generator import Generator
        model = Generator()
        model.load_state_dict(torch.load("output/generator.pth", map_location="cpu"))
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

def generate_images(model, num_images, seed=None):
    """Generate images using the DCGAN model"""
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn(num_images, 100)
    with torch.no_grad():
        generated = model(noise).view(num_images, 28, 28)
    
    # Convert to numpy and normalize to [0, 255]
    images = generated.cpu().numpy()
    images = ((images + 1) / 2 * 255).astype(np.uint8)
    
    return images

def create_image_grid(images, grid_size=None):
    """Create a grid layout for displaying images"""
    num_images = len(images)
    
    if grid_size is None:
        # Auto-calculate grid size
        if num_images <= 4:
            cols = num_images
            rows = 1
        elif num_images <= 16:
            cols = 4
            rows = (num_images + 3) // 4
        else:
            cols = 8
            rows = (num_images + 7) // 8
    else:
        cols, rows = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.patch.set_facecolor('white')
    
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(rows * cols):
        if i < num_images:
            axes[i].imshow(images[i], cmap='gray')
            axes[i].set_title(f'Image {i+1}', fontsize=10, pad=5)
        else:
            axes[i].set_visible(False)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def images_to_zip(images):
    """Convert images to a ZIP file for download"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_array in enumerate(images):
            # Convert numpy array to PIL Image
            img = Image.fromarray(img_array, mode='L')
            
            # Save image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Add to zip
            zip_file.writestr(f'generated_image_{i+1:03d}.png', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def save_individual_image(img_array, filename):
    """Save individual image as PNG"""
    img = Image.fromarray(img_array, mode='L')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return img_buffer.getvalue()

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 DCGAN Image Generator</h1>
        <p>Generate high-quality MNIST-style digits using Deep Convolutional GANs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"❌ Error loading model: {error}")
        st.info("Please ensure the model file 'output/generator.pth' exists and the Generator class is properly defined.")
        return
    
    if model is None:
        st.error("❌ Failed to load the model. Please check your model file and dependencies.")
        return
    
    st.success("✅ DCGAN Generator model loaded successfully!")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Generation Controls")
        
        # Number of images
        num_images = st.slider(
            "Number of Images",
            min_value=1,
            max_value=64,
            value=16,
            help="Select how many images to generate"
        )
        
        # Seed for reproducibility
        use_seed = st.checkbox("Use Random Seed", help="Enable for reproducible results")
        seed = None
        if use_seed:
            seed = st.number_input("Seed Value", min_value=0, max_value=9999, value=42)
        
        # Grid layout options
        st.subheader("📐 Display Options")
        auto_grid = st.checkbox("Auto Grid Layout", value=True)
        
        if not auto_grid:
            col1, col2 = st.columns(2)
            with col1:
                grid_cols = st.number_input("Columns", min_value=1, max_value=10, value=4)
            with col2:
                grid_rows = st.number_input("Rows", min_value=1, max_value=10, value=4)
            grid_size = (grid_cols, grid_rows)
        else:
            grid_size = None
    
    # Generation section
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        generate_btn = st.button("🎨 Generate Images", use_container_width=True)
    
    if generate_btn:
        with st.spinner("🎨 Generating images..."):
            try:
                # Generate images
                images = generate_images(model, num_images, seed)
                
                # Store in session state
                st.session_state.generated_images = images
                st.session_state.generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            except Exception as e:
                st.error(f"❌ Error generating images: {str(e)}")
                return
    
    # Display generated images
    if 'generated_images' in st.session_state:
        st.header("🖼️ Generated Images")
        
        # Create and display image grid
        fig = create_image_grid(st.session_state.generated_images, grid_size)
        st.pyplot(fig)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Images</h3>
                <h2>{}</h2>
            </div>
            """.format(len(st.session_state.generated_images)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📏 Resolution</h3>
                <h2>28×28</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎨 Format</h3>
                <h2>Grayscale</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>⏰ Generated</h3>
                <h2>{}</h2>
            </div>
            """.format(st.session_state.generation_time.split()[1]), unsafe_allow_html=True)
        
        # Download section
        st.markdown("""
        <div class="download-section">
            <h3>💾 Download Options</h3>
            <p>Choose your preferred download format:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download all as ZIP
            zip_data = images_to_zip(st.session_state.generated_images)
            st.download_button(
                label="📦 Download All as ZIP",
                data=zip_data,
                file_name=f"dcgan_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        with col2:
            # Download matplotlib figure
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            st.download_button(
                label="🖼️ Download Grid as PNG",
                data=img_buffer.getvalue(),
                file_name=f"dcgan_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        # Individual image downloads
        if len(st.session_state.generated_images) <= 16:  # Only show for reasonable number of images
            st.subheader("📄 Individual Downloads")
            
            cols = st.columns(min(4, len(st.session_state.generated_images)))
            
            for i, img_array in enumerate(st.session_state.generated_images):
                with cols[i % 4]:
                    img_data = save_individual_image(img_array, f"image_{i+1}")
                    st.download_button(
                        label=f"Image {i+1}",
                        data=img_data,
                        file_name=f"dcgan_image_{i+1:03d}.png",
                        mime="image/png",
                        key=f"download_{i}",
                        use_container_width=True
                    )
        
        # Additional info
        with st.expander("ℹ️ About This Generation"):
            st.write(f"**Generation Time:** {st.session_state.generation_time}")
            st.write(f"**Number of Images:** {len(st.session_state.generated_images)}")
            st.write(f"**Seed Used:** {seed if use_seed else 'Random'}")
            st.write(f"**Model:** DCGAN Generator")
            st.write(f"**Dataset:** MNIST-style digits")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🧠 Powered by DCGAN | Built with Streamlit | Generate amazing digit images with AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()