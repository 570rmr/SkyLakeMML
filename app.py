import torch
import streamlit as st
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class UniversalMMLModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using CLIP for vision
        self.vision_encoder = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.vision_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Using BERT instead of BART for text (more compatible)
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Note: Audio processing removed for simplicity (wav2vec2 requires audio files)
        # You can add it back if you have audio processing needs
        
        # Adjusted dimensions to match actual model outputs
        # CLIP vision: 768, BERT: 768
        self.fusion_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
            num_layers=2
        )
        
        self.multi_task_head = nn.Sequential(
            nn.Linear(768 * 2, 1024),  # 2 modalities now (vision + text)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.classification_head = nn.Linear(512, 10)  # Reduced classes for demo
        self.regression_head = nn.Linear(512, 1)
        
    def forward(self, text_input, image_input):
        # Process text
        with torch.no_grad():
            text_features = self.text_encoder(**text_input).last_hidden_state
            # Use CLS token
            text_features = text_features[:, 0:1, :]  # Shape: [batch, 1, 768]
        
        # Process image
        with torch.no_grad():
            image_features = self.vision_encoder.get_image_features(**image_input)
            image_features = image_features.unsqueeze(1)  # Shape: [batch, 1, 512]
            # Project to 768 dimensions
            image_features = F.pad(image_features, (0, 256))  # Pad to 768
        
        # Combine features
        combined_features = torch.cat([text_features, image_features], dim=1)
        
        # Fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Flatten for task heads
        fused_features = fused_features.reshape(fused_features.size(0), -1)
        
        # Multi-task output
        multi_task_output = self.multi_task_head(fused_features)
        
        classification_output = self.classification_head(multi_task_output)
        regression_output = self.regression_head(multi_task_output)
        
        return classification_output, regression_output

class SmartMMLApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    @st.cache_resource
    def load_model(_self):
        """Load model with caching"""
        model = UniversalMMLModel()
        model.to(_self.device)
        model.eval()
        return model
        
    def process_multimodal_input(self, text, image):
        if self.model is None:
            self.model = self.load_model()
            
        # Preprocess inputs
        processed_text = self._preprocess_text(text)
        processed_image = self._preprocess_image(image)
        
        # Multi-modal inference
        with torch.no_grad():
            classification, regression = self.model(processed_text, processed_image)
        
        return {
            'classification_logits': classification.cpu().numpy(),
            'classification_prediction': torch.argmax(classification, dim=1).item(),
            'regression_value': regression.item()
        }
    
    def _preprocess_text(self, text):
        """Preprocess text input"""
        if not text:
            text = "empty"
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        encoded = tokenizer(
            text, 
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded
    
    def _preprocess_image(self, image):
        """Preprocess image input"""
        if image is None:
            # Create blank image if none provided
            image = Image.new('RGB', (224, 224), color='white')
        
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        processed = processor(images=image, return_tensors='pt')
        
        # Move to device
        processed = {k: v.to(self.device) for k, v in processed.items()}
        return processed

def main():
    st.set_page_config(page_title="Multi-Modal ML Platform", page_icon="🤖", layout="wide")
    
    st.title("🤖 Universal Multi-Modal Learning Platform")
    st.markdown("---")
    
    st.markdown("""
    This demo shows a multi-modal model that processes both **text** and **images**.
    
    **Note:** This is a demonstration model with randomly initialized weights. 
    In production, you would train this on your specific dataset.
    """)
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Input")
        text_input = st.text_area(
            "Enter text to analyze",
            placeholder="Type something here...",
            height=150
        )
    
    with col2:
        st.subheader("🖼️ Image Input")
        image_upload = st.file_uploader(
            "Upload an image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a PNG or JPG image"
        )
        
        if image_upload:
            image = Image.open(image_upload)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.info("No image uploaded - will use blank image")
    
    st.markdown("---")
    
    # Process button
    if st.button("🚀 Process Multi-Modal Input", type="primary", use_container_width=True):
        if not text_input and not image_upload:
            st.warning("⚠️ Please provide at least text or image input!")
        else:
            with st.spinner("Processing multi-modal input..."):
                try:
                    app = SmartMMLApp()
                    
                    # Process the inputs
                    image = Image.open(image_upload) if image_upload else None
                    result = app.process_multimodal_input(text_input, image)
                    
                    # Display results
                    st.success("✅ Processing complete!")
                    
                    st.subheader("📊 Results")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric(
                            "Classification Prediction", 
                            f"Class {result['classification_prediction']}"
                        )
                    
                    with col_b:
                        st.metric(
                            "Regression Value", 
                            f"{result['regression_value']:.4f}"
                        )
                    
                    # Show raw outputs in expander
                    with st.expander("🔍 View Raw Model Outputs"):
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"❌ Error processing input: {str(e)}")
                    st.exception(e)
    
    # Sidebar info
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.info("""
        This is a demonstration of a multi-modal machine learning model that can process:
        - Text (using BERT)
        - Images (using CLIP)
        
        The model outputs:
        - **Classification**: Predicts one of 10 classes
        - **Regression**: Predicts a continuous value
        """)
        
        st.subheader("⚙️ Model Info")
        st.write(f"**Device:** {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        st.write("**Text Encoder:** BERT-base-uncased")
        st.write("**Vision Encoder:** CLIP ViT-B/32")
        
        st.markdown("---")
        st.caption("Built with Streamlit & PyTorch")

if __name__ == "__main__":
    main()
