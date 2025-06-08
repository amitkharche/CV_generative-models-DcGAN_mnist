import streamlit as st
import torch
from src.models.generator import Generator
import matplotlib.pyplot as plt

st.title("🧠 DCGAN Image Generator - MNIST")
model = Generator()
model.load_state_dict(torch.load("output/generator.pth", map_location="cpu"))
model.eval()

num_images = st.slider("Number of Images", 1, 64, 25)
if st.button("Generate"):
    noise = torch.randn(num_images, 100)
    with torch.no_grad():
        generated = model(noise).view(num_images, 28, 28)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i, ax in enumerate(axes):
        ax.imshow(generated[i], cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
