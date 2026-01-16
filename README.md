# Generative-models-dcgan

## 🎯 Scope
DCGAN (Deep Convolutional Generative Adversarial Network) for image generation using the MNIST dataset.

---

## 💼 Business Use Case

Generative models like DCGAN are widely used for:
- **Data augmentation** for imbalanced datasets
- **Synthetic sample generation** in privacy-constrained environments
- **AI creativity tools** like art and digit generation
- Prototyping for **style transfer** and image-to-image translation tasks

This project demonstrates how DCGAN can be trained on handwritten digits to produce synthetic but realistic outputs.

---

## 📁 Project Structure

```
generative-models-dcgan/
├── notebooks/                 # Notebook to launch training
├── src/
│   ├── models/                # Generator and Discriminator
│   ├── train.py               # Training script
├── streamlit_app/
│   └── app.py                 # Interactive image generator
├── output/                    # Trained models and sample outputs
├── demo/
│   └── dcgan_training.gif     # Visualization of output progression
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m src.train
```

This will:
- Train DCGAN on MNIST
- Save model checkpoints and image samples to `output/`

---

## 🌐 Launch Streamlit App

```bash
streamlit run streamlit_app/app.py
```

- Use the slider to choose how many digits to generate
- Click **Generate** to view model output in real time

---

## 🎥 Training Progress Demo

![DCGAN Training](demo/dcgan_training.gif)

---

## 📌 Output

- ✅ Generator and Discriminator models: `output/generator.pth`
- ✅ Sample images saved every epoch: `output/samples/epoch_*.png`

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## Let’s Connect!

* 💼 [LinkedIn](https://www.linkedin.com/in/amitkharche)
* 📝 [Medium](https://medium.com/@amitkharche)
* 💻 [GitHub](https://github.com/amitkharche)

---
