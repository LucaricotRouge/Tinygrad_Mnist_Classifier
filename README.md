# TinyGrad MNIST Classifier with WebGPU 

## 🌐 Live Demo
[TinyGrad MNIST Classifier Demo](https://lucaricotrouge.github.io/Tinygrad_Mnist_Classifier/)

## Overview 
This project implements a handwritten digit recognition system using TinyGrad for model training and WebGPU for real-time inference in the browser. The application features both an MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Network) model trained on the MNIST dataset, allowing users to draw digits and receive instant predictions with confidence scores.

## ✨ Features
- Switch between MLP and CNN models in real-time
- Interactive drawing canvas with pen and eraser tools
- Real-time digit classification
- Probability visualization through dynamic bar charts
- Responsive design for both desktop and mobile
- WebGPU acceleration for fast inference
- Clear canvas functionality
- Real-time confidence scoring

## 🤖 Model Architecture 

### MLP Model
| Layer | Units | Activation |
|-------|--------|------------|
| Input | 784 (28x28) | - |
| Dense | 512 | SiLU |
| Dense | 512 | SiLU |
| Dense | 10 | - |

### CNN Model
| Layer | Filters/Units | Activation |
|-------|---------------|------------|
| Conv2D | 8 filters, 3x3 | ReLU |
| MaxPool2D | 2x2 | - |
| Conv2D | 16 filters, 3x3 | ReLU |
| MaxPool2D | 2x2 | - |
| Dense | 10 | - |

## Setup and Local Development

1. Install Dawn WebGPU library:
```bash
# For Linux (x86_64)
sudo curl -L https://github.com/wpmed92/pydawn/releases/download/v0.3.0/libwebgpu_dawn_x86_64.so -o /usr/lib/libwebgpu_dawn.so
```

2. Clone the repository:
```bash
git clone https://github.com/LucaricotRouge/Tinygrad_Mnist_Classifier.git
cd Tinygrad_Mnist_Classifier
```

3. Train the models:
```bash
# Train MLP
STEPS=100 JIT=1 python mnist_mlp.py

# Train CNN
STEPS=100 JIT=1 python mnist_convnet.py
```

4. Start the local server:
```bash
cd Classifier
python -m http.server 8000
```

5. Open in your browser:
   - Visit [http://localhost:8000](http://localhost:8000)
   - Use a WebGPU-enabled browser (ex : Chrome, Firefox, Edge)

## Hyperparameter Analysis
For detailed information about model training and hyperparameter optimization, see our [Hyperparameter Documentation](Classifier/HYPERPARAMETERS.md).

## Project Retrospective

### Technical Challenges
1. **WebGPU Integration**: Implementing the model export pipeline from TinyGrad to WebGPU required careful attention to tensor layout and preprocessing normalization.

2. **Input Preprocessing**: Achieving consistent recognition required matching the web input preprocessing with MNIST training data characteristics, particularly for:
   - Digit centering and scaling
   - Stroke thickness normalization
   - Input value range normalization (-1 to 1)

3. **Model Optimization**: Finding the right balance of hyperparameters to achieve both high accuracy and good generalization to hand-drawn digits required extensive experimentation, particularly with:
   - Data augmentation parameters (rotation, scale, shift)
   - Learning rate scheduling
   - Batch size optimization

