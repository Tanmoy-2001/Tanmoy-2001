# Image Classification with Vision Transformer

A computer-vision demo that runs image classification using a pretrained Vision Transformer (ViT) model from Hugging Face.

## Features
- Upload JPG, JPEG, or PNG images.
- Run pretrained image classification.
- Display top-k predicted labels and confidence scores.
- Uses a public pretrained checkpoint.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads model weights and requires an internet connection and sufficient disk space. CPU inference works but may be slower; a compatible GPU can accelerate inference.

## Model
Default checkpoint: `google/vit-base-patch16-224`. See the Hugging Face model card for its training data, intended use, and limitations.

## Responsible use
Predictions may be incorrect, especially for images outside the model's training distribution. Do not use the output for safety-critical or high-impact decisions.
