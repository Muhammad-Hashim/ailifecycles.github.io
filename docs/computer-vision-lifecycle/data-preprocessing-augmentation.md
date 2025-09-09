# Data Preprocessing and Augmentation

## Overview

Data preprocessing and augmentation prepare your images for training and increase dataset diversity to improve model generalization.

## Preprocessing Techniques

### Image Normalization
- Pixel value scaling (0-1 or -1 to 1)
- Channel-wise normalization
- Z-score normalization

### Resizing and Cropping
- Uniform image dimensions
- Aspect ratio preservation
- Center/random cropping

## Data Augmentation

### Geometric Transformations
- Rotation, scaling, translation
- Horizontal/vertical flipping
- Shearing and perspective changes

### Color Augmentations  
- Brightness and contrast adjustment
- Hue and saturation modification
- Gamma correction

## Implementation

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## Next Steps

- [Feature Engineering](./feature-engineering.md)
- [Model Training](./model-training.md)
