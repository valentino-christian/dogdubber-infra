# Dog Breed Classifier

This model identifies dog breeds from images using a fine-tuned MobileNetV2 architecture. It can classify 120 different dog breeds with high accuracy.

## Usage

Upload an image of a dog and get back the predicted breed with confidence scores.

## API

- **Input**: Image file (JPG, PNG, etc.)
- **Output**: Top breed predictions with confidence scores

## Example

```python
import replicate

output = replicate.run(
    "your-username/dog-breed-classifier",
    input={"image": open("dog_photo.jpg", "rb")}
)

print(f"This is a {output['top_breed']} with {output['confidence']*100:.1f}% confidence")
```

## Supported Breeds

This model can identify 120 dog breeds including:
- Golden Retriever
- Labrador Retriever  
- German Shepherd
- Beagle
- And 116 more breeds...

## Model Details

- Architecture: MobileNetV2
- Input size: 224x224 pixels
- Framework: PyTorch + HuggingFace Transformers