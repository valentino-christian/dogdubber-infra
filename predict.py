import torch
from PIL import Image
from transformers import pipeline
from cog import BasePredictor, Input, Path
import io

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading dog breed classifier...")
        
        # Load your model from HuggingFace
        self.classifier = pipeline(
            "image-classification", 
            model="valentinocc/dog-breed-classifier",
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        print("Model loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Input image of a dog"),
        top_k: int = Input(
            description="Number of top predictions to return", 
            default=3, 
            ge=1, 
            le=10
        )
    ) -> dict:
        """Run a single prediction on the model"""
        
        # Load and validate image
        try:
            pil_image = Image.open(image)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
        except Exception as e:
            return {"error": f"Failed to load image: {str(e)}"}
        
        # Run prediction
        try:
            results = self.classifier(pil_image)
            
            # Format results
            predictions = []
            for i, result in enumerate(results[:top_k]):
                predictions.append({
                    "rank": i + 1,
                    "breed": result["label"],
                    "confidence": round(result["score"], 4),
                    "percentage": f"{result['score'] * 100:.2f}%"
                })
            
            return {
                "predictions": predictions,
                "top_breed": predictions[0]["breed"] if predictions else "Unknown",
                "confidence": predictions[0]["confidence"] if predictions else 0.0
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}