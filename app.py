import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load model once at startup
classifier = pipeline("image-classification", model="valentinocc/dog-breed-classifier")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Must be an image")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        results = classifier(image)
        
        predictions = []
        for i, result in enumerate(results[:5]):
            predictions.append({
                "rank": i + 1,
                "breed": result["label"],
                "confidence": round(result["score"], 4),
                "percentage": f"{result['score'] * 100:.2f}%"
            })
        
        return {
            "predictions": predictions,
            "top_breed": predictions[0]["breed"],
            "confidence": predictions[0]["confidence"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)