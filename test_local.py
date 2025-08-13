def test_locally():
    """Test your predictor locally before deploying to Replicate"""
    from predict import Predictor
    import tempfile
    import requests
    from PIL import Image
    
    print("Testing predictor locally...")
    
    # Initialize predictor
    predictor = Predictor()
    predictor.setup()
    
    try:
        img_path = "../englishsetter.jpg"
        
        # Run prediction
        result = predictor.predict(image=img_path, top_k=3)
        
        print("Test Results:")
        print("=" * 50)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Top prediction: {result['top_breed']}")
            print(f"Confidence: {result['confidence']}")
            print("\nAll predictions:")
            for pred in result['predictions']:
                print(f"  {pred['rank']}. {pred['breed']}: {pred['percentage']}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_locally()