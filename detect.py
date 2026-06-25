import os
import sys
from paddleocr import TextDetection

def main():
    # Allow passing image path as argument, default to test/1.jpg if not provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test/1.jpg"
        print(f"No image path provided. Defaulting to: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    print("Initializing PP-OCRv6 medium detection model...")
    # PP-OCRv6_medium_det is the official registered model name in paddlex
    model = TextDetection(model_name="PP-OCRv6_medium_det")
    
    print(f"Running text detection on: {image_path}")
    output = model.predict(input=image_path, batch_size=1)
    
    # Ensure the output directory exists
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    for res in output:
        # Print results to console
        res.print()
        
        # Save annotated image (saves as <filename>_res.jpg in the output folder)
        res.save_to_img(save_path=output_dir)
        
        # Save coordinates to json file
        json_path = os.path.join(output_dir, "res.json")
        res.save_to_json(save_path=json_path)
        
    print(f"\nDone! Results saved in '{output_dir}/' directory.")

if __name__ == "__main__":
    main()
