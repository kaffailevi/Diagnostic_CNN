import torch
from torchvision import transforms
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import numpy as np
import os
from torchvision.models import googlenet
import torch.nn as nn



# Load the trained U-Net model
unet_model = torch.load("unet_model_full.pth", weights_only=False, map_location=torch.device('cpu'))
unet_model.eval()

#Load trained CNN GoogLeNet model
googlenet_model = googlenet(weights='IMAGENET1K_V1')
num_classes = 4
googlenet_model.fc = nn.Sequential(
    nn.Linear(googlenet_model.fc.in_features, 512),  # Első rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(512, 256),  # Második rejtett réteg
    nn.LeakyReLU(0.01),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)  # Kimeneti réteg
)
state_dict = torch.load("googlenet_model_weights.pth", map_location=torch.device('cpu'))
googlenet_model.load_state_dict(state_dict)

googlenet_model.eval()


# Define preprocessing function
def preprocess_image(image_path):
    """
    Preprocesses an image for input into a machine learning model.

    This function reads an image from the specified file path, converts it to grayscale,
    resizes it to 256x256 pixels, and transforms it into a PyTorch tensor with an added
    batch dimension.

    Args:
        image_path (str): The file path to the image to be preprocessed.

    Returns:
        torch.Tensor: A 4D tensor representing the preprocessed image, with shape (1, 1, 256, 256).
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale if needed
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust to your model's input size
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define postprocessing function
def postprocess_output(output_tensor, original_size):
    """
    Post-processes the output tensor from a model to generate a segmented image.

    This function applies binary thresholding to the output tensor, converts it 
    into a binary mask, and resizes it to match the original image size.

    Args:
        output_tensor (torch.Tensor): The output tensor from the model, expected 
            to have a single channel with values in the range [0, 1].
        original_size (tuple): A tuple (width, height) representing the size of 
            the original image to which the segmented image will be resized.

    Returns:
        PIL.Image.Image: A binary segmented image resized to the original size, 
        where pixel values are either 0 or 255.
    """
    output_image = output_tensor.squeeze().detach().numpy()
    binary_mask = (output_image > 0.5).astype("uint8") * 255  # Binary thresholding (0 or 255)
    segmented_image = Image.fromarray(binary_mask).resize(original_size)  # Resize back to original size
    return segmented_image


app = FastAPI()

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    """
    Asynchronously processes an uploaded image file, segments it using a pre-trained model, 
    and returns the segmented image as a response.
    Args:
        file (UploadFile): The uploaded image file to be segmented.
    Returns:
        FileResponse: A response containing the segmented image file in PNG format.
        dict: An error message in case of an exception.
    Workflow:
        1. Reads the uploaded image file and saves it temporarily.
        2. Opens the image to retrieve its original size.
        3. Preprocesses the image for model inference.
        4. Performs segmentation using a pre-trained model.
        5. Postprocesses the output to resize it back to the original dimensions.
        6. Saves the segmented image to a file.
        7. Returns the segmented image as a downloadable response.
    Raises:
        Exception: If any error occurs during the processing of the image.
    """
    try:
        # Read uploaded file
        image_data = await file.read()
        with open("temp_image.png", "wb") as f:
            f.write(image_data)

        # Open the original image to get its size
        original_image = Image.open("temp_image.png")
        original_size = original_image.size

        # Preprocess the image
        input_tensor = preprocess_image("temp_image.png")

        # Perform inference
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)

        # Postprocess the output (resize back to original size)
        segmented_image = postprocess_output(output_tensor, original_size)

        # Save the segmented image to a file
        segmented_image_path = "segmented_image.png"
        segmented_image.save(segmented_image_path)

        # Return the segmented image as a response
        return FileResponse(segmented_image_path, media_type="image/png", filename="segmented_image.png")
    except Exception as e:
        return {"error": str(e)}


def create_masked_image(original_image_path, segmentation_mask, mode="overlay"):
    """
    Creates a masked image by combining the original image and its segmentation mask.

    Args:
        original_image_path (str): Path to the original input image.
        segmentation_mask (PIL.Image.Image): The segmentation mask as a binary image.
        mode (str): The masking mode - "overlay" or "extract".

    Returns:
        PIL.Image.Image: The masked image based on the specified mode.
    """
    # Open original image
    original_image = Image.open(original_image_path).convert("RGB")
    
    # Convert segmentation mask to numpy array (0 or 255 values)
    mask_array = np.array(segmentation_mask.convert("L"))
    
    # Convert original image to numpy array
    original_array = np.array(original_image)
    
    if mode == "extract":
        # Extract only the masked region (set background to black)
        # Normalize mask to 0-1 range for multiplication
        normalized_mask = mask_array / 255.0
        
        # Apply mask to each channel (multiply by mask)
        result_array = np.zeros_like(original_array)
        for i in range(3):  # RGB channels
            result_array[:,:,i] = original_array[:,:,i] * normalized_mask
            
        # Convert back to PIL Image
        result_image = Image.fromarray(result_array.astype(np.uint8))
        
        
    if mode == "overlay":  # Default overlay mode
        # Convert segmentation mask to RGBA with transparency (red overlay)
        mask_overlay = ImageOps.colorize(segmentation_mask.convert("L"), black="black", white="red")
        mask_overlay = mask_overlay.convert("RGBA")
        mask_overlay.putalpha(100)  # Set transparency level
        
        # Convert original to RGBA for compositing
        original_image = original_image.convert("RGBA")
        
        # Resize mask to match original image dimensions
        mask_overlay = mask_overlay.resize(original_image.size)
        
        # Composite (overlay) mask onto original image
        result_image = Image.alpha_composite(original_image, mask_overlay)
        result_image = result_image.convert("RGB")  # Convert back to RGB
    
    return result_image


@app.post("/masked_segment/")
async def masked_segment(file: UploadFile = File(...), mode: str = "extract"):
    """
    Processes an uploaded image file and returns a masked version 
    of the input with segmentation results.
    
    Args:
        file (UploadFile): The uploaded input file.
        mode (str): The masking mode - "overlay", "extract", or "multiply".
                   Default is "extract".

    Returns:
        FileResponse: A response containing a masked version of 
                     input in PNG format.
    """
    try:
        # Read uploaded file
        image_data = await file.read()
        with open("temp_image.png", "wb") as f:
            f.write(image_data)

        # Open the original image to get its size
        original_image = Image.open("temp_image.png")
        original_size = original_image.size

        # Preprocess the image
        input_tensor = preprocess_image("temp_image.png")

        # Perform inference
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)

        # Postprocess the output (resize back to original size)
        segmented_image = postprocess_output(output_tensor, original_size)

        # Create masked image based on the specified mode
        masked_image = create_masked_image("temp_image.png", segmented_image, mode=mode)
        
        # Save the masked image to a file
        masked_image_path = "masked_image.png"
        masked_image.save(masked_image_path)

        # Return the masked image as a response
        response = FileResponse(masked_image_path, media_type="image/png", filename="masked_image.png")
        # Clean up temporary files if needed
        # os.remove("temp_image.png")
        # os.remove(masked_image_path)
        # Optionally, you can remove the segmented image as well
        # os.remove("segmented_image.png")
        return response
    
    except Exception as e:
        return {"error": str(e)}



@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    """
    Processes an uploaded image file, segments it, creates a masked version,
    and returns the GoogleNet model's prediction.
    
    Args:
        file (UploadFile): The uploaded chest X-ray image file.
        
    Returns:
        dict: A JSON response containing the prediction class and confidence scores.
    """
    try:
        # Read uploaded file
        image_data = await file.read()
        with open("temp_image.png", "wb") as f:
            f.write(image_data)

        # Open the original image to get its size
        original_image = Image.open("temp_image.png")
        original_size = original_image.size

        # Preprocess the image for segmentation
        input_tensor = preprocess_image("temp_image.png")

        # Perform segmentation
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)

        # Postprocess the output (resize back to original size)
        segmented_image = postprocess_output(output_tensor, original_size)

        # Create masked image (using "extract" mode)
        masked_image = create_masked_image("temp_image.png", segmented_image, mode="extract")

        # Define preprocessing for classification
        classify_transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Preprocess the masked image for classification
        input_tensor = classify_transform(masked_image).unsqueeze(0)
        
        # Set model to evaluation mode
        googlenet_model.eval()
        
        # Perform classification
        with torch.no_grad():
            outputs = googlenet_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get prediction
        classes = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
        class_probs = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        predicted_class = classes[torch.argmax(probabilities).item()]
        
        # Return prediction as JSON
        return {
            "prediction": predicted_class,
            "confidence_scores": class_probs
        }
        
    except Exception as e:
        return {"error": str(e)}
