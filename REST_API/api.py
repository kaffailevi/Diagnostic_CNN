import torch
from torchvision import transforms
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import numpy as np
import os
from torchvision.models import googlenet
import torch.nn as nn
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import logging


from models import resnet50_model, inception_resnet_model, googlenet_model

models = {
    'google_net': googlenet_model,
    'resnet50': resnet50_model,
    'inc_resnet_v2': inception_resnet_model
}

for model_name, model in models.items():
    # Set the model to evaluation mode
    model.eval()
    # Load the model weights
    state_dict = torch.load(f"classification_models/{model_name}.pth", map_location=torch.device('cpu'))
    # Load the state dict into the model
    model.load_state_dict(state_dict)

unet_model = torch.load("segmentation/unet_model_full.pth", weights_only=False, map_location=torch.device('cpu'))
unet_model.eval()


# Define preprocessing function
def preprocess_image(image_path):
    """
    Preprocesses an image for input into a machine learning model.
    This function opens an image from the specified file path, converts it to 
    grayscale, resizes it to 256x256 pixels, and transforms it into a tensor 
    with an added batch dimension.
    Args:
        image_path (str): The file path to the image to be preprocessed.
    Returns:
        torch.Tensor: A tensor representation of the preprocessed image with 
        shape (1, 1, 256, 256), where the dimensions represent batch size, 
        channels, height, and width respectively.
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
    Args:
        output_tensor (torch.Tensor): The output tensor from the model, 
            typically containing predicted values for each pixel.
        original_size (tuple): A tuple (width, height) representing the 
            original size of the input image to which the output should be resized.
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
    Segments an uploaded image using a pre-trained U-Net model.
    Args:
        file (UploadFile): The uploaded image file to be segmented.
    Returns:
        FileResponse: The segmented image file in PNG format, returned as a response.
        dict: An error message if an exception occurs during processing.
    Workflow:
        1. Reads the uploaded image file and saves it temporarily.
        2. Opens the image to retrieve its original size.
        3. Preprocesses the image for input into the U-Net model.
        4. Performs inference using the U-Net model to generate a segmented output.
        5. Postprocesses the output to resize it back to the original image size.
        6. Saves the segmented image to a file.
        7. Returns the segmented image as a downloadable response.
    Raises:
        Exception: If any error occurs during the image processing or segmentation workflow.
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
    Creates a masked image by either overlaying a segmentation mask on the original image
    or extracting the masked region from the original image.
    Args:
        original_image_path (str): The file path to the original image.
        segmentation_mask (PIL.Image.Image): The segmentation mask as a PIL Image object.
        mode (str, optional): The mode of masking. Can be "overlay" (default) or "extract".
            - "overlay": Overlays the segmentation mask on the original image with a red tint.
            - "extract": Extracts only the masked region from the original image, setting the background to black.
    Returns:
        PIL.Image.Image: The resulting masked image as a PIL Image object.
    Raises:
        ValueError: If the provided mode is not "overlay" or "extract".
    Notes:
        - The segmentation mask should be a grayscale image where the mask region is white (255) 
          and the background is black (0).
        - In "overlay" mode, the mask is resized to match the dimensions of the original image 
          and applied with a red tint and transparency.
        - In "extract" mode, the mask is normalized to a 0-1 range and applied to the original 
          image to isolate the masked region.
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
    Processes an uploaded image file, performs segmentation using a pre-trained model, 
    and returns a masked image based on the specified mode.
    Args:
        file (UploadFile): The uploaded image file to be processed.
        mode (str, optional): The mode for creating the masked image. Defaults to "extract".
    Returns:
        FileResponse: A response containing the masked image file in PNG format.
        dict: An error message if an exception occurs.
    Raises:
        Exception: If any error occurs during the processing of the image.
    Workflow:
        1. Reads the uploaded image file and saves it temporarily.
        2. Opens the image to retrieve its original size.
        3. Preprocesses the image for input into the segmentation model.
        4. Performs inference using a pre-trained UNet model.
        5. Postprocesses the model's output to resize it back to the original image size.
        6. Creates a masked image based on the segmentation output and the specified mode.
        7. Saves the masked image to a file and returns it as a response.
        8. Handles any exceptions that occur during the process and returns an error message.
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
    Handles the prediction of an uploaded medical image by performing segmentation 
    and classification to determine the likelihood of specific conditions.
    Args:
        file (UploadFile): The uploaded image file to be processed.
    Returns:
        dict: A dictionary containing the prediction results or an error message.
            - "prediction" (str): The predicted class label (e.g., "COVID", "Normal").
            - "confidence_scores" (dict): A dictionary mapping class labels to their 
              respective confidence scores.
            - "error" (str, optional): An error message if an exception occurs.
    Workflow:
        1. Reads the uploaded image file and saves it temporarily.
        2. Preprocesses the image for segmentation using a UNet model.
        3. Resizes the segmented output back to the original image size.
        4. Creates a masked image based on the segmentation result.
        5. Preprocesses the masked image for classification using a GoogLeNet model.
        6. Performs classification to predict the likelihood of specific conditions.
        7. Returns the predicted class and confidence scores as a JSON response.
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


# A képek tárolási helye
BASE_DIR = "test_data"  # A fő mappa, ahol a kategóriák találhatóak (pl. COVID, Lung_Opacity, stb.)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Modell a kép adatairól
class ImageResponse(BaseModel):
    filename: str
    url: str

logging.basicConfig(level=logging.DEBUG)

def get_images_from_directory(directory: str, category: Optional[str] = None,type="images"):
    images = []

    # Bejárjuk a fő kategóriát (pl. COVID)
    for root, dirs, files in os.walk(directory):
        # Ha meg van adva kategória, akkor csak az adott kategóriában nézünk körül
        if category:
            logging.info(f"Checking directory: {root} for category: {category}")
            # Ellenőrizzük, hogy a kategória mappa benne van a mappa nevében
            if category.lower() in os.path.basename(root).lower():
                # Csak az "images" almappát nézzük meg az adott kategórián belül
                for dir_name in dirs:
                    if dir_name.lower() == type.lower():
                        images_path = os.path.join(root, dir_name)
                        for file in os.listdir(images_path):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                images.append(os.path.relpath(os.path.join(images_path, file), start=directory))
        else:
            # Ha nincs kategória megadva, akkor minden mappában keresünk az "images" almappában
            if os.path.basename(root).lower() == "images":
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        images.append(os.path.relpath(os.path.join(root, file), start=directory))

    return images

# Get images endpoint with pagination and category filter
@app.get("/images/", response_model=List[ImageResponse])
async def get_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(10,gt=0),
    category: Optional[str] = Query(None),
    type: Optional[str] = Query("images" , description="The type of images to retrieve (default is 'images').")
):
    """
    Visszaadja a képeket a tárolóból paginált módon.
    Ha kategória van megadva, akkor azt is figyelembe veszi a szűrés során.
    `skip` az első képek száma, amiket átugrunk, `limit` pedig a visszaadott képek számát adja meg.
    """
    # A képek listázása az összes almappából vagy csak egy adott kategóriából
    all_images = get_images_from_directory(BASE_DIR, category,type=type)

    # Paginálás alkalmazása
    paginated_images = all_images[skip:skip + limit]

    if not paginated_images:
        raise HTTPException(status_code=404, detail="No images found.")

    # Kép URL-ek visszaadása
    image_urls = [
        ImageResponse(
            filename=image,
            url=f"/static/{image}"  # A képek elérési útvonala
        ) for image in paginated_images
    ]

    return image_urls