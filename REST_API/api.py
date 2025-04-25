import torch
from torchvision import transforms
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Form, Depends, Cookie, status
from pydantic import BaseModel
from fastapi.responses import FileResponse
import numpy as np
import os
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict
import logging
import jwt
from starlette.responses import RedirectResponse
from authlib.integrations.base_client import MismatchingStateError
import models, schemas, database
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from starlette.requests import Request
from dotenv import load_dotenv
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from cnn_models import resnet50_model, inception_resnet_model, googlenet_model
from urllib.parse import urlparse
from fastapi.responses import JSONResponse


cnn_models = {
    'google_net': googlenet_model,
    'resnet50': resnet50_model,
    'inc_resnet_v2': inception_resnet_model
}

for model_name, model in cnn_models.items():
    # Set the model to evaluation mode
    model.eval()
    # Load the model weights
    state_dict = torch.load(f"classification_models/{model_name}.pth", map_location=torch.device('cpu'))
    # Load the state dict into the model
    model.load_state_dict(state_dict)

unet_model = torch.load("segmentation/unet_model_full.pth", weights_only=False, map_location=torch.device('cpu'))
unet_model.eval()




# Load and validate environment variables
required_env = [
    "GOOGLE_CLIENT_ID",
    "GOOGLE_CLIENT_SECRET",
    "REDIRECT_URL",
    "JWT_SECRET_KEY",
    "SESSION_SECRET_KEY",
    "FRONTEND_URL",
    "ENVIRONMENT"
]
load_dotenv()
for key in required_env:
    if not os.getenv(key):
        raise RuntimeError(f"Missing required environment variable: {key}")


# Validate that REDIRECT_URL and FRONTEND_URL share the same host
redirect_host = urlparse(os.getenv("REDIRECT_URL")).hostname
frontend_host = urlparse(os.getenv("FRONTEND_URL")).hostname
print(f"Environment variables: {os.getenv('REDIRECT_URL')}, {os.getenv('FRONTEND_URL')}")
print(f"Redirect host: {redirect_host}, Frontend host: {frontend_host}")
if redirect_host != frontend_host:
    raise RuntimeError(
        f"Host mismatch: REDIRECT_URL host '{redirect_host}' != FRONTEND_URL host '{frontend_host}'. "
        "Use the same host (e.g. both 'localhost' or both '127.0.0.1')."
    )



@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield

app = FastAPI(lifespan=lifespan)

def get_or_create_user_db(db: Session, email: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        user = models.User(email=email) 
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@app.get("/models/", response_model=List[str])
async def get_models():
    """
    Returns a list of available model names.
    """
    model_names = list(cnn_models.keys())
    return model_names


frontend_origin = os.getenv("FRONTEND_URL").rstrip('/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session middleware for OAuth state (ensure cookie persists across redirects)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET_KEY"),
    session_cookie="session_id",
    same_site="lax",
    https_only=False,
    max_age=3600,  
)
# Configure OAuth
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

@app.get("/login")
async def login(request: Request):
    # Debug: show incoming cookies
    print("[/login] cookies on request:", request.cookies)
    redirect_uri = os.getenv("REDIRECT_URL")
    return await oauth.google.authorize_redirect(request, redirect_uri)



# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/auth")
async def auth(request: Request,
               db: Session = Depends(get_db),):
    try:
        # Debug: inspect cookies present at callback
        print("[/auth] cookies on callback:", request.cookies)

        # Exchange authorization code for tokens
        token = await oauth.google.authorize_access_token(request)

        # Attempt to parse ID token
        user = None
        if token.get("id_token"):
            try:
                user = await oauth.google.parse_id_token(request, token)
            except Exception:
                user = None

        # Fallback to userinfo
        if not user:
            user = await oauth.google.userinfo(token=token)
        get_or_create_user_db(db, user["email"])  


        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to fetch user information"
            )

        # Issue our own JWT
        jwt_token = jwt.encode(
            {"sub": user["sub"], "email": user["email"], "name": user.get("name")},
            os.getenv("JWT_SECRET_KEY"),
            algorithm="HS256"
        )

        # Set HTTP-only cookie and redirect to frontend
        response = RedirectResponse(url=os.getenv("FRONTEND_URL"))
        response.set_cookie(
        key="token",
        value=jwt_token,
        httponly=True,
        secure=False,          # OK in dev
        samesite="lax",        # ← not “none”
        max_age=3600,
        # remove domain
        )
        return response

    except MismatchingStateError:
        # Likely caused by missing session cookie on callback
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=("Session expired or mismatched state. "
                    "Ensure you navigate to /login in your browser (not via Swagger UI), "
                    "and that cookies are enabled and sent on redirects.")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}"
        )

from sqlalchemy.orm import Session
import models


# Dependency for verifying our JWT
def get_current_user(token: str = Cookie(None)):
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token"
        )
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET_KEY"),
            algorithms=["HS256"]
        )
        if not payload.get("sub") or not payload.get("email"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return {"user_id": payload["sub"], "user_email": payload["email"], "name": payload.get("name")}  
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )






# Define preprocessing function
def preprocess_image(image_path):
    """
    Preprocesses an image for input into a machine learning model.
    This function opens an image from the specified file path, converts it to 
    grayscale, resizes it to 256x256 pixels, and transforms it into a tensor 
    with an added batch dimension.
    This function opens an image from the specified file path, converts it to 
    grayscale, resizes it to 256x256 pixels, and transforms it into a tensor 
    with an added batch dimension.
    Args:
        image_path (str): The file path to the image to be preprocessed.
    Returns:
        torch.Tensor: A tensor representation of the preprocessed image with 
        shape (1, 1, 256, 256), where the dimensions represent batch size, 
        channels, height, and width respectively.
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
        output_tensor (torch.Tensor): The output tensor from the model, 
            typically containing predicted values for each pixel.
        original_size (tuple): A tuple (width, height) representing the 
            original size of the input image to which the output should be resized.
    Returns:
        PIL.Image.Image: A binary segmented image resized to the original size, 
            where pixel values are either 0 or 255.
            where pixel values are either 0 or 255.
    """


    output_image = output_tensor.squeeze().detach().numpy()
    binary_mask = (output_image > 0.5).astype("uint8") * 255  # Binary thresholding (0 or 255)
    segmented_image = Image.fromarray(binary_mask).resize(original_size)  # Resize back to original size
    return segmented_image






def create_masked_image(original_image_path, segmentation_mask, mode="overlay"):
    """
    Creates a masked image by either overlaying a segmentation mask on the original image
    or extracting the masked region from the original image.
    Creates a masked image by either overlaying a segmentation mask on the original image
    or extracting the masked region from the original image.
    Args:
        original_image_path (str): The file path to the original image.
        segmentation_mask (PIL.Image.Image): The segmentation mask as a PIL Image object.
        mode (str, optional): The mode of masking. Can be "overlay" (default) or "extract".
            - "overlay": Overlays the segmentation mask on the original image with a red tint.
            - "extract": Extracts only the masked region from the original image, setting the background to black.
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


@app.get("/masked_segment/")
async def masked_segment(
    filename: str = Query(..., description="Relative path to image in test_data folder"),
    mode: str = Query("extract", description="Mode for masking: 'extract' or 'overlay'")
):
    """
    Processes a stored image file, performs segmentation using a pre-trained model, 
    and returns a masked image based on the specified mode.
    
    Args:
        filename (str): Relative path to image in the test_data folder.
        mode (str, optional): The mode for creating the masked image. Defaults to "extract".
        
    Returns:
        FileResponse: A response containing the masked image file in PNG format.
        
    Raises:
        HTTPException: If the file is not found or an error occurs during processing.
    """
    try:
        # Construct the path to the image file
        image_path = os.path.join("test_data", filename)
        
        # Verify the file exists
        if not os.path.isfile(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Open the original image to get its size
        original_image = Image.open(image_path)
        original_size = original_image.size

        # Preprocess the image
        input_tensor = preprocess_image(image_path)

        # Perform inference
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)

        # Postprocess the output (resize back to original size)
        segmented_image = postprocess_output(output_tensor, original_size)

        # Create masked image based on the specified mode
        masked_image = create_masked_image(image_path, segmented_image, mode=mode)
        
        # Save the masked image to a file
        masked_image_path = "masked_image.png"
        masked_image.save(masked_image_path)

        # Return the masked image as a response
        return FileResponse(masked_image_path, media_type="image/png", filename="masked_image.png")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/masked_segment_upload/")
async def masked_segment_upload(
    file: UploadFile = File(...), 
    mode: str = Form("extract"),
    current_user: dict = Depends(get_current_user)
):
    """
    Protected endpoint that processes an uploaded image file, performs segmentation 
    using a pre-trained model, and returns a masked image based on the specified mode.
    
    Args:
        file (UploadFile): The uploaded image file to be processed.
        mode (str, optional): The mode for creating the masked image. Defaults to "extract".
        current_user (dict): The authenticated user information.
        
    Returns:
        FileResponse: A response containing the masked image file in PNG format.
        
    Raises:
        HTTPException: If an error occurs during the processing of the image.
    """
    try:
        # Read uploaded file
        image_data = await file.read()
        temp_path = f"temp_image_{current_user['user_id']}.png"
        with open(temp_path, "wb") as f:
            f.write(image_data)

        # Open the original image to get its size
        original_image = Image.open(temp_path)
        original_size = original_image.size

        # Preprocess the image
        input_tensor = preprocess_image(temp_path)

        # Perform inference
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)

        # Postprocess the output (resize back to original size)
        segmented_image = postprocess_output(output_tensor, original_size)

        # Create masked image based on the specified mode
        masked_image = create_masked_image(temp_path, segmented_image, mode=mode)
        
        # Save the masked image to a file
        masked_image_path = f"masked_image_{current_user['user_id']}.png"
        masked_image.save(masked_image_path)

        # Clean up temporary files
        os.remove(temp_path)

        # Return the masked image as a response
        return FileResponse(
            masked_image_path, 
            media_type="image/png", 
            filename=f"masked_{file.filename}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



# @app.post("/predict/")
# async def predict_image(file: UploadFile = File(...), model_name: str="google_net"):
#     """
#     Handles the prediction of an uploaded medical image by performing segmentation 
#     and classification to determine the likelihood of specific conditions.
#     Args:
#         file (UploadFile): The uploaded image file to be processed.
#     Returns:
#         dict: A dictionary containing the prediction results or an error message.
#             - "prediction" (str): The predicted class label (e.g., "COVID", "Normal").
#             - "confidence_scores" (dict): A dictionary mapping class labels to their 
#               respective confidence scores.
#             - "error" (str, optional): An error message if an exception occurs.
#     Workflow:
#         1. Reads the uploaded image file and saves it temporarily.
#         2. Preprocesses the image for segmentation using a UNet model.
#         3. Resizes the segmented output back to the original image size.
#         4. Creates a masked image based on the segmentation result.
#         5. Preprocesses the masked image for classification using a GoogLeNet model.
#         6. Performs classification to predict the likelihood of specific conditions.
#         7. Returns the predicted class and confidence scores as a JSON response.
#     Raises:
#         Exception: If any error occurs during the processing of the image.
#     """
    
#     try:
#         # Read uploaded file
#         image_data = await file.read()
#         with open("temp_image.png", "wb") as f:
#             f.write(image_data)

#         # Open the original image to get its size
#         original_image = Image.open("temp_image.png")
#         original_size = original_image.size

#         # Preprocess the image for segmentation
#         input_tensor = preprocess_image("temp_image.png")

#         # Perform segmentation
#         with torch.no_grad():
#             output_tensor = unet_model(input_tensor)

#         # Postprocess the output (resize back to original size)
#         segmented_image = postprocess_output(output_tensor, original_size)

#         # Create masked image (using "extract" mode)
#         masked_image = create_masked_image("temp_image.png", segmented_image, mode="extract")

#         # Define preprocessing for classification
#         classify_transform = transforms.Compose([
#             transforms.Resize((256, 256)),  
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#         # Preprocess the masked image for classification
#         input_tensor = classify_transform(masked_image).unsqueeze(0)
        
#         # Set model to evaluation mode
#         model = cnn_models[model_name]
#         googlenet_model.eval()
        
#         # Perform classification
#         with torch.no_grad():
#             outputs = googlenet_model(input_tensor)
#             probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
#         # Get prediction
#         classes = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
#         class_probs = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
#         predicted_class = classes[torch.argmax(probabilities).item()]
        
#         # Return prediction as JSON
#         return {
#             "prediction": predicted_class,
#             "confidence_scores": class_probs
#         }
        
#     except Exception as e:
#         return {"error": str(e)}


@app.post("/predict_stored/")
async def predict_stored_image(
    image_path: str = Query(..., description="Relative path to image in mounted storage"),
    model_name: str = Query("google_net"),
) -> Dict:
    """
    Predict on an image from the mounted storage (test_data).
    """
    BASE_DIR = "test_data"
    abs_image_path = os.path.join(BASE_DIR, image_path)
    if not os.path.isfile(abs_image_path):
        raise HTTPException(status_code=404, detail="Image not found in storage.")

    try:
        # Segmentation
        original_image = Image.open(abs_image_path)
        original_size = original_image.size
        input_tensor = preprocess_image(abs_image_path)
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)
        segmented_image = postprocess_output(output_tensor, original_size)
        masked_image = create_masked_image(abs_image_path, segmented_image, mode="extract")

        # Classification
        classify_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        input_tensor = classify_transform(masked_image).unsqueeze(0)
        model = cnn_models[model_name]
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        classes = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
        class_probs = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        predicted_class = classes[torch.argmax(probabilities).item()]
        return {
            "prediction": predicted_class,
            "confidence_scores": class_probs,
            "image_path": image_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

from fastapi.responses import Response

@app.get("/myimages/{image_id}")
async def get_image(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieves a specific image by ID if it belongs to the authenticated user.
    
    Args:
        image_id: The ID of the image to retrieve
        db: Database session dependency
        current_user: The authenticated user information from the JWT token
        
    Returns:
        Response: The image binary data with appropriate content type
        
    Raises:
        HTTPException: If the image is not found or doesn't belong to the user
    """
    email = current_user.get("user_email")
    
    # Get the user from the database
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get the specific image
    image = db.query(models.Image).filter(
        models.Image.id == image_id,
        models.Image.user_id == user.id
    ).first()
    
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image not found or you don't have permission to access it"
        )
    
    # Return the image data with the appropriate content type
    return Response(
        content=image.data,
        media_type=image.mime_type
    )





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
            if os.path.basename(root).lower() == type.lower():
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



@app.post("/upload_image/", status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Protected endpoint for uploading image data.
    Associates the uploaded image with the authenticated user.
    """
    # Validate file type (accept only images)
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read file data
    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Empty file")

    # Get user from DB
    user = get_or_create_user_db(db, current_user["user_email"])

    # Store image in DB
    new_image = models.Image(
        filename=file.filename,
        data=image_data,
        mime_type=file.content_type,
        owner=user
    )
    db.add(new_image)
    db.commit()
    db.refresh(new_image)

    return {"filename": new_image.filename, "message": "Image uploaded successfully"} 


@app.get("/my-images/", response_model=List[schemas.ImageOut])
async def get_user_images(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Returns list of authenticated user's images with URLs
    """
    # Get user from DB
    user = db.query(models.User).filter(
        models.User.email == current_user["user_email"]
    ).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get user's images and build URLs
    images = []
    for img in user.images:
        images.append({
            "id": img.id,
            "filename": img.filename,
            "url": f"/images/{img.id}"  # Uses existing image endpoint
        })
    
    return images

@app.post("/predict-db/")
async def predict_database_image(
    image_id: int = Query(..., description="ID of image in database"),
    model_name: str = Query("google_net"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Protected endpoint for predicting on images stored in the database.
    Only allows prediction on images owned by the authenticated user.
    """
    
    temp_path = None  # Initialize temp_path to avoid undefined variable issues
    try:
        user = db.query(models.User).filter(models.User.email == current_user["user_email"]).first()
        # Get image from database with ownership check
        image = db.query(models.Image).filter(
            models.Image.id == image_id,
            models.Image.user_id == user.id
        ).first()

        if not image:
            raise HTTPException(status_code=404, detail="Image not found or unauthorized")

        # Create temporary file from database blob
        temp_path = f"temp_db_image_{image_id}.png"
        with open(temp_path, "wb") as f:
            f.write(image.data)

        # Reuse existing prediction pipeline
        original_image = Image.open(temp_path)
        original_size = original_image.size
        original_image.close()  # Close the image after getting its size
        
        # Segmentation
        input_tensor = preprocess_image(temp_path)
        with torch.no_grad():
            output_tensor = unet_model(input_tensor)
        segmented_image = postprocess_output(output_tensor, original_size)
        masked_image = create_masked_image(temp_path, segmented_image, mode="extract")
        # Classification
        classify_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        input_tensor = classify_transform(masked_image).unsqueeze(0)
        model = cnn_models[model_name]
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Process results
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        classes = ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]
        class_probs = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        predicted_class = classes[torch.argmax(probabilities).item()]

        # Make sure all image handles are closed
        if 'masked_image' in locals() and masked_image is not None:
            masked_image.close()
        if 'segmented_image' in locals() and segmented_image is not None:
            segmented_image.close()

        # Cleanup temporary file
        os.remove(temp_path)

        return {
            "prediction": predicted_class,
            "confidence_scores": class_probs,
            "image_id": image_id
        }
    except Exception as e:
        if temp_path and os.path.exists(temp_path):  # Check if temp_path is defined and exists
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/myimages/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Deletes an image by ID if it belongs to the authenticated user.
    Returns 204 No Content on success, 404 if not found or unauthorized.
    """
    # Get the user from the database
    user = db.query(models.User).filter(models.User.email == current_user["user_email"]).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Find the image belonging to the user
    image = db.query(models.Image).filter(
        models.Image.id == image_id,
        models.Image.user_id == user.id
    ).first()

    if not image:
        raise HTTPException(
            status_code=404,
            detail="Image not found or you don't have permission to delete it"
        )

    db.delete(image)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/image_count/")
async def get_image_count(
    category: Optional[str] = Query(None, description="Filter by category (e.g., COVID, Normal)"),
    type: Optional[str] = Query("images", description="Type of items to count ('images' or 'masks')")
):
    """
    Returns the count of images or masks in the static directory.
    
    Args:
        category: Optional filter by category (e.g., COVID, Normal)
        type: Type of items to count (default is 'images', can be 'masks')
        
    Returns:
        dict: A dictionary with the count of images/masks
    """
    try:
        # Use the existing function to get all matching images
        images = get_images_from_directory(BASE_DIR, category, type=type)
        
        # Return the count
        return {
            "count": len(images),
            "type": type,
            "category": category if category else "all"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting images: {str(e)}")



@app.get("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logs out the current user by deleting the authentication token cookie.
    
    Args:
        current_user: The authenticated user information from the JWT token
        
    Returns:
        A response with a deleted token cookie and success message
    """
    # Create response with success message using JSONResponse
    response = JSONResponse(content={"message": "Logout successful"})
    
    # Delete the token cookie
    response.delete_cookie(
        key="token",
        httponly=True,
        secure=False,  # OK in dev
        samesite="lax"
    )
    
    return response