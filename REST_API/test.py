from fastapi import FastAPI, Depends, HTTPException, Cookie
from fastapi.security import OAuth2AuthorizationCodeBearer
from authlib.integrations.base_client import MismatchingStateError  # Add this
from authlib.integrations.starlette_client import OAuth  # Keep existing
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
import jwt
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("JWT_SECRET_KEY"),
    session_cookie="session_id",
    same_site="lax",
    max_age=3600  # 1 hour expiration
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

# After (fixed)
@app.get("/login")
async def login(request: Request):
    redirect_uri = os.getenv("REDIRECT_URL")
    return await oauth.google.authorize_redirect(request, redirect_uri)

# Auth callback route
@app.get("/auth")
async def auth(request: Request):
    try:

        request.session["oauth_state"] = request.query_params.get("state")
        

        token = await oauth.google.authorize_access_token(request)
        user = token.get("userinfo")
        
        # Create JWT
        jwt_token = jwt.encode(
            {"sub": user["sub"], "email": user["email"], "name": user.get("name")},
            os.getenv("JWT_SECRET_KEY"),
            algorithm="HS256"
        )
        
        # Proper redirect with cookie
        response = RedirectResponse(url=os.getenv("FRONTEND_URL"))
        response.set_cookie(
            key="token",
            value=jwt_token,
            httponly=True,
            secure=not os.getenv("DEVELOPMENT", False)
        )
        return response
        
    except MismatchingStateError:
        raise HTTPException(400, "Session expired - restart authentication flow")

# Auth dependency for protected routes
def get_current_user(token: str = Cookie(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
        user_id = payload.get("sub")
        user_email = payload.get("email")
        
        if user_id is None or user_email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
            
        return {"user_id": user_id, "user_email": user_email}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Protected endpoint example
@app.get("/protected")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": "This is a protected route", "user": current_user}


