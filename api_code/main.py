from fastapi import FastAPI, Response, Request
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from routers import data_and_chart

app = FastAPI()

# Add session middleware with a secret key
app.add_middleware(SessionMiddleware, secret_key="840ecc2dfc070af39d1b")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"]   # Allows all headers
)

# Health check endpoint
@app.get("/healthy")
def health_check():
    return {"status": "healthy"}

# @app.get("/set-cookie/")
# def set_cookie(response: Response):
#     response.set_cookie(key="session_cookie", value="your-session-id", httponly=True)
#     return {"message": "Cookie set"}

# @app.get("/get-cookie/")
# def get_cookie(request: Request):
#     session_cookie = request.cookies.get("session_cookie")
#     return {"session_cookie": session_cookie}

# Router for search
# app.include_router(fastAPI_permission_control.router)
app.include_router(data_and_chart.router)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=3004)
