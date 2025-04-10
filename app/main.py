import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from models.nutrition_model import UserData
from fastapi.middleware.cors import CORSMiddleware
from services import load_knowledge_base
from services import calculate_bmr, calculate_tdee, calculate_macros
from services import calculate_meal_nutrition
from services import is_food_image 

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000", 
    "http://localhost:5173",  
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

client = AsyncOpenAI()

@app.post("/calculate")
def calculate(request: UserData):
    try:
        kb = load_knowledge_base()
        weight = request.weight
        height = request.height
        age = request.age
        gender = request.gender
        activity_level = request.activity_level.lower()
        goal = request.goal

        available_goals = list(kb["goals"].keys())
        if goal not in available_goals:
            raise HTTPException(
                status_code=400, 
                detail=f"Goal '{goal}' is not recognized. Available options: {', '.join(available_goals)}"
            )

        bmr = calculate_bmr(weight, height, age, gender, kb)
        tdee = calculate_tdee(bmr, activity_level, kb)

        goal_rules = kb["goals"][goal]
        tdee_goal = tdee + goal_rules["tdee_adjustment"]

        macros = calculate_macros(weight, tdee_goal, goal_rules)

        result = macros
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    temp_file_path = os.path.join("../images/", file.filename)

    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        if not is_food_image(temp_file_path):
            return JSONResponse({"error": "Image does not appear to contain food."}, status_code=400)

        with open(temp_file_path, "rb") as image_file:
            meal_nutrition = calculate_meal_nutrition(image_file)

        if not meal_nutrition:
            return JSONResponse({"error": "Failed to extract ingredients or calculate nutrition."}, status_code=400)

        return JSONResponse(meal_nutrition)
    finally:
        print("done")

@app.get("/test")
async def greet_serban():
    return {"message": "Hello, Serban!"}