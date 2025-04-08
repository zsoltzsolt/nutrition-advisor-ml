from fastapi import FastAPI, File, UploadFile, Query, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from services.nutrition import calculate_meal_nutrition, retrieve_similar_ingredients
from services.vectorstore import retriever
import pandas as pd
from typing import AsyncGenerator, NoReturn
from dotenv import load_dotenv
from openai import AsyncOpenAI
import uuid
from pydantic import BaseModel
import json
from models.nutrition_model import CalculationRequest 
from fastapi.middleware.cors import CORSMiddleware
from services.nutrition import load_knowledge_base, calculate_bmr, calculate_tdee, calculate_macros
from utils.file_utils import save_temp_file
import os

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
def calculate(request: CalculationRequest):
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
    save_temp_file(file, temp_file_path)  

    try:
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