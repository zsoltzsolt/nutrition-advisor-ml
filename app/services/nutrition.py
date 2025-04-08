from collections import defaultdict
import pandas as pd
from services.vectorstore import retriever
import os
import json
from services.image_processing import encode_image
from openai import OpenAI
from typing import Dict, Any, List

client = OpenAI()

prompt = """Analyze the given image and identify all the visible ingredients. For each ingredient, provide its name and the estimated weight in grams.

Naming Rules:
Avoid generic names like "Carrot" or "Tomato." Instead:
Specify type: (e.g., "Carrot, fresh, organic," "Tomato, raw, Roma").
Specify state or form: (e.g., "shredded," "sliced," "diced," "whole").
Specify preparation or intended use, if applicable (e.g., "Tomato, raw, for salad" or "Carrot, cooked, diced, for soup").
If the ingredient is raw, explicitly state it (e.g., "Tomato, raw").
If the ingredient is part of a prepared dish, describe it (e.g., "Tomato soup, canned, ready-to-serve").
Response Requirements:
The response must be a strict JSON object containing an array of objects.
Each object must have:
name (the detailed name of the ingredient).
grams (the estimated weight in grams).
Additional Notes:
Generic ingredient names are not acceptable.
Only include descriptive names that match the ingredient's form, state, and intended use.
Example Output:
[
  {"name": "Carrot, fresh, organic, whole", "grams": 50},
  {"name": "Tomato, raw, Roma, for salad", "grams": 30},
  {"name": "Feta cheese, crumbled, for garnish", "grams": 30}
]

"""

def extract_ingredients_from_image(image_file):
    base64_image = encode_image(image_file)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )
    print(response)
    result_text = response.choices[0].message.content.strip()

    if result_text.startswith("```json") and result_text.endswith("```"):
        result_text = result_text[7:-3].strip()  

    try:
        ingredients_data = json.loads(result_text)  
    except json.JSONDecodeError:
        return None 

    return ingredients_data  


def retrieve_nutritional_data(ingredient_name: str):
    docs = retriever.invoke(ingredient_name)
    if docs:
        return parse_nutritional_info(docs[0].page_content)
    else:
        return {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}

def parse_nutritional_info(nutrition_text):
    food_df = pd.read_json("/Users/diriczizsolt-csaba/Desktop/DemoCopy/fitbites-ml/app/services/food_db.json")
    return food_df[food_df["name"] == nutrition_text]["nutrients"].values[0]

def calculate_meal_nutrition(image_file):
    
    ingredients_data = extract_ingredients_from_image(image_file)
    print(ingredients_data)
    if not ingredients_data:
        return None

    total_nutrition = defaultdict(float)
    ingredient_nutrition = {}
    
    print(f"*******{ingredients_data}")
    for ingredient in ingredients_data:
        name = ingredient["name"]
        weight = ingredient["grams"] / 100
        nutrition_info = retrieve_nutritional_data(name)

        scaled_nutrition = {k: round(v, 2) for k, v in nutrition_info.items()}
        ingredient_nutrition[name] = scaled_nutrition
        ingredient_nutrition[name]["grams"] = ingredient["grams"]

        for key, value in scaled_nutrition.items():
            total_nutrition[key] += value
            total_nutrition[key] = round(total_nutrition[key], 2)
            
        elements = name.split(',')
        ingredient_nutrition[name]["name"] = ','.join(elements[:2])

    return {"ingredients": ingredient_nutrition, "total_meal": dict(total_nutrition)}


def retrieve_similar_ingredients(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    docs = retriever.invoke(query, top_k=top_k)
    results = []
    print(len(docs))
    for doc in docs:
        ingredient = parse_nutritional_info(doc.page_content)
        ingredient["name"] = doc.page_content
        ingredient["grams"] = 100
        results.append(ingredient)
    return results

def load_knowledge_base(file_path: str = "../knowledge-base/nutrition_db.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            kb = json.load(f)
    else:
        raise FileNotFoundError(f"Knowledge base file '{file_path}' not found.")
    return kb

def calculate_bmr(weight: float, height: float, age: int, gender: str, kb: dict) -> float:
    gender_key = gender.lower()
    formula_str = kb["bmr_formula"].get(gender_key)
    if not formula_str:
        raise ValueError("Unsupported gender. Use 'male' or 'female'.")
    return eval(formula_str, {"__builtins__": {}}, {"weight": weight, "height": height, "age": age})

def calculate_tdee(bmr: float, activity_level: str, kb: dict) -> float:
    factor = kb["activity_factors"].get(activity_level)
    if factor is None:
        valid_levels = ", ".join(kb["activity_factors"].keys())
        raise ValueError(f"Unsupported activity level. Choose from: {valid_levels}")
    return bmr * factor

def calculate_macros(weight: float, tdee_goal: float, goal_rules: dict) -> dict:
    protein_grams = weight * goal_rules["protein_g_per_kg"]
    protein_cal = protein_grams * 4  

    remaining_cal = tdee_goal - protein_cal

    total_ratio = goal_rules["carbs_percentage"] + goal_rules["fats_percentage"]
    carbs_cal = remaining_cal * (goal_rules["carbs_percentage"] / total_ratio)
    fats_cal = remaining_cal * (goal_rules["fats_percentage"] / total_ratio)

    carbs_grams = carbs_cal / 4  
    fats_grams = fats_cal / 9   

    return {
        "calories": tdee_goal,
        "protein": round(protein_grams, 2),
        "carbs": round(carbs_grams, 2),
        "fats": round(fats_grams, 2)
    }
