import json
import os

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
