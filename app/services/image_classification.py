from transformers import pipeline
from PIL import Image

image_classifier = pipeline(
    task="zero-shot-image-classification",
    model="google/siglip2-base-patch16-224",
)

def is_food_image(image_path: str) -> bool:
    image = Image.open(image_path)

    candidate_labels = [
        "a dish of food",
        "a fruit",
        "a vegetable",
        "a drink",
        "a person",
        "an electronic device",
        "a car",
        "an animal",
        "a landscape",
        "no food"
    ]

    outputs = image_classifier(image, candidate_labels=candidate_labels)
    top_label = sorted(outputs, key=lambda x: x['score'], reverse=True)[0]['label']

    return any(keyword in top_label for keyword in ["food", "fruit", "vegetable", "drink", "dish"])