import json

with open("tinylama/formatted_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset loaded successfully!")
print("Total examples:", len(data))
