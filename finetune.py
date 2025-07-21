import json

# Load your raw dataset
with open("tinylama\dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

formatted_data = []

for item in raw_data:
    topic = item.get("instruction", "")
    response_obj = item.get("response", {})

    if "qna" in response_obj:
        formatted_data.append({
            "instruction": f"What is {topic}?",
            "output": response_obj["qna"]
        })

    if "summary" in response_obj:
        formatted_data.append({
            "instruction": f"Summarize {topic}",
            "output": response_obj["summary"]
        })

    if "notes" in response_obj:
        notes = response_obj["notes"]
        if isinstance(notes, list):
            notes = "\n".join(notes)
        formatted_data.append({
            "instruction": f"Notes for {topic}",
            "output": notes
        })

    if "cheatsheet" in response_obj:
        cheatsheet = response_obj["cheatsheet"]
        if isinstance(cheatsheet, list):
            cheatsheet = "\n".join(cheatsheet)
        formatted_data.append({
            "instruction": f"Cheatsheet for {topic}",
            "output": cheatsheet
        })

# Save the formatted dataset
with open("formatted_dataset.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(" Formatted JSON saved successfully as 'formatted_dataset.json'")
