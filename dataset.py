# Defamation Detection Dataset
# 10 defamatory statements (label 1) and 10 non-defamatory statements (label 0)

import pandas as pd
import json

# Create the dataset
defamation_data = [
    # Defamatory statements (label 1)
    {"text": "Billy is a liar who deceives everyone around him.", "label": 1},
    {"text": "Sarah cheated on all her exams to get her degree.", "label": 1},
    {"text": "The company's CEO has been embezzling funds for years.", "label": 1},
    {"text": "Professor Johnson is incompetent and doesn't understand his own subject.", "label": 1},
    {"text": "Maria deliberately spread false rumors to damage her colleague's reputation.", "label": 1},
    {"text": "The restaurant owner knowingly serves expired food to customers.", "label": 1},
    {"text": "Dr. Smith falsified his medical credentials and isn't actually qualified.", "label": 1},
    {"text": "The politician accepted bribes from corporate lobbyists.", "label": 1},
    {"text": "The charity director pockets donation money instead of helping those in need.", "label": 1},
    {"text": "The news reporter fabricates sources to support controversial stories.", "label": 1},
    
    # Non-defamatory statements (label 0)
    {"text": "Billy wears a baseball cap to school every day.", "label": 0},
    {"text": "Sarah studied for her exams last weekend.", "label": 0},
    {"text": "The company's CEO announced new initiatives at the meeting.", "label": 0},
    {"text": "Professor Johnson teaches advanced mathematics at the university.", "label": 0},
    {"text": "Maria shared her thoughts about the project during the discussion.", "label": 0},
    {"text": "The restaurant owner offers a diverse menu with vegetarian options.", "label": 0},
    {"text": "Dr. Smith specializes in cardiology at the regional hospital.", "label": 0},
    {"text": "The politician spoke at the community forum yesterday.", "label": 0},
    {"text": "The charity director organized a fundraising event for disaster relief.", "label": 0},
    {"text": "The news reporter covered the local festival this weekend.", "label": 0}
]

# Save as CSV and JSON for flexibility with different frameworks
df = pd.DataFrame(defamation_data)
df.to_csv("defamation_dataset.csv", index=False)

# Also save as JSON for easier loading in some contexts
with open("defamation_dataset.json", "w") as f:
    json.dump(defamation_data, f, indent=2)

print("Dataset created and saved to defamation_dataset.csv and defamation_dataset.json")
print("Total examples:", len(defamation_data))
print("Defamatory examples:", sum(1 for item in defamation_data if item["label"] == 1))
print("Non-defamatory examples:", sum(1 for item in defamation_data if item["label"] == 0))
