# 5400ProjectCode

Random Forest Code:
# ---------------------------------
# IMPORTS
# ---------------------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# LOAD THE DATASET
# ---------------------------------
df = pd.read_csv("/Users/rileycook/Desktop/5400 Files/LLM Project/netflix_cleaned_ML.csv")

# ---------------------------------
# FIXES FOR GENRE + STRING ISSUE
# ---------------------------------

# Ensure listed_in is treated as string
df["listed_in"] = df["listed_in"].astype(str)

# Create simplified "primary genre"
df["primary_genre"] = (
    df["listed_in"]
    .str.split(",")
    .str[0]
    .str.strip()
)

# ---------------------------------
# CLEAN + PREP FEATURES
# ---------------------------------

# Convert duration to string FIRST, then extract numeric
df["duration"] = df["duration"].astype(str)
df["duration"] = df["duration"].str.extract(r"(\d+)").astype(float)

# Final ML feature set
features = ["type", "release_year", "rating", "duration", "country"]
target = "primary_genre"

# Categorical columns ONLY (release_year & duration stay numeric)
categorical_cols = ["type", "rating", "country", target]

# Encode categorical columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# ---------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# TRAIN RANDOM FOREST
# ---------------------------------

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)

rf.fit(X_train, y_train)

# ---------------------------------
# ACCURACY
# ---------------------------------

print("Random Forest Accuracy:", rf.score(X_test, y_test))

# ---------------------------------
# VARIABLE IMPORTANCE PLOT
# ---------------------------------

importances = rf.feature_importances_

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

Interactive Prediction Code:
# ---------------------------------
# SMART INTERACTIVE RANDOM FOREST INPUT
# ---------------------------------

print("\n===============================")
print("        NETFLIX GENRE PREDICTOR")
print("===============================\n")

# ---------------------------------
# VALID OPTIONS FROM THE MODEL
# ---------------------------------

# Types (Movie, TV Show)
valid_types = list(le_dict["type"].classes_)

# Full rating list from dataset
all_ratings = list(le_dict["rating"].classes_)

# Allowed REAL MPAA + TV Ratings
allowed_ratings = [
    "G", "PG", "PG-13", "R", "NC-17", "NR", "UR",
    "TV-MA", "TV-14", "TV-PG", "TV-Y7", "TV-Y", "TV-G"
]

# Filter out bad/unwanted rating values
clean_ratings = [r for r in all_ratings if r in allowed_ratings]

# Split movie vs TV ratings
movie_ratings = [r for r in clean_ratings if not r.startswith("TV")]
tv_ratings = [r for r in clean_ratings if r.startswith("TV")]

# Valid countries (typed, not menu-based)
valid_countries = list(le_dict["country"].classes_)

# ---------------------------------
# Helper Function for Menu Choices
# ---------------------------------

def ask_choice(prompt, choices):
    """Ask user to choose one of the valid choices."""
    print(f"\n{prompt}")
    for idx, choice in enumerate(choices):
        print(f"{idx+1}. {choice}")
    while True:
        selection = input("Enter a number: ")
        if selection.isdigit() and 1 <= int(selection) <= len(choices):
            return choices[int(selection)-1]
        print("Invalid choice. Try again.")

# ---------------------------------
# 1. TYPE (Movie or TV Show)
# ---------------------------------

chosen_type = ask_choice("Choose a type:", valid_types)

# ---------------------------------
# 2. RELEASE YEAR
# ---------------------------------

while True:
    try:
        chosen_year = int(input("\nEnter release year (e.g., 2019): "))
        break
    except ValueError:
        print("Invalid input. Enter a number.")

# ---------------------------------
# 3. RATING (based on type)
# ---------------------------------

if chosen_type == "TV Show":
    chosen_rating = ask_choice("Choose a rating:", tv_ratings)
else:
    chosen_rating = ask_choice("Choose a rating:", movie_ratings)

# ---------------------------------
# 4. DURATION
# ---------------------------------

while True:
    try:
        chosen_duration = int(input("\nEnter duration (minutes for movies / seasons for shows): "))
        break
    except ValueError:
        print("Invalid input. Enter a number.")

# ---------------------------------
# 5. COUNTRY (typed, not menu-based)
# ---------------------------------

print("\nExamples of common countries: United States, India, United Kingdom, Canada, Japan")

while True:
    chosen_country = input("\nEnter a country exactly as it appears in the data: ").strip()
    if chosen_country in valid_countries:
        break
    print("Invalid country. Make sure spelling/capitalization match the dataset.")

# ---------------------------------
# ENCODE USER INPUT FOR MODEL
# ---------------------------------

user_input = {
    "type": chosen_type,
    "release_year": chosen_year,
    "rating": chosen_rating,
    "duration": chosen_duration,
    "country": chosen_country
}

encoded_input = {}
for col, val in user_input.items():
    if col in categorical_cols:   # type, rating, country, primary_genre
        encoded_input[col] = le_dict[col].transform([val])[0]
    else:
        encoded_input[col] = val  # numeric (year, duration)

user_df = pd.DataFrame([encoded_input])

# ---------------------------------
# PREDICT GENRE
# ---------------------------------

pred = rf.predict(user_df)[0]
predicted_genre = le_dict[target].inverse_transform([pred])[0]

# ---------------------------------
# OUTPUT RESULT
# ---------------------------------

print("\n===============================")
print("        USER INPUT SUMMARY")
print("===============================")
for k, v in user_input.items():
    print(f"{k}: {v}")

print("\nPredicted Primary Genre:", predicted_genre)
print("===============================\n")
