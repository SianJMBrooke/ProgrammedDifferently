from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import pandas as pd
import os
import glob

# Original Url: https://github.com/tue-mdse/genderComputer
from genderComputer import GenderComputer

# Set the working directory
os.chdir("/Users/BrookeSJ/PycharmProjects/Leverhulme/Final Data/")

# Load all CSV files into a single DataFrame
all_data_files = glob.glob("*.csv")
github_repos_df = pd.concat([pd.read_csv(file, index_col=None, header=0, engine="python") for file in all_data_files])

# Drop unnecessary columns and duplicates
github_repos_df = github_repos_df.drop(["Unnamed: 0", "Unnamed: 0.1", "level_0"], axis=1).drop_duplicates().reset_index(drop=True)

# Display user count
count_users = len(github_repos_df)
print("User Count:", count_users)

# Define a dictionary for text replacement
text_leet = {"8": "B", "5": "S", "0": "O", "|": "l", "1": "l", "7": "T",
             "4": "A", "£": "E", "$": "S", "€": "E", "¥": "Y", "3": "E",
             ".": " ", ",": " ", "@": " ", "-": " ", "_": ""}
# Apply text replacement to specified columns
for col in ["Repo Owner Email", "Repo Owner Login"]:
    for k, v in text_leet.items():
        github_repos_df[col] = github_repos_df[col].str.replace(k, v)

# Clean email address prefix for 3rd name source
github_repos_df["Repo Owner Email Prefix"] = github_repos_df["Repo Owner Email"].str.replace(r"(\d+)(?=@)", "", regex=True)
github_repos_df["Repo Owner Email Prefix"] = github_repos_df["Repo Owner Email Prefix"].str.extract(r"(\S*)(?=@)")

# Initialize GenderComputer
gc = GenderComputer()

# Clean data from reading in
github_repos_df = github_repos_df[github_repos_df["Repo Name"] != "/appengine/db.py""]
github_repos_df = github_repos_df.loc[:, ~github_repos_df.columns.str.contains("^Unnamed")]

# Initialize geopy
geopy = Nominatim(user_agent="http")

# Iterate over rows and process gender information
for i, row in github_repos_df.iterrows():
    print("[Gender] Processing User:", i + 1, "/", count_users)

    # Clean Location of Users
    if pd.notna(row["Repo Owner Location"]):
        try:
            geocode = RateLimiter(geopy.geocode, min_delay_seconds=1)
            github_repos_df.at[i, "Repo Owner Location"] = str(geocode(row["Repo Owner Location"])[0])
        except:
            pass

    # Get Gender from Name, if possible.
    for col in ["Repo Owner Name", "Repo Owner Login", "Repo Owner Email Prefix"]:
        try:
            github_repos_df.at[i, f"Repo Owner Gender ({col})"] = gc.resolveGender(row[col], row["Repo Owner Location"])
        except:
            pass

    # Save as we go
    github_repos_df.to_csv("GitHub_All_Repos_Gender.csv", index=False)

# Read the saved CSV
github_repos_df = pd.read_csv("GitHub_All_Repos_Gender.csv")

# Display gender sample
gender_columns = ["Repo Owner Gender (Name)", "Repo Owner Gender (Login)", "Repo Owner Gender (Email Prefix)"]
gender_sample = github_repos_df[gender_columns].apply(lambda x: x.value_counts(dropna=False)).T.rename_axis("Gender").reset_index(name="Frequency")
gender_sample.to_csv("GitHub_Repos_Gender_Sample.csv", index=False)

# Create a "Gender ID" column
github_repos_df["Gender ID"] = github_repos_df["Repo Owner Gender (Name)"].combine_first(github_repos_df["Repo Owner Gender (Login)"])

# Iterate over rows and process gender information using email prefix
for i, row in github_repos_df.iterrows():
    if pd.notna(row["Repo Owner Email Prefix"]):
        try:
            print(row["Repo Owner Email Prefix"])
            github_repos_df.at[i, "Repo Owner Gender (Email Prefix)"] = gc.resolveGender(row["Repo Owner Email Prefix"], row["Repo Owner Location"])
        except:
            pass
