import pandas as pd
import numpy as np
import datetime
import time
from github import Github, RateLimitExceededException
from ShareCode.pygit_func import github_ratelimit

# Original Url: https://github.com/tue-mdse/genderComputer
from genderComputer import GenderComputer


def extract_python_files(contents):
    # Function to extract Python files from repository contents
    python_list = []

    while contents:
        file_content = contents.pop(0)
        if file_content.path.endswith(".py") and "__" not in file_content.path:
            python_list.append(file_content.path)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))

    return python_list


def process_repo(repo, gc, already_inc, fem_dp, repo_sample):
    start_rate_limit = g.get_rate_limit().core

    # Two calls to the API to get the number of contributors
    contrib = repo.get_contributors(anon=True)
    num_contrib = contrib.totalCount
    files = repo.get_contents("")
    num_files = len(list(files))
    owner_type = repo.owner.type
    repo_name = repo.name

    login = repo.owner.login
    name = repo.owner.name
    local = repo.owner.location

    if ((already_inc["Repo Name"] == repo_name) & (already_inc["Repo Owner Login"] == login)).any():
        print("\n\tDatapoint already exists in dataset.")
        return fem_dp

    # One contributor and less than 50 files.
    if num_contrib == 1 and owner_type == "User":
        try:
            gender_name = gc.resolveGender(name, local)
            gender_login = gc.resolveGender(login, local)

            if gender_name or gender_login == "female":
                fem_dp += 1
                print("\n\t{0} Feminine Data Point Identified.".format(fem_dp))

                repo_dict = {"Repo Name": repo_name, "Repo Owner Login": login, "Repo Owner Name": name,
                             "Repo Owner ID": repo.owner.id, "Repo Owner Gender (Name)": gender_name,
                             "Repo Owner Gender (Login)": gender_login, "Repo Owner Location": local,
                             "Repo Owner Type": repo.owner.type, "Repo Owner Bio": repo.owner.bio,
                             "Repo Owner Email": repo.owner.email, "Repo Owner Collaborators": repo.owner.collaborators,
                             "Repo Owner Followers": repo.owner.followers, "Repo Owner Following": repo.owner.following,
                             "Repo Created At": repo.created_at.strftime("%Y-%m-%d"),
                             "Repo Updated At": repo.updated_at.strftime("%Y-%m-%d"),
                             "Repo Description": repo.description, "Repo Language": repo.language,
                             "Repo Is Fork": repo.fork, "Repo Forks Counts": repo.forks,
                             "Repo Organization": repo.organization, "Repo Labels": [i.name for i in repo.get_labels()],
                             "Num Contributors": num_contrib, "Contributors": contrib,
                             "Collaborators": repo.owner.collaborators, "Num Files": num_files,
                             "Num Commits": repo.get_commits().totalCount, "Contents": files,
                             "Python Files": extract_python_files(files), "Pylint Scores": np.nan}

                repo_sample.append(repo_dict)
                save_repo_data(repo_sample)

        except:
            pass

    return fem_dp


def save_repo_data(repo_sample):
    temp_df = pd.DataFrame(repo_sample)
    temp_df.to_csv("GitHub_Data_{0}.csv".format(start_date.strftime("%Y-%m-%d")), index=False)
    print("\n\t Python GitHub Repos Collected:", len(repo_sample))


# Initialize GitHub API instance
user_name = ""
client_secret = ""
client_id = ""
g = Github(client_id, client_secret, per_page=100)

# Initialize GenderComputer
gc = GenderComputer()

# Initialize date variables
start_date = datetime.datetime.strptime("2019-01-01", "%Y-%m-%d")
days_count = (datetime.datetime.now() - start_date).days

# Load existing data
already_inc = pd.read_csv("Already_Included.csv", index_col="Unnamed: 0")
already_inc = already_inc.drop_duplicates(subset=["Repo Name", "Repo Owner Login"])

# Initialize feminine data point counter
fem_dp = 0

# Initialize repo sample
repos_output = []
repo_sample = []

# Main loop
for dc in range(days_count):
    date_from = start_date + datetime.timedelta(days=dc)
    date_to = date_from + datetime.timedelta(days=1)

    print("\n---------------------------------------------------------\n", dc,
          "Querying Date Range:", str(date_from)[:10], "to", str(date_to)[:10])

    query = f"language:python forks:10..250 size:1000..1000000 stars:10..250 fork:false created:{str(date_from)[:10]}..{str(date_to)[:10]}"

    try:
        repositories = g.search_repositories(query=query)

        for repo in repositories:
            fem_dp = process_repo(repo, gc, already_inc, fem_dp, repo_sample)

    except RateLimitExceededException:
        github_ratelimit(g)

print("Time Finished:", time.strftime("%l:%M%p %Z on %b %d, %Y"))
