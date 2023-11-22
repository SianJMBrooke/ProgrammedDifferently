# Import libraries
import warnings
from github import Github, GithubException
import pandas as pd
from ast import literal_eval
import time
from pygit_func_stats import pylint_test, github_ratelimit


# GitHub Credentials
user_name = ""
client_secret = ""
client_id = ""

g = Github(client_id, client_secret, per_page=100)

github_repos_df = pd.read_csv("GitHub_All_Repos_Gender.csv")

# All repos below the 99 Quantile of the number of files (is 42 for whole data)
ceiling_file_num = github_repos_df["Num Files"].quantile(0.99)
print("--- The 0.99 Quantile for Number of Files in Repo is:", ceiling_file_num, "---")

user_count = len(github_repos_df)

for i, row in github_repos_df.iterrows():
    print("\n", i + 1, "/", user_count, "Github Repository")

    py_file_list = literal_eval(row["Python Files"])
    num_py_files = len(py_file_list)
    print("The number of files is", num_py_files)
    repo_path = row["Repo Path"]

    pylint_output = []

    if num_py_files > ceiling_file_num:
        print("\t File ceiling exceeded.", num_py_files, ".py"s")
        continue

    for ix, py_file in enumerate(py_file_list):

        for _ in range(3):

            try:
                # Access python file using REST API
                py_data = g.get_repo(repo_path).get_contents(py_file)

                try:
                    # Decode and write python file
                    file_content = py_data.decoded_content.decode()
                    with open("python_file.py", "w") as python_file:
                        python_file.write(file_content)

                    # Run pylint over .py
                    py_file_results = pylint_test("python_file.py")
                    py_res = {py_file: py_file_results}

                    # Add to list
                    pylint_output.append(py_res)

                    print("\t", ix + 1, "/", num_py_files, "Pylinting Complete.")

                except AssertionError:
                    print("Assert Error")

                except Exception as e:
                    print(f"Error encountered: {e}")
                    continue

            except GithubException:
                github_ratelimit(g)

    # Save to DataFrame.
    github_repos_df.loc[i, "Pylint Scores"] = str(pylint_output)
    github_repos_df.to_csv("GitHub_Files_Data.csv", index=False)

print("Time Finished:", time.strftime("%l:%M%p %Z on %b %d, %Y"))
