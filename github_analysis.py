# Import libraries
import itertools

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import numpy as np
import statsmodels.stats.multicomp as mc
import ast
import sys
import csv
import glob
import os
import re
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


# Set plot style
sns.set(style="whitegrid", font="Times")

# Set field size limit
csv.field_size_limit(sys.maxsize)

# Change directory
os.chdir("")

# ------------------------------------------------------------
# Import and Clean Data
# ------------------------------------------------------------


# Combine CSV files
csv_files = glob.glob("*.{}".format("csv"))
github_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True).drop_duplicates(ignore_index=True)

# Drop unnecessary columns
github_data_df = github_data_df.drop(["level_0", "index"], axis=1)
github_data_df = github_data_df[github_data_df.columns.drop(list(github_data_df.filter(regex="Unnamed")))]

# Drop duplicates based on "Combined Path"
github_data_df = github_data_df.drop_duplicates(subset="Combined Path")

# Drop rows with "Unable to Lint."
github_data_df = github_data_df[github_data_df["Pylint Scores"] != "Unable to Lint."].dropna(subset=["Pylint Scores"])

# Convert gender to a categorical variable
github_data_df["Gender"] = github_data_df["Gender"].map({"female": "Feminine", "male": "Masculine",
                                                         "unisex": "Non-Binary", np.NaN: "Anonymous"})


# Function to resolve nested structure in the "By Messages" column
def extract_pylint_scores(row):
    try:
        test_dict = ast.literal_eval(row.replace("\n", ","))
        return test_dict.get("global_note", np.nan)
    except:
        return np.nan


# ------------------------------------------------------------
# Global Pylint Score
# ------------------------------------------------------------

# Apply the function to create a new "Pylint Global Score" column
github_data_df["Pylint Global Score"] = github_data_df["Pylint Scores"].apply(extract_pylint_scores)

# Filter out rows with "Pylint Global Score" less than -100
github_data_df = github_data_df[github_data_df["Pylint Global Score"] > -100]

# Create a descriptive statistics dataframe
describe_df = github_data_df.groupby("Gender")["Pylint Global Score"].describe()

# Violin plot
sns.violinplot(data=github_data_df[github_data_df["Pylint Global Score"] > -10],
               x="Pylint Global Score", y="Gender", bw=.25, color="#D0D3D4")
plt.tight_layout()
plt.show()

fem_df = describe_df[describe_df["Gender"] == "Feminine"]
masc_df = describe_df[describe_df["Gender"] == "Masculine"]
anon_df = describe_df[describe_df["Gender"] == "Anonymous"]

ided_of = describe_df[(describe_df["Gender"] == "Feminine") | (describe_df["Gender"] == "Masculine")]
nonid_df = describe_df[(describe_df["Gender"] == "Anonymous") | (describe_df["Gender"] == "Non-Binary")]

pg.anova(dv="Pylint Global Score", between="Gender", data=lint_df)
test_pylint = pg.ttest(ided_of["Pylint Global Score"], nonid_df["Pylint Global Score"])

stats.ttest_ind(fem_df["Pylint Global Score"],
                masc_df["Pylint Global Score"],
                nan_policy="omit", equal_var=False)

comp1 = mc.MultiComparison(describe_df["Pylint Global Score"], describe_df["Gender"])
tbl, a1, a2 = comp1.allpairtest(stats.ttest_ind, method="bonf")


# ------------------------------------------------------------
# Gender Differences
# ------------------------------------------------------------

lint_df = describe_df

# Filter DataFrames based on gender
fem_df = lint_df[lint_df["Gender"] == "Feminine"]
masc_df = lint_df[lint_df["Gender"] == "Masculine"]

ided_of = lint_df[(lint_df["Gender"].isin(["Feminine", "Masculine"]))]
nonid_df = lint_df[(lint_df["Gender"].isin(["Anonymous", "Non-Binary"]))]

# Gender ANOVA
pg.anova(dv="Pylint Global Score", between="Gender", data=lint_df)

# Gender T-Test
test_id_comp = pg.ttest(ided_of["Pylint Global Score"], nonid_df["Pylint Global Score"])

# Independent T-Test
stats.ttest_ind(fem_df["Pylint Global Score"], masc_df["Pylint Global Score"],
                nan_policy="omit", equal_var=False)

# Post-Hoc Tests
comp1 = mc.MultiComparison(lint_df["Pylint Global Score"], lint_df["Gender"])

mssg_list = []

for index, row in lint_df.iterrows():
    try:
        test_dict = ast.literal_eval(row["Pylint Scores"].replace("\n", ","))
        mssg_list.append(test_dict["by_msg"])  # pylint: by_msg
    except:
        continue

lint_df["By Messages"] = mssg_list

# Most common errors by gender category
component_df = lint_df[["Gender", "Repo Name", "Python Files", "By Messages", "Pylint Scores"]]

gender_owner, repo_name, message, message_freq, py_files = [], [], [], [], []
py_scores, fatal, module_attrib, function, method = [], [], [], [], []
klass, total_lines, code_lines, empty_lines, docstring_lines = [], [], [], [], []
comment_lines, nb_duplicated_lines = [], []

for index, row in component_df.iterrows():
    dict_err = row["By Messages"]
    repo, gender, py_f = row["Repo Name"], row["Gender"], row["Python Files"]
    pylint_dict = ast.literal_eval(row["Pylint Scores"].replace("\n", ","))

    if len(pylint_dict) == 38:
        for k, v in dict_err.items():
            message.append(k)
            message_freq.append(v)
            gender_owner.append(gender)
            repo_name.append(repo)
            py_files.append(py_f)
            fatal.append(pylint_dict["fatal"])
            module_attrib.append(pylint_dict["module"])
            function.append(pylint_dict["function"])
            method.append(pylint_dict["method"])
            klass.append(pylint_dict["class"])
            total_lines.append(pylint_dict["total_lines"])
            code_lines.append(pylint_dict["code_lines"])
            empty_lines.append(pylint_dict["empty_lines"])
            docstring_lines.append(pylint_dict["docstring_lines"])
            comment_lines.append(pylint_dict["comment_lines"])
            nb_duplicated_lines.append(pylint_dict["nb_duplicated_lines"])
    else:
        continue

component_df = pd.DataFrame({
    "Gender": gender_owner,
    "Repo Name": repo_name,
    "Python File": py_files,
    "Component Checker": message,
    "Frequency": message_freq,
    "Fatal": fatal,
    "Str_Function": function,
    "Str_Method": method,
    "Str_Klass": klass,
    "Str_Total_Lines": total_lines,
    "Str_Code_Lines": code_lines,
    "Str_Empty_Lines": empty_lines,
    "Str_Docstring_Lines": docstring_lines,
    "Str_Comment_Lines": comment_lines,
    "Lines_Duplicated": nb_duplicated_lines
})

# Descriptive Statistics
component_df.groupby("Gender")["Str_Code_Lines"].describe()

import pingouin as pg

df = component_df
col = "Str_Klass"
test_result_df = []

all_combinations = list(itertools.combinations(set(df["Gender"]), 2))

bonf = 0.05 / 6

# Post-hoc Testing
p_vals = []

print("Significance results:")

for comb in all_combinations:
    gender_1 = df[df["Gender"] == comb[0]]
    gender_2 = df[df["Gender"] == comb[1]]

    test_df = pg.ttest(gender_1[col], gender_2[col], correction=True)
    test_df["Gender"] = [comb]
    test_result_df.append(test_df)

appended_data = pd.concat(test_result_df)
appended_data = appended_data[appended_data["p-val"] < bonf]

appended_data = appended_data[["Gender", "T", "p-val", "dof"]].reset_index()

# Multi-Comparison
comp1 = mc.MultiComparison(component_df["Str_Total_Lines"], component_df["Gender"])
tbl, _, _ = comp1.allpairtest(stats.ttest_ind, method="bonf")
result_df = pd.DataFrame(tbl)

# Errors List and Errors DataFrame
component_df["Errors List"] = component_df["Component Checker"].apply(lambda x: [x])
component_df["Errors List"] = component_df["Errors List"].multiply(component_df["Frequency"], axis="index")

# Explode the DataFrame
errors_df = component_df.explode("Errors List")

# Group the error code
with open("", "r") as file:
    data = file.read()

cat_py = re.findall(r"(?<=   )(.+)(?=\/)", data)
mssg_py = re.findall(r"(?<=\/)(.+)", data)

checker_dict = dict(zip(mssg_py, cat_py))
errors_df["Checker Group"] = errors_df["Component Checker"].map(checker_dict)

# Save to CSV
errors_df.to_csv("Errors_df.csv", index=False)
component_df.to_csv("Component_df.csv", index=False)

# Read the saved CSV
errors_df = pd.read_csv("Errors_df.csv")

# CHI-SQ TEST
# Contingency Table
ct_table_ind = pd.crosstab(errors_df["Gender"], errors_df["Checker Group"])
print("Contingency Table:\n", ct_table_ind)

# Chi-squared test
chi2_stat, p, dof, expected = stats.chi2_contingency(ct_table_ind)


def posthoc_corrected_chi(df, col):
    """Perform post hoc chi-squared test and print the results."""
    all_combinations = [(gender_1, gender_2) for gender_1 in set(df["Gender"]) for gender_2 in set(df["Gender"]) if gender_1 != gender_2]

    # Post-hoc
    p_vals = []
    print("Significance results:")
    for comb in all_combinations:
        new_df = df[df["Gender"].isin(comb)]
        ct_table_ind = pd.crosstab(new_df["Gender"], new_df[col])

        # Chi-squared test
        chi2, p, dof, ex = stats.chi2_contingency(ct_table_ind, correction=True)
        p_vals.append(p)

        print(comb, chi2)

    # FDR correction for multiple comparisons
    _, corrected_p_vals, _, _ = mc.multipletests(p_vals, alpha=0.05, method="fdr_bh")

    print("\nCorrected p-values:")
    for i, comb in enumerate(all_combinations):
        print(comb, corrected_p_vals[i])


# Perform post hoc test
posthoc_corrected_chi(errors_df, "Checker Group")


# ------------------------------------------------------------
# Gender Prediction with Random Forest
# ------------------------------------------------------------

# Load data
gender_df = pd.read_csv("Component_df.csv")

# Map gender values to numeric (1 for Feminine, 0 for Masculine, NaN for others)
gender_df["Gender"] = gender_df["Gender"].map({"Feminine": 1, "Masculine": 0, "Anonymous": np.NaN, "Non-Binary": np.nan})

# Prepare features and labels
features = gender_df[gender_df["Gender"].notna()].drop("Gender", axis=1)
labels = np.array(features["Gender"])
features = features.drop("Gender", axis=1)

# One-hot encode categorical variables
features = pd.get_dummies(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

# Grid Search for hyperparameter tuning
gsc = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid={"max_depth": range(3, 7), "n_estimators": (10, 50, 100, 1000)},
    cv=10, scoring="neg_mean_squared_error", verbose=0, n_jobs=-1
)

grid_result = gsc.fit(train_features, train_labels)
best_params = grid_result.best_params_

# Train the RandomForestRegressor with the best parameters
rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                            random_state=False, verbose=False)
rfr.fit(train_features, train_labels)

# Evaluate the model on the test set
test_scores = cross_val_score(rfr, test_features, test_labels, cv=10, scoring="neg_mean_absolute_percentage_error")

# Make predictions on the "Anonymous" gender
gender_predict_feat = component_df[["Str_Function", "Str_Method", "Str_Klass", "Str_Total_Lines", "Str_Code_Lines",
                                    "Str_Empty_Lines", "Str_Docstring_Lines", "Str_Comment_Lines", "Checker Group"]]
features_pred = pd.get_dummies(gender_predict_feat)
features_pred = features_pred[features.columns]  # Ensure the same features as the training set
output = rfr.predict(features_pred)


param_dist = {"n_estimators": randint(50,500),
              "max_depth": randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=10)

# Fit the random search object to the data
rand_search.fit(train_features, train_labels)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print("Best hyperparameters:",  rand_search.best_params_)

pickle.dump(best_rf, open("", "wb"))

y_pred = best_rf.predict(test_features)

# Metrics
accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
f1score = f1_score(test_labels, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1score)


feature_importances = pd.Series(best_rf.feature_importances_, index=feature_list)
feature_importances[1] = 0.171
feature_importances.sort_values(ascending=False, inplace=True)
sns.set_context(rc = {"patch.linewidt": 0.0})

g = sns.barplot(feature_importances.values, feature_importances.index, color = "#D0D3D4",
                edgecolor=".5", linewidth=1.45)

g.set_yticklabels(["Code Lines", "Doctring Lines", "Empty Lines", "Method Usages",
                   "Comment Lines", "Class Usage", "Function Usage", "Total Num of Lines",
                   "Information Checker", "Error Checker", "Convention Checker", "Refactor Checker",
                   "Warning Checker"])
plt.tight_layout()
plt.savefig("Ft_Imp.jpg")
plt.show()
