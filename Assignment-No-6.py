import pandas as pd

import pandas as pd

df = pd.read_csv(r"C:\Users\ADMIN\Desktop\python abhijit\PYTHON ASSIGNMENT\final_employee_data.csv")

print(df)
print("="*100)

print("--- Concept 1: Basic Grouping (Average Salary & Bonus per Department---)")
#Group by Department and compute the mean for numerical attributes
dept_means = df.groupby("Department")[["Salary", "Bonus"]]. mean()

#Formatting output values for clarity
dept_means_formatted = dept_means,round(2)
print(dept_means_formatted)
print("="*100)


print("--- Concept 2: Multi-Column Grouping (Average Salary by Dept & Role) ---")
#Multi-level grouping
role_hierarchy = (
    df.groupby(["Department", "Role"]) [["Salary", "Performance_Rating"]]
    .mean()
    .round(2)
)
print(role_hierarchy)


print("="*100)
print("--- Concept 3: Advanced Aggregation using .agg() ---")
# Dictionary mapping specific columns to desired statistical operations
agg_operations = {
    "Salary": ["mean", "min", "max"],
    "Bonus": "sum",
    "Projects_Completed": "sum",
    "Performance_Rating": "mean",
}

dept_summary = df.groupby("Department").agg(agg_operations).round(2)
print(dept_summary)
print("="*100)