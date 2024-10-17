# Importing necessary libraries
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# === 1. Loading the Datasets === #
challenges_df = pd.read_json('arc-agi_training-challenges.json')
solutions_df = pd.read_json('arc-agi_training-solutions.json')
print("Datasets successfully loaded.")

# === 2. Initial Data Exploration === #
print("\n--- Data Overview ---")
print(f"Number of challenges: {len(challenges_df)}")
print(f"Number of solutions: {len(solutions_df)}")

print("\nSample challenges data:")
print(challenges_df.head())

print("\nSample solutions data:")
print(solutions_df.head())

print("\nSummary statistics for challenges dataset:")
print(challenges_df.describe())

# === 3. Exploring the Structure of Challenges === #
print("\n--- Exploring Structure ---")
print("Columns in challenges dataset:", challenges_df.columns.tolist())

example_task = challenges_df.iloc[0]
print("\nSample challenge input:")
print(example_task['input'])
print("\nCorresponding ground truth output:")
print(example_task['output'])

missing_outputs = challenges_df[challenges_df['output'].isnull()]
if not missing_outputs.empty:
    print("\nWarning: Some challenges have missing outputs.")
else:
    print("\nAll challenges have corresponding outputs.")

# === 4. Analyzing Task Complexity === #
print("\n--- Task Complexity ---")
grid_sizes = challenges_df['input'].apply(lambda x: (len(x), len(x[0])) if x else (0, 0))
print("\nDistribution of input grid sizes:")
print(grid_sizes.value_counts())

unique_transformations = solutions_df['transformation'].nunique()
print(f"\nUnique transformations: {unique_transformations}")

# === 5. Validating Output Format === #
def validate_challenge_format(data):
    issues = []
    for idx, row in data.iterrows():
        outputs = row.get('output', [])
        if len(outputs) != 2:
            issues.append(f"Task {idx} lacks exactly 2 predicted outputs.")
    return issues

format_issues = validate_challenge_format(challenges_df)
if format_issues:
    print("Format validation issues:")
    for issue in format_issues:
        print(issue)
else:
    print("All tasks have valid output formats.")

# === 6. Generating a Sample Submission === #
def create_mock_submission(data):
    submission = {}
    for idx in range(len(data)):
        task_id = f"task_{idx}"
        submission[task_id] = [
            {'attempt_1': [[0, 0], [0, 0]]},
            {'attempt_2': [[1, 1], [1, 1]]}
        ]
    return submission

sample_submission = create_mock_submission(challenges_df)
submission_filename = 'sample_submission.json'
with open(submission_filename, 'w') as f:
    json.dump(sample_submission, f)

print(f"\nSample submission saved to {submission_filename}.")

with open(submission_filename, 'r') as f:
    saved_submission = json.load(f)
print("\nSaved submission preview:")
print(json.dumps(saved_submission, indent=2))

# === 7. Transformation Pattern Analysis === #
def extract_transformations(data):
    transformations = []
    for idx, row in data.iterrows():
        input_grid = row['input']
        output_grid = row['output']
        input_shape = (len(input_grid), len(input_grid[0]))
        output_shape = (len(output_grid), len(output_grid[0]))
        is_flipped = input_grid[::-1] == output_grid
        is_rotated = input_grid == [list(reversed(col)) for col in zip(*output_grid)]
        transformations.append({
            "task": idx,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "flipped": is_flipped,
            "rotated": is_rotated
        })
    return pd.DataFrame(transformations)

transformation_patterns = extract_transformations(challenges_df)
print("\nTransformation patterns summary:")
print(transformation_patterns.head())

# === 8. Correlation Between Input and Output Grid Sizes === #
transformation_patterns['input_size'] = transformation_patterns['input_shape'].apply(lambda x: x[0] * x[1])
transformation_patterns['output_size'] = transformation_patterns['output_shape'].apply(lambda x: x[0] * x[1])

plt.figure(figsize=(8, 6))
plt.scatter(transformation_patterns['input_size'], transformation_patterns['output_size'], alpha=0.5)
plt.title("Correlation Between Input and Output Grid Sizes")
plt.xlabel("Input Grid Size")
plt.ylabel("Output Grid Size")
plt.grid(True)
plt.show()

correlation = transformation_patterns[['input_size', 'output_size']].corr().iloc[0, 1]
print(f"\nCorrelation between input and output grid sizes: {correlation:.2f}")

# === 9. Modular Prediction Model === #
def modular_predictor(task_row):
    input_grid = task_row['input']
    if task_row['flipped']:
        return [input_grid[::-1], input_grid[::-1]]
    elif task_row['rotated']:
        rotated_grid = [list(reversed(col)) for col in zip(*input_grid)]
        return [rotated_grid, rotated_grid]
    elif task_row['input_shape'] != task_row['output_shape']:
        smaller_grid = [[1] * (len(input_grid[0]) // 2)] * (len(input_grid) // 2)
        return [smaller_grid, smaller_grid]
    else:
        return [input_grid, input_grid]

predictions = transformation_patterns.apply(modular_predictor, axis=1)
print("\nSample predictions:")
for i, pred in enumerate(predictions[:3]):
    print(f"Task {i} Predictions: {pred}")

# === 10. Generalization Techniques === #
def find_similar_tasks(task_row, patterns_df):
    return patterns_df[
        (patterns_df['input_shape'] == task_row['input_shape']) &
        (patterns_df['flipped'] == task_row['flipped']) &
        (patterns_df['rotated'] == task_row['rotated'])
    ]

def generalized_predictor(task_row, patterns_df):
    similar_tasks = find_similar_tasks(task_row, patterns_df)
    if not similar_tasks.empty:
        return [similar_tasks.iloc[0]['output'], similar_tasks.iloc[0]['output']]
    else:
        return [task_row['input'], task_row['input']]

generalized_predictions = transformation_patterns.apply(
    lambda row: generalized_predictor(row, transformation_patterns), axis=1
)

print("\nGeneralized predictions:")
for i, pred in enumerate(generalized_predictions[:3]):
    print(f"Task {i} Predictions: {pred}")
