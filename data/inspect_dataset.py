from pathlib import Path
import pickle

path = Path("datasets/dataset.pkl")

with path.open("rb") as f:
    dataset = pickle.load(f)

print(f"Dataset size: {len(dataset)}")

print("\nFirst entry:\n")
print(dataset[0])