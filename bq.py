from datasets import load_dataset

# Load the dataset
fbq = load_dataset("fixie-ai/boolq-audio", split="train")
gbq = load_dataset("google/boolq", split="train")


# Define the function to add the new column
def add_passage(example):
    example["passage"] = gbq[example["idx"]]["passage"]
    return example


# Apply the function to the dataset
nbq = fbq.map(add_passage)

# Check the result
print(nbq)
print(nbq[0])
nbq.to_parquet("new-boolq-audio.parquet")
