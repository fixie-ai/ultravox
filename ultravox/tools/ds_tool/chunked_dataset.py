from datasets import Dataset, DatasetDict

class ChunkedDataset(Dataset):
    @classmethod
    def from_dataset(cls, dataset):
        """
        Create a ChunkedDataset from an existing Dataset.
        """
        obj = cls(dataset.data.table)
        obj.__dict__.update(dataset.__dict__)
        return obj

    def push_to_hub(self, *args, **kwargs):
        # Temporarily disable readme update
        original_create_readme = kwargs.get('create_readme', True)
        kwargs['create_readme'] = False
        
        # Call the original push_to_hub method
        result = super().push_to_hub(*args, **kwargs)
        
        # Restore the original create_readme value
        kwargs['create_readme'] = original_create_readme
        
        return result

# Function to convert DatasetDict to dict of ChunkedDatasets
def convert_to_chunked_dataset(data_dict) -> DatasetDict:
    return {k: ChunkedDataset.from_dataset(v) if isinstance(v, Dataset) else v 
            for k, v in data_dict.items()}