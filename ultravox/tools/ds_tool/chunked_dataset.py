# mypy: ignore-errors
import fnmatch
import json
import math
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import datasets
import huggingface_hub
from datasets.data_files import sanitize_patterns
from datasets.info import DatasetInfo
from datasets.info import DatasetInfosDict
from datasets.naming import _split_re
from datasets.splits import SplitDict
from datasets.splits import SplitInfo
from datasets.utils import logging
from datasets.utils.metadata import MetadataConfigs
from datasets.utils.py_utils import asdict
from datasets.utils.py_utils import glob_pattern_to_regex
from datasets.utils.py_utils import string_to_dict
from huggingface_hub.hf_api import RepoFile

logger = logging.get_logger(__name__)
PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED = (
    "data/{split}-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.parquet"
)


class ChunkedDataset(datasets.Dataset):
    @classmethod
    def from_dataset(cls, dataset):
        """
        Create a ChunkedDataset from an existing Dataset.
        """
        obj = cls(dataset.data.table)
        obj.__dict__.update(dataset.__dict__)
        return obj

    def push_to_hub(
        self,
        repo_id: str,
        config_name: str = "default",
        set_default: Optional[bool] = None,
        split: Optional[str] = None,
        data_dir: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        branch="deprecated",
        create_pr: Optional[bool] = False,
        max_shard_size: Optional[Union[int, str]] = None,
        num_shards: Optional[int] = None,
        embed_external_files: bool = True,
    ) -> huggingface_hub.CommitInfo:
        """
        This overrides the push_to_hub method to work with chunked uploads. The old method assumed
        each write was supposed to override the existing split data in the README, but this method will append to the
        existing split values in the README (ie download_size, num_examples, etc).
        """
        if config_name == "data":
            raise ValueError(
                "`config_name` cannot be 'data'. Please, choose another name for configuration."
            )

        if max_shard_size is not None and num_shards is not None:
            raise ValueError(
                "Failed to push_to_hub: please specify either max_shard_size or num_shards, but not both."
            )

        if split is None:
            split = str(self.split) if self.split is not None else "train"

        if not re.match(_split_re, split):
            raise ValueError(
                f"Split name should match '{_split_re}' but got '{split}'."
            )

        if branch != "deprecated":
            warnings.warn(
                "'branch' was deprecated in favor of 'revision' in version 2.15.0 and will be removed in 3.0.0.\n"
                f"You can remove this warning by passing 'revision={branch}' instead.",
                FutureWarning,
            )
            revision = branch

        api = huggingface_hub.HfApi(endpoint=datasets.config.HF_ENDPOINT, token=token)

        repo_url = api.create_repo(
            repo_id,
            token=token,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        if revision is not None:
            api.create_branch(
                repo_id,
                branch=revision,
                token=token,
                repo_type="dataset",
                exist_ok=True,
            )

        if not data_dir:
            data_dir = (
                config_name if config_name != "default" else "data"
            )  # for backward compatibility

        additions, uploaded_size, dataset_nbytes = self._push_parquet_shards_to_hub(
            repo_id=repo_id,
            data_dir=data_dir,
            split=split,
            token=token,
            revision=revision,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            create_pr=create_pr,
            embed_external_files=embed_external_files,
        )

        # Check if the repo already has a README.md and/or a dataset_infos.json to update them with the new split info (size and pattern)
        # and delete old split shards (if they exist)
        repo_with_dataset_card, repo_with_dataset_infos = False, False
        deletions, deleted_size = [], 0
        repo_splits = []  # use a list to keep the order of the splits
        repo_files_to_add = [addition.path_in_repo for addition in additions]
        for repo_file in api.list_repo_tree(
            repo_id=repo_id,
            revision=revision,
            repo_type="dataset",
            token=token,
            recursive=True,
        ):
            if not isinstance(repo_file, RepoFile):
                continue
            if repo_file.rfilename == datasets.config.REPOCARD_FILENAME:
                repo_with_dataset_card = True
            elif repo_file.rfilename == datasets.config.DATASETDICT_INFOS_FILENAME:
                repo_with_dataset_infos = True
            elif (
                repo_file.rfilename.startswith(f"{data_dir}/{split}-")
                and repo_file.rfilename not in repo_files_to_add
            ):
                deletions.append(
                    huggingface_hub.CommitOperationDelete(
                        path_in_repo=repo_file.rfilename
                    )
                )
                deleted_size += repo_file.size
            elif fnmatch.fnmatch(
                repo_file.rfilename,
                PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED.replace(
                    "{split}", "*"
                ),
            ):
                repo_split = string_to_dict(
                    repo_file.rfilename,
                    glob_pattern_to_regex(
                        PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED
                    ),
                )["split"]
                if repo_split not in repo_splits:
                    repo_splits.append(repo_split)

        organization, dataset_name = (
            repo_id.split("/") if "/" in repo_id else (None, repo_id)
        )
        info_to_dump = self.info.copy()
        info_to_dump.download_checksums = None
        info_to_dump.download_size = uploaded_size
        info_to_dump.dataset_size = dataset_nbytes
        info_to_dump.size_in_bytes = uploaded_size + dataset_nbytes
        info_to_dump.config_name = config_name
        info_to_dump.splits = SplitDict(
            {
                split: SplitInfo(
                    split,
                    num_bytes=dataset_nbytes,
                    num_examples=len(self),
                    dataset_name=dataset_name,
                )
            }
        )
        # get the info from the README to update them
        if repo_with_dataset_card:
            dataset_card_path = api.hf_hub_download(
                repo_id,
                datasets.config.REPOCARD_FILENAME,
                repo_type="dataset",
                revision=revision,
            )
            dataset_card = huggingface_hub.DatasetCard.load(Path(dataset_card_path))
            dataset_card_data = dataset_card.data
            metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
            dataset_infos: DatasetInfosDict = DatasetInfosDict.from_dataset_card_data(
                dataset_card_data
            )
            if dataset_infos and config_name in dataset_infos:
                repo_info = dataset_infos[config_name]
            else:
                repo_info = None
        # get the deprecated dataset_infos.json to update them
        elif repo_with_dataset_infos:
            dataset_card = None
            dataset_card_data = huggingface_hub.DatasetCardData()
            metadata_configs = MetadataConfigs()
            dataset_infos_path = api.hf_hub_download(
                repo_id,
                datasets.config.DATASETDICT_INFOS_FILENAME,
                repo_type="dataset",
                revision=revision,
            )
            with open(dataset_infos_path, encoding="utf-8") as f:
                dataset_infos: dict = json.load(f)
                dataset_info = (
                    dataset_infos.get(config_name, None) if dataset_infos else None
                )
                repo_info = (
                    DatasetInfo.from_dict(dataset_info) if dataset_info else None
                )
        else:
            dataset_card = None
            dataset_card_data = huggingface_hub.DatasetCardData()
            metadata_configs = MetadataConfigs()
            repo_info = None
        # Update the total info to dump from existing info.
        if repo_info is not None:
            logger.info("Updating downloaded metadata with the new split.")
            # MODIFIED:
            # New Addition:
            # Keep the old split info to update the new split info
            old_split = repo_info.splits.get(split, SplitInfo())
            # MODIFIED:
            # Old:
            # if repo_info.splits and list(repo_info.splits) != [split]:
            if repo_info.splits:
                if self._info.features != repo_info.features:
                    raise ValueError(
                        f"Features of the new split don't match the features of the existing splits on the hub: {self._info.features} != {repo_info.features}"
                    )

                repo_info.download_checksums = None
                repo_info.download_size = (repo_info.download_size or 0) + uploaded_size
                repo_info.dataset_size = (repo_info.dataset_size or 0) + dataset_nbytes
                repo_info.size_in_bytes = (
                    repo_info.download_size + repo_info.dataset_size
                )
                repo_info.splits.pop(split, None)
                # MODIFIED:
                # Old:
                # repo_info.splits[split] = SplitInfo(
                #     split, num_bytes=dataset_nbytes, num_examples=len(self), dataset_name=dataset_name
                # )
                repo_info.splits[split] = SplitInfo(
                    split,
                    num_bytes=old_split.num_bytes + dataset_nbytes,
                    num_examples=old_split.num_examples + len(self),
                    dataset_name=dataset_name,
                )
                info_to_dump = repo_info
        # create the metadata configs if it was uploaded with push_to_hub before metadata configs existed
        if not metadata_configs and repo_splits:
            default_metadata_configs_to_dump = {
                "data_files": [
                    {"split": split, "path": f"data/{split}-*"} for split in repo_splits
                ]
            }
            MetadataConfigs(
                {"default": default_metadata_configs_to_dump}
            ).to_dataset_card_data(dataset_card_data)
        # update the metadata configs
        if config_name in metadata_configs:
            metadata_config = metadata_configs[config_name]
            if "data_files" in metadata_config:
                data_files_to_dump = sanitize_patterns(metadata_config["data_files"])
            else:
                data_files_to_dump = {}
            # add the new split
            # MODIFIED:
            # Old:
            # data_files_to_dump[split] = [f"{data_dir}/{split}-*"]
            data_files_to_dump[split] = [f"{config_name}/{split}/**"]
            metadata_config_to_dump = {
                "data_files": [
                    {
                        "split": _split,
                        "path": _pattern[0] if len(_pattern) == 1 else _pattern,
                    }
                    for _split, _pattern in data_files_to_dump.items()
                ]
            }
        else:
            # MODIFIED:
            # Old:
            # metadata_config_to_dump = {"data_files": [{"split": split, "path": f"{data_dir}/{split}-*"}]}
            metadata_config_to_dump = {
                "data_files": [{"split": split, "path": f"{config_name}/{split}/**"}]
            }

        if set_default and config_name != "default":
            if metadata_configs:
                default_config_name = metadata_configs.get_default_config_name()
                if default_config_name == "default":
                    raise ValueError(
                        "There exists a configuration named 'default'. To set a different configuration as default, "
                        "rename the 'default' one first."
                    )
                else:
                    _ = metadata_configs[default_config_name].pop("default")
            metadata_config_to_dump["default"] = True
        # push to the deprecated dataset_infos.json
        if repo_with_dataset_infos:
            dataset_infos_path = api.hf_hub_download(
                repo_id,
                datasets.config.DATASETDICT_INFOS_FILENAME,
                repo_type="dataset",
                revision=revision,
            )
            with open(dataset_infos_path, encoding="utf-8") as f:
                dataset_infos: dict = json.load(f)
            dataset_infos[config_name] = asdict(info_to_dump)
            buffer = BytesIO()
            buffer.write(json.dumps(dataset_infos, indent=4).encode("utf-8"))
            additions.append(
                huggingface_hub.CommitOperationAdd(
                    path_in_repo=datasets.config.DATASETDICT_INFOS_FILENAME,
                    path_or_fileobj=buffer,
                )
            )
        # push to README
        DatasetInfosDict({config_name: info_to_dump}).to_dataset_card_data(
            dataset_card_data
        )
        MetadataConfigs({config_name: metadata_config_to_dump}).to_dataset_card_data(
            dataset_card_data
        )
        dataset_card = (
            huggingface_hub.DatasetCard(f"---\n{dataset_card_data}\n---\n")
            if dataset_card is None
            else dataset_card
        )
        additions.append(
            huggingface_hub.CommitOperationAdd(
                path_in_repo=datasets.config.REPOCARD_FILENAME,
                path_or_fileobj=str(dataset_card).encode(),
            )
        )

        commit_message = (
            commit_message if commit_message is not None else "Upload dataset"
        )
        if len(additions) <= datasets.config.UPLOADS_MAX_NUMBER_PER_COMMIT:
            commit_info = api.create_commit(
                repo_id,
                operations=additions + deletions,
                commit_message=commit_message,
                commit_description=commit_description,
                token=token,
                repo_type="dataset",
                revision=revision,
                create_pr=create_pr,
            )
        else:
            logger.info(
                f"Number of files to upload is larger than {datasets.config.UPLOADS_MAX_NUMBER_PER_COMMIT}. Splitting the push into multiple commits."
            )
            num_commits = math.ceil(
                len(additions) / datasets.config.UPLOADS_MAX_NUMBER_PER_COMMIT
            )
            for i in range(0, num_commits):
                operations = additions[
                    i
                    * datasets.config.UPLOADS_MAX_NUMBER_PER_COMMIT : (i + 1)
                    * datasets.config.UPLOADS_MAX_NUMBER_PER_COMMIT
                ] + (deletions if i == 0 else [])
                commit_info = api.create_commit(
                    repo_id,
                    operations=operations,
                    commit_message=commit_message
                    + f" (part {i:05d}-of-{num_commits:05d})",
                    commit_description=commit_description,
                    token=token,
                    repo_type="dataset",
                    revision=revision,
                    create_pr=create_pr,
                )
                logger.info(
                    f"Commit #{i+1} completed"
                    + (
                        f" (still {num_commits - i - 1} to go)"
                        if num_commits - i - 1
                        else ""
                    )
                    + "."
                )
        return commit_info


# Function to convert Dataset to ChunkedDataset
def convert_to_chunked_dataset(data) -> ChunkedDataset:
    return (
        ChunkedDataset.from_dataset(data)
        if isinstance(data, datasets.Dataset)
        else data
    )
