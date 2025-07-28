"""
This script trains a linear regression model to map from the embeddings of a small model to the embeddings of a large model.
It then pushes the model to the HF hub.

The idea is that training a larger model is harder, so if we can find a mapping between the embeddings of a smaller model and a larger model,
we can merge that into an adapter trained for a smaller model and get a larger model for free.


See projector_combine_tool.py for how the projectors are combined.


The method works by learning a mapping between token embeddings of different sized models.
Each token in the vocabulary (e.g. 'hello', 'user', etc.) has an embedding vector in both models:
- v1 = small_model('hello') -> embedding vector in small model 
- v2 = large_model('hello') -> corresponding embedding vector in large model

We train a projection model s2b (small-to-big) to map v1 to v2:
s2b(v1) ≈ v2

This is done for all tokens in the vocabulary. The quality of the mapping is measured
by the cosine similarity between the projected and target vectors:
quality = cosine_similarity(s2b(small_embeddings), large_embeddings)

The projection can be used in two ways (the first one is the one we use):
1. Project a small model adapter to a large model:
    8B_adapter + s2b -> 70B_adapter

2. Train on large model then project down:
    Train 70B_adapter -> Apply b2s (big-to-small) -> Get 8B_adapter

The projection matrices handle the dimension differences between models,
e.g. mapping between 4096D (8B) and 8192D (70B) embedding spaces.
"""

import dataclasses
import json
import os
from typing import Any, Callable, Dict, List, Optional

import huggingface_hub
import safetensors.torch
import simple_parsing
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transformers

# Example usage:
# HF_TOKEN=$HF_WRITE_TOKEN  python -m ultravox.tools.projector_regression_tool -s meta-llama/Llama-3.2-1B-Instruct -b meta-llama/Llama-3.1-8B-Instruct -p fixie-ai/llama-3.2-1b-8b-projection
# HF_TOKEN=$HF_WRITE_TOKEN  python -m ultravox.tools.projector_regression_tool -s meta-llama/Llama-3.1-8B-Instruct -b meta-llama/Llama-3.1-70B-Instruct -p fixie-ai/llama-3.1-8b-70b-projection


@dataclasses.dataclass
class ProjectorRegressionArgs:
    small_model: str = simple_parsing.field(alias="-s")
    big_model: str = simple_parsing.field(alias="-b")
    projector_repo: str = simple_parsing.field(alias="-p")
    s2b_model_name: str = simple_parsing.field(alias="-s2b", default="projection_s2b")
    b2s_model_name: str = simple_parsing.field(alias="-b2s", default="projection_b2s")
    both_directions: bool = simple_parsing.field(alias="-b", default=True)
    debug: bool = simple_parsing.field(alias="-d", default=False)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CosineLoss(nn.Module):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - torch.nn.functional.cosine_similarity(input, target, dim=1).mean()


class PushableWrapper(transformers.modeling_utils.PushToHubMixin):
    """
    A wrapper around a model that lets it be pushed to the HF hub.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.metadata = metadata

    def save_pretrained(self, working_dir: str, *args, **kwargs):
        safetensors.torch.save_file(
            self.model.state_dict(),
            os.path.join(working_dir, f"{self.model_name}.safetensors"),
        )
        if self.metadata:
            with open(os.path.join(working_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f)


def load_embeds(repo: str, weights_key: str = "model.embed_tokens.weight"):
    """
    Loads the embeddings from the given repo.
    It automatically handles both the case of the model.safetensors file (e.g. 1B) and the model-00001-of-xxxxx.safetensors file (8B, 70B, etc.)

    Args:
        repo: The repo to load the embeddings from.
        weights_key: The key to load the embeddings from.

    Returns:
        The embeddings.
    """

    try:
        local_path = huggingface_hub.hf_hub_download(repo, "model.safetensors")
    except huggingface_hub.utils.EntryNotFoundError:
        local_path = huggingface_hub.hf_hub_download(
            repo, "model.safetensors.index.json"
        )
        weights_index = json.load(open(local_path))
        file_path = weights_index["weight_map"][weights_key]
        local_path = huggingface_hub.hf_hub_download(repo, file_path)

    weights = safetensors.torch.load_file(local_path)
    return weights[weights_key]


def train_linear_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    Xt: Optional[torch.Tensor] = None,
    yt: Optional[torch.Tensor] = None,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    num_layers: int = 1,
    epochs: int = 100,
    batch_size: int = 64,
    log_every: int = 100,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
    early_stop_iters: int = 5,
) -> nn.Sequential:
    """Trains a linear regression model to map from X to y.

    Args:
        X: Input training data tensor
        y: Target training data tensor
        Xt: Optional validation input data
        yt: Optional validation target data
        lr: Learning rate
        weight_decay: Weight decay coefficient
        num_layers: Number of layers in the model
        epochs: Number of training epochs
        batch_size: Training batch size
        log_every: How often to log training progress
        criterion_cls: Loss function class to use
        early_stop_iters: Number of validation loss increases before early stopping

    Returns:
        Trained model
    """
    # Build model architecture
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    layers: List[nn.Module] = []
    for i in range(3 * (num_layers - 2)):
        if i % 3 == 0:
            layers.append(nn.Linear(input_dim, input_dim, bias=False))
        elif i % 3 == 1:
            layers.append(nn.LayerNorm(input_dim))
        else:
            layers.append(nn.ReLU())

    layers.append(nn.Linear(input_dim, output_dim))
    model = nn.Sequential(*layers).to(X.device)

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    least_test_loss = float("inf")
    test_loss = 0.0
    not_improved_count = 0

    try:
        progress_bar = tqdm.tqdm(range(epochs), desc="Training")
        for epoch in progress_bar:
            # Training loop
            model.train()
            epoch_losses = []
            indices = torch.randperm(X.shape[0])

            for i in range(0, X.shape[0], batch_size):
                batch_indices = indices[i : i + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

                if epoch == 0 and i == 0:
                    print(f"Initial loss: {loss.item():.4f}")

                # Validation and early stopping
                if (
                    Xt is not None
                    and yt is not None
                    and early_stop_iters
                    and i % 5 == 0
                ):
                    model.eval()
                    with torch.inference_mode():
                        test_loss = criterion(model(Xt), yt).item()

                        if test_loss > least_test_loss * 1.001:
                            not_improved_count += 1
                            if not_improved_count > early_stop_iters:
                                print("Test loss increased, stopping training")
                                return model
                        else:
                            not_improved_count = 0
                            least_test_loss = min(least_test_loss, test_loss)
                    model.train()

            # Logging
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            progress_bar.set_postfix(
                {"running_loss": f"{avg_loss:.4f}", "test_loss": f"{test_loss:.4f}"}
            )

            if (epoch + 1) % log_every == 0:
                print(
                    f"Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}"
                )

    except KeyboardInterrupt:
        # Allow the user to interrupt the training without losing the model
        print("\nTraining interrupted. Returning current model.")
        return model

    return model


def _train_projector(source_embeds, target_embeds, train_inds, val_inds, test_inds):
    # best params found with optuna:
    # {'lr': 3.9939641800203374e-05, 'batch_size': 1024, 'criterion_cls': <class '__main__.CosineLoss'>, 'weight_decay': 0.3582125898968158, 'num_layers': 1, 'threshold': 0.014284332513111963, 'init_layer_norm': False, 'epochs': 100, 'mult': 1.0}
    train_config = {
        "lr": 0.00005,
        "batch_size": 1024,
        "epochs": 100,
        "init_layer_norm": False,
        "num_layers": 1,
        "weight_decay": 0.5,
    }
    model = train_linear_regression(
        source_embeds[train_inds],
        target_embeds[train_inds],
        source_embeds[val_inds],
        target_embeds[val_inds],
        criterion=CosineLoss(),
        log_every=10,
        **train_config,
    )

    with torch.inference_mode():
        projected = model(source_embeds)
        results = torch.nn.functional.cosine_similarity(projected, target_embeds, dim=1)
        print(results.mean(), results[train_inds].mean(), results[test_inds].mean())

    return model, results, train_config


def main():
    args = simple_parsing.parse(ProjectorRegressionArgs)

    small = load_embeds(args.small_model)
    large = load_embeds(args.big_model)

    small = small.float().to(args.device)
    large = large.float().to(args.device)

    # create a subset of the tokens into train/test set
    # removing the reserved tokens (likely untrained)
    total_tokens = small.shape[0] - 256
    assert (
        total_tokens == 128000
    ), "The code has only been tested on Llama 3. Other models may need slight adjustments."

    train_inds = torch.randperm(total_tokens)

    val_inds = train_inds[:200].cuda()
    test_inds = train_inds[200:400].cuda()
    train_inds = train_inds[400:].sort().values.cuda()

    # Train small->big projector
    s2b_model, _, train_config = _train_projector(
        small, large, train_inds, val_inds, test_inds
    )
    metadata = {
        "small_model": args.small_model,
        "big_model": args.big_model,
        "train_config": train_config,
        "model_info": {
            "num_all_tokens": small.shape[0],
            "small_hidden_size": small.shape[1],
            "big_hidden_size": large.shape[1],
        },
    }
    # Push model to HF hub
    PushableWrapper(s2b_model, args.s2b_model_name, metadata=metadata).push_to_hub(
        args.projector_repo
    )

    if args.both_directions:
        # Train big->small projector
        b2s_model, b2s_results = _train_projector(
            large, small, train_inds, val_inds, test_inds
        )
        # Push model to HF hub
        PushableWrapper(b2s_model, args.b2s_model_name).push_to_hub(args.projector_repo)

    if args.debug:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.small_model)
        print(
            "best mapped tokens:",
            [
                tokenizer.decode(x.item())
                for x in b2s_results[:total_tokens].topk(50).indices
            ],
        )
        print(
            "worst mapped tokens:",
            [
                tokenizer.decode(-1 * x.item())
                for x in b2s_results[:total_tokens].topk(50).indices
            ],
        )


if __name__ == "__main__":
    main()
