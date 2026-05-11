# Get started with Tensor Parallelism and Sequence Parallelism using PyTorch DTensor and Ray Train

**Time to complete:** 25 min

This template shows how to train large language models with Ray Train and PyTorch's native [DTensor (Distributed Tensor) API](https://docs.pytorch.org/docs/stable/distributed.tensor.html), combining:

- **Tensor Parallelism (TP)** to shard model weights across GPUs in each TP group.
- **Sequence Parallelism (SP)** to keep transformer activations sharded on the sequence dimension around TP-compatible blocks.
- **Data Parallelism (DP)** with FSDP2 across DP groups.

This tutorial focuses on the Megatron-LM / TorchTitan-style sequence-parallel pattern used with tensor parallelism:

- Keep activations as `Shard(1)` on the sequence dimension around transformer blocks.
- All-gather sequence-sharded activations before attention so attention runs over the full sequence.
- Use row-wise TP output projections with `output_layouts=Shard(1)` so DTensor performs a reduce-scatter after attention and after the MLP.

**This is not Context Parallelism.** Context Parallelism splits the attention computation for long contexts and exchanges key/value tensors across context shards. This tutorial does not split attention across context shards. Each TP rank gathers the full sequence before attention and computes attention for its local TP shard of heads/hidden dimensions.

**Note:** This tutorial uses PyTorch's native `DTensor`, tensor-parallel, and `fully_shard` APIs. These APIs require PyTorch 2.4 or later and are still evolving. The code below uses PyTorch 2.9.1.

<div id="anyscale-note" class="alert alert-block alert-warning">

  <strong>Anyscale Specific Configuration</strong>

  <p><strong>Note:</strong> This tutorial is optimized for the Anyscale platform. When running on open source Ray, additional configuration is required. For example, you would need to manually:</p>

  <ul>
    <li><strong>Configure your Ray Cluster</strong>: Set up your multi-node environment and manage resource allocation without Anyscale's automation.</li>
    <li><strong>Manage Dependencies</strong>: Manually install and manage dependencies on each node.</li>
    <li><strong>Set Up Storage</strong>: Configure your own distributed or shared storage system for model checkpointing.</li>
  </ul>
</div>

<style>
  div#anyscale-note > p,
  div#anyscale-note > ul,
  div#anyscale-note > ul li {
    color: black;
  }

  div#anyscale-note {
    background-color: rgb(255, 243, 205);
  }

  div#anyscale-note {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 15px;
  }

</style>

## Understanding TP + SP + DP

This tutorial uses a 2D PyTorch `DeviceMesh` with dimensions `(dp, tp)`:

```text
Device Mesh (2x2):
        TP Dim
      [0]  [1]
 DP   +---+---+
 Dim  | 0 | 1 |  <- TP group, same batch, sharded model
      +---+---+
      | 2 | 3 |  <- TP group, same batch, sharded model
      +---+---+
        ^   ^
       DP groups, different batches, FSDP gradient sync
```

TP shards model weights across each row of the mesh. DP/FSDP shards parameters and optimizer state across each column of the mesh.

SP changes the activation layout inside the TP group. Instead of keeping every TP rank's activation tensor as `[batch, sequence, hidden]`, the sequence dimension is sharded so each TP rank holds `[batch, sequence / tp_size, hidden]` between TP-compatible submodules.

For Qwen-style Hugging Face models, the embedding output remains replicated until the model has built its causal mask and rotary position embeddings. The first decoder layer then shards the hidden states on the sequence dimension. This keeps model-level position handling unchanged while still keeping activations sequence-sharded across the transformer stack.

The forward flow through one decoder layer is:

```text
Sequence-sharded activation: [B, S/tp, H]
        |
        | input_layernorm with SequenceParallel()
        v
Shard(1) activation
        |
        | PrepareModuleInput(... Shard(1) -> Replicate())
        | This is the attention-pre all-gather.
        v
Replicated full sequence: [B, S, H]
        |
        | q_proj/k_proj/v_proj: ColwiseParallel()
        | attention over the full sequence, local TP heads
        | o_proj: RowwiseParallel(output_layouts=Shard(1))
        | This turns the row-wise partial output into Shard(1)
        | through reduce-scatter.
        v
Sequence-sharded activation: [B, S/tp, H]
        |
        | post_attention_layernorm with SequenceParallel()
        | MLP input all-gather: Shard(1) -> Replicate()
        | MLP down projection reduce-scatter: Partial -> Shard(1)
        v
Sequence-sharded activation for the next layer
```

In DTensor terms, `Shard(1) -> Replicate()` performs an all-gather on the sequence dimension, and a row-wise TP linear whose output is requested as `Shard(1)` performs a reduce-scatter from the row-wise partial output.

## 1. Package and environment setup

Install the required dependencies:

```bash
%%bash
python -m pip install -U torch==2.9.1 torchvision==0.24.1 transformers==4.48.0 datasets==2.21.0
```

```python
import json
import logging
import os
import tempfile
import uuid

import torch

logger = logging.getLogger(__name__)


def get_mixed_precision_dtype() -> torch.dtype:
    """Select a mixed-precision dtype that the current GPU supports."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

## 2. Data loading with TP-aware sharding

All TP ranks in the same TP group must receive identical input data because they collectively compute one model replica. Standard distributed data loading shards by `world_rank`, which would give each TP rank a different batch. For TP, shard the data by `dp_rank` and `dp_size` instead.

The global batch size is `batch_size_per_gpu * dp_size`, not `batch_size_per_gpu * world_size`, because TP ranks process the same examples.

```python
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

import ray.train


def create_dataloader(
    model_name: str,
    dataset_name: str,
    seq_length: int,
    batch_size_per_gpu: int,
    dp_rank: int,
    dp_size: int,
    seed: int = 42,
    dataset_percentage: float = 10.0,
) -> DataLoader:
    """
    Create a dataloader with TP-aware sharding.

    The sampler uses dp_rank/dp_size, not world_rank/world_size, so all TP
    ranks in the same DP group receive identical batches.
    """
    dataset_config = "wikitext-2-raw-v1" if dataset_name == "wikitext" else None
    dataset_percentage = float(dataset_percentage)
    if not 0 < dataset_percentage <= 100:
        raise ValueError(
            f"dataset_percentage must be in (0, 100], got {dataset_percentage}."
        )
    split_spec = f"train[:{dataset_percentage:.15g}%]"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split_spec,
        download_config=DownloadConfig(disable_tqdm=True),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )
    tokenized = tokenized.filter(
        lambda example: sum(example["attention_mask"]) > 1,
        keep_in_memory=True,
    )

    def add_labels(examples):
        labels = []
        for input_ids, attention_mask in zip(
            examples["input_ids"], examples["attention_mask"]
        ):
            labels.append(
                [
                    token if mask == 1 else -100
                    for token, mask in zip(input_ids, attention_mask)
                ]
            )
        examples["labels"] = labels
        return examples

    tokenized = tokenized.map(add_labels, batched=True, num_proc=1, keep_in_memory=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    sampler = DistributedSampler(
        tokenized,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
    )

    return DataLoader(
        tokenized,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        drop_last=True,
    )
```

## 3. Model parallelization with DTensor

The sequence-parallel plan below uses PyTorch DTensor layouts directly:

- `SequenceParallel(sequence_dim=1)` keeps layernorm activations sharded on the sequence dimension.
- `PrepareModuleInput(... Shard(1) -> Replicate())` all-gathers activations before attention and before the MLP.
- `RowwiseParallel(output_layouts=Shard(1))` asks DTensor to reduce-scatter row-wise TP partial outputs back to sequence shards.

The example targets Qwen/Llama-style Hugging Face decoder models, where transformer layers are available as `model.model.layers`, attention projections are under `self_attn`, MLP projections are under `mlp`, and the hidden state is passed into attention as the `hidden_states` keyword argument.

```python
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import AutoConfig, AutoModelForCausalLM

import ray.train.torch


def validate_tp_sp_config(
    model_name: str,
    tp_size: int,
    dp_size: int,
    world_size: int,
    seq_length: int,
):
    """Validate the model and mesh sizes before initializing DTensor state."""
    if tp_size <= 0 or dp_size <= 0:
        raise ValueError(
            f"tp_size and dp_size must be positive, got tp_size={tp_size}, "
            f"dp_size={dp_size}."
        )
    if dp_size * tp_size != world_size:
        raise ValueError(
            f"dp_size ({dp_size}) * tp_size ({tp_size}) must equal "
            f"world_size ({world_size})."
        )
    if seq_length % tp_size != 0:
        raise ValueError(
            f"seq_length ({seq_length}) must be divisible by tp_size ({tp_size}) "
            "so sequence shards are evenly sized."
        )

    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_heads = getattr(hf_config, "num_attention_heads", None)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)

    if num_heads is None:
        raise ValueError("Model config must define num_attention_heads.")
    if num_heads % tp_size != 0:
        raise ValueError(
            f"TP size {tp_size} must divide attention head count {num_heads}."
        )
    if num_kv_heads is not None and num_kv_heads % tp_size != 0:
        raise ValueError(
            f"TP size {tp_size} must divide key/value head count {num_kv_heads}."
        )


def parallelize_qwen_llama_layer_with_tp_sp(
    layer,
    tp_mesh,
    input_layout,
) -> None:
    """Apply tensor parallelism plus sequence parallelism to one decoder layer."""
    # The first layer receives a full-sequence replicated activation because the
    # Hugging Face model computes RoPE and masks before entering the layer stack.
    # Later layers receive the sequence shard produced by the previous layer.
    parallelize_module(
        layer,
        tp_mesh,
        PrepareModuleInput(
            input_layouts=(input_layout,),
            desired_input_layouts=(Shard(1),),
            use_local_output=True,
        ),
    )

    layer_tp_sp_plan = {
        # Keep normalization activations sequence-sharded.
        "input_layernorm": SequenceParallel(sequence_dim=1),
        # Attention needs the full sequence. This Shard(1) -> Replicate()
        # redistribution is the attention-pre all-gather.
        "self_attn": PrepareModuleInput(
            input_kwarg_layouts={"hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"hidden_states": Replicate()},
        ),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        # Row-wise TP produces a partial output. Requesting Shard(1)
        # makes DTensor reduce-scatter it back to sequence shards.
        "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "post_attention_layernorm": SequenceParallel(sequence_dim=1),
        # The MLP also starts from a sequence shard and gathers before
        # column-wise projections.
        "mlp": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
    }
    parallelize_module(layer, tp_mesh, layer_tp_sp_plan)


def setup_model_with_tp_sp(
    model_name: str,
    tp_size: int,
    dp_size: int,
    world_rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
    seq_length: int,
):
    """
    Set up tensor parallelism, sequence parallelism, and FSDP2.

    Returns:
        tuple: (model, tp_mesh, dp_mesh, tp_rank, dp_rank)
    """
    validate_tp_sp_config(
        model_name=model_name,
        tp_size=tp_size,
        dp_size=dp_size,
        world_size=world_size,
        seq_length=seq_length,
    )

    tp_rank = world_rank % tp_size
    dp_rank = world_rank // tp_size

    if world_rank == 0:
        logger.info(
            "Setting up 2D mesh with sequence parallelism: "
            f"dp_size={dp_size}, tp_size={tp_size}"
        )

    device_mesh = init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    dtype = get_mixed_precision_dtype()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError(
            "This tutorial expects a Qwen/Llama-style causal LM with "
            "`model.model.layers`."
        )

    layers = model.model.layers
    if world_rank == 0:
        logger.info(f"Applying TP + SP to {len(layers)} transformer layers")

    # Keep embeddings replicated so the model-level causal mask and rotary
    # position embeddings are computed for the full sequence before SP starts.
    parallelize_module(
        model,
        tp_mesh,
        {
            # Return a local sequence shard because Hugging Face slices hidden
            # states before calling lm_head. The lm_head plan below annotates
            # that local tensor as Shard(1) and all-gathers before projection.
            "model.norm": SequenceParallel(sequence_dim=1, use_local_output=True),
            # The final projection gathers sequence shards before computing logits.
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            ),
        },
    )

    for layer_idx, layer in enumerate(layers):
        layer_input_layout = Replicate() if layer_idx == 0 else Shard(1)
        parallelize_qwen_llama_layer_with_tp_sp(
            layer,
            tp_mesh,
            input_layout=layer_input_layout,
        )

    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

    if dp_size > 1:
        if world_rank == 0:
            logger.info("Applying FSDP2 across the data-parallel mesh")
        for layer in layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
    else:
        if world_rank == 0:
            logger.info("dp_size=1, skipping FSDP2 sharding")

    if world_rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized with {num_params:,} parameters")

    return model, tp_mesh, dp_mesh, tp_rank, dp_rank
```

## 4. Checkpointing

With TP + SP + FSDP2, each worker holds a shard of the model and optimizer state. Use PyTorch [Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) APIs so sharded state dicts are saved and restored correctly.

In this example, `dcp.save` and `dcp.load` handle the distributed state dicts, and rank 0 writes `metadata.json` for the epoch and step. Calling `ray.train.report(..., checkpoint=...)` on all workers lets Ray package the distributed checkpoint into one logical artifact.

```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from ray.train import Checkpoint


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    world_rank: int,
    epoch: int,
    step: int,
    avg_loss: float,
) -> None:
    """Save a distributed checkpoint and report it to Ray Train."""
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

        dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_dir)

        if world_rank == 0:
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump({"epoch": epoch, "step": step}, f)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        ray.train.report({"loss": avg_loss, "epoch": epoch}, checkpoint=checkpoint)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load model and optimizer shards from the latest Ray Train checkpoint."""
    checkpoint = ray.train.get_checkpoint()
    if checkpoint is None:
        return 0

    with checkpoint.as_directory() as checkpoint_dir:
        metadata_path = os.path.join(checkpoint_dir, "metadata.json")

        model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)
        state_dict = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }
        dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        start_epoch = 0
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            start_epoch = metadata.get("epoch", -1) + 1

    return start_epoch
```

## 5. Training loop

The main training function brings together the TP-aware dataloader, DTensor layout plan, FSDP2 sharding, checkpointing, and Ray Train reporting.

```python
def train_func(config):
    """
    Main training loop executed by each Ray Train worker.

    This function:
    1. Sets up a 2D device mesh for DP x TP.
    2. Applies tensor parallelism and sequence parallelism with DTensor.
    3. Applies FSDP2 across the DP dimension.
    4. Runs a short training loop with distributed checkpointing.
    """
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    device = ray.train.torch.get_device()

    tp_size = config["tp_size"]
    dp_size = config["dp_size"]

    if world_rank == 0:
        logger.info(f"Worker started: world_rank={world_rank}, world_size={world_size}")

    model, _, _, _, dp_rank = setup_model_with_tp_sp(
        model_name=config["model_name"],
        tp_size=tp_size,
        dp_size=dp_size,
        world_rank=world_rank,
        world_size=world_size,
        device=device,
        seed=config.get("seed", 42),
        seq_length=config["seq_length"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-5),
        weight_decay=config.get("weight_decay", 0.01),
        foreach=False,
    )
    start_epoch = load_checkpoint(model, optimizer)

    dtype = get_mixed_precision_dtype()
    if world_rank == 0:
        logger.info(f"Parallelism: {dp_size} DP x {tp_size} TP with SP")
        logger.info(f"torch.autocast enabled with dtype={dtype}")

    dataloader = create_dataloader(
        model_name=config["model_name"],
        dataset_name=config["dataset_name"],
        seq_length=config["seq_length"],
        batch_size_per_gpu=config["batch_size_per_gpu"],
        dp_rank=dp_rank,
        dp_size=dp_size,
        seed=config.get("seed", 42),
        dataset_percentage=config.get("dataset_percentage", 10.0),
    )

    steps_per_epoch = len(dataloader)
    if world_rank == 0:
        logger.info(f"Dataloader created: {steps_per_epoch} steps per epoch")
    if steps_per_epoch == 0:
        raise ValueError(
            "Dataloader is empty. Increase dataset_percentage or reduce "
            "batch_size_per_gpu."
        )

    log_interval = config.get("log_interval", 10)
    model.train()

    for epoch in range(start_epoch, config["num_epochs"]):
        dataloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0
        last_step = -1

        for step, batch in enumerate(dataloader):
            last_step = step
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = outputs.loss

            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            num_batches += 1

            if (
                world_rank == 0
                and log_interval is not None
                and log_interval > 0
                and step % log_interval == 0
            ):
                logger.info(
                    f"Epoch: {epoch} Step: {step + 1}/{steps_per_epoch} "
                    f"Loss: {loss_value:.4f}"
                )

            if config.get("debug_steps", 0) > 0 and step + 1 >= config["debug_steps"]:
                if world_rank == 0:
                    logger.info(f"Debug steps finished. Stopping epoch {epoch}.")
                break

        if num_batches == 0:
            if world_rank == 0:
                logger.warning(
                    f"Epoch {epoch} processed zero batches. Skipping checkpoint."
                )
            continue

        avg_loss = running_loss / num_batches
        save_checkpoint(model, optimizer, world_rank, epoch, last_step, avg_loss)

        if world_rank == 0:
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
```

## 6. Launch the distributed training job

Configure and launch the training job using Ray Train's `TorchTrainer`. This example uses 4 GPUs with 2-way tensor parallelism, sequence parallelism inside each TP group, and 2-way data parallelism.

```python
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


tp_size = 2
dp_size = 2
num_workers = tp_size * dp_size

scaling_config = ScalingConfig(
    num_workers=num_workers,
    use_gpu=True,
)

train_loop_config = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "dataset_name": "wikitext",
    "dataset_percentage": 5.0,
    "tp_size": tp_size,
    "dp_size": dp_size,
    "batch_size_per_gpu": 1,
    "seq_length": 512,
    "num_epochs": 1,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "log_interval": 5,
    "debug_steps": 20,
    "seed": 42,
}

experiment_name = f"tp_sp_dtensor_{uuid.uuid4().hex[:8]}"
storage_path = "/mnt/cluster_storage/ray_train_tp_sp_dtensor"

run_config = RunConfig(
    name=experiment_name,
    storage_path=storage_path,
    worker_runtime_env={
        "pip": [
            "torch==2.9.1",
            "torchvision==0.24.1",
            "transformers==4.48.0",
            "datasets==2.21.0",
        ],
    },
)

trainer = TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    train_loop_config=train_loop_config,
    run_config=run_config,
)

print(
    "Starting TP + SP training with "
    f"{tp_size}-way TP and {dp_size}-way DP..."
)
result = trainer.fit()
print("Training completed successfully!")
print(f"Final metrics: {result.metrics}")

RUN_RESUME_DEMO = False
if RUN_RESUME_DEMO:
    resume_train_loop_config = dict(train_loop_config)
    resume_train_loop_config["num_epochs"] = 2
    resume_trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config=resume_train_loop_config,
        run_config=run_config,
    )
    resume_result = resume_trainer.fit()
    print(f"Resumed metrics: {resume_result.metrics}")
```

## Scaling to larger models

To train larger models like Qwen2-7B or Llama-3-8B, increase the tensor-parallel and data-parallel sizes. For example, on 8 GPUs you can use 4-way TP with sequence parallelism and 2-way DP:

```text
tp_size = 4
dp_size = 2
num_workers = 8
seq_length = 2048
```

Tips for scaling:

- Keep `seq_length` divisible by `tp_size` so sequence shards are evenly sized.
- Keep TP within high-bandwidth GPU interconnect domains when possible.
- Ensure `tp_size` divides both attention heads and key/value heads for GQA models.
- SP reduces activation memory around TP-compatible blocks, but attention still sees the full sequence. Use Context Parallelism or another long-context attention strategy only when you need to split the attention computation itself.

## Summary

In this tutorial, you learned:

- How to combine Ray Train, PyTorch DTensor tensor parallelism, sequence parallelism, and FSDP2.
- How `Shard(1)` keeps activations sequence-sharded between transformer submodules.
- How the attention-pre `Shard(1) -> Replicate()` transition all-gathers the full sequence before attention.
- How `RowwiseParallel(output_layouts=Shard(1))` reduce-scatters attention and MLP outputs back to sequence shards.
- Why this Megatron-LM / TorchTitan-style sequence parallelism is different from Context Parallelism.
