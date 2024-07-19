import transformers
from datasets import load_dataset
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_path: str = field(default=None)


@dataclass
class DataArguments:
    ground_path: str = field(default=None, metadata={"help": "Path to the training data."})
    project_name: str = field(default=None, metadata={"help": "wandb project name"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    iter_times: int = field(default=1)



def make_datamodules(tokenizer):
    pass



def train():
    load_dataset("PKU-Alignment/PKU-SafeRLHF-10K")