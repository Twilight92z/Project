import torch
import wandb
import transformers
import torch.nn as nn
from datasets import load_dataset
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


class Classifier(nn.Module):
    def __init__(self, hidden_size, n_class):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(hidden_size, 512)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 256)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(256, n_class)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x


class BertClassifierConfig(BertConfig):   
    def set_class(self, n_class):
        self.n_class = n_class


class GoogleBertClassifier(BertPreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super(GoogleBertClassifier, self).__init__(config)
        self.classifier = Classifier(config.hidden_size, config.n_class)
        self.bert_model = BertModel(config)
        self.post_init()
    
    def forward(self, input_ids, attention_mask, labels):
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(bert_outputs[1])
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
        )


@dataclass
class BertCollactor(object):
    tokenizer: BertTokenizer
    def __call__(self, dataset):
        input_ids = [torch.tensor(item['input_ids']) for item in dataset]
        labels = torch.stack([torch.tensor(item['labels']) for item in dataset])
        tokenizer = self.tokenizer
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(tokenizer.pad_token_id))
   
    
@dataclass
class ModelArguments:
    project: str


def make_datamodule(tokenizer):
    train_set = load_dataset("google-research-datasets/poem_sentiment")["train"]
    def maps_function(row):
        labels = row["label"]
        input_ids = tokenizer([row['verse_text']], truncation=True, max_length=512)["input_ids"][0]
        return dict(labels=labels, input_ids=input_ids)
    train_set = train_set.map(maps_function)
    data_collactor = BertCollactor(tokenizer)
    return dict(train_dataset=train_set, data_collator=data_collactor, eval_dataset=None)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_yaml_file("./test.yaml")
    wandb.login(key="540c7e59de90ce2fc694c3658dd3caa0e9b9fb33", relogin=True)
    wandb.init(project=model_args.project, name=training_args.run_name)
    config = BertClassifierConfig.from_pretrained("google-bert/bert-base-uncased")
    config.set_class(4)
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bertclassifier = GoogleBertClassifier(config)
    bertclassifier.to("cuda:0")
    data_module = make_datamodule(tokenizer)
    trainer = Trainer(model=bertclassifier, args=training_args, **data_module)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train()