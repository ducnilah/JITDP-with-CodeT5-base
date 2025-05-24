import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, T5EncoderModel, AutoModel
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
import wandb
from accelerate import Accelerator
import numpy as np
import evaluate
from datasets import Dataset
import torch.nn as nn
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.stats import boxcox

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, numeric_features, labels):
        self.encodings = encodings
        self.numeric_features = numeric_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["numeric_feats"] = torch.tensor(self.numeric_features[idx], dtype=torch.float)
        return item

class CodeT5WithNumeric(nn.Module):
    def __init__(self, model_name="Salesforce/codet5-base", num_labels=2, num_numeric=13, class_weights=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.text_encoder = T5EncoderModel.from_pretrained(model_name)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(num_numeric, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, numeric_feats, labels=None):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = text_output.last_hidden_state[:, 0, :]
        numeric_embedding = self.numeric_mlp(numeric_feats)
        combined = torch.cat([cls_embedding, numeric_embedding], dim=1)
        logits = self.classifier(combined)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {'loss': loss, 'logits': logits}
        return {'logits': logits}

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    numeric_feats = torch.stack([item['numeric_feats'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'numeric_feats': numeric_feats,
    }

numeric_cols = ['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'nuc', 'exp', 'rexp', 'sexp']
scaler = StandardScaler()

def apply_transformations(df):
        transformed_df = df.copy()
        for feature in numeric_cols:
            if feature in transformed_df.columns:
                transformed_df[feature] = transformed_df[feature] + 0.0000000001
                transformed_df[feature], _ = boxcox(transformed_df[feature])
                transformed_df[feature] = scaler.fit_transform(transformed_df[feature].values.reshape(-1,1)).flatten()
                
        return transformed_df

path_1 = ''
path_2 = ''
path_3 = ''

defects = pd.read_json(path_1, lines=True)
non_defects = pd.read_json(path_2, lines=True)
fixes = pd.read_json(path_3, lines=True)

all_samples = pd.concat([fixes, defects, non_defects],)
output_file_path = '/kaggle/working/labels.jsonl'
all_samples = all_samples.sample(frac=1, random_state=42)
all_samples.to_json(output_file_path, orient='records', lines=True)

path_4 = ''
path_5 = ''
path_6 = ''

labels = pd.read_json(path_4, lines=True)
patches = pd.read_json(path_5, lines=True)
features = pd.read_json(path_6, lines=True)

st = pd.merge(labels, patches, left_on="commit_id", right_on="commit_id", how="inner")
df = pd.merge(st, features, left_on="commit_id", right_on="commit_id", how="inner")

df.drop('date_y', axis=1, inplace=True)
df.rename(columns={'date_x': 'date'}, inplace=True)

df['input_text'] = df['messages'] + " " + df['code_change']
df[numeric_cols] = apply_transformations(df[numeric_cols])

# df -> list
X = list(df['input_text'])
Xn = df[numeric_cols].values.tolist()
y = list(df['label'])

# Split the data into training and testing sets
X_train, X_test, Xn_train, Xn_test, y_train, y_test = train_test_split(X, Xn,y, test_size=0.15, random_state=42)

# Skipping samples with weird utf that CodeT5 Tokenizer could not consume smh (Skipping was not a really right way, sounds like stalin sort, but feels like skipping some samples out of ~100k samples does not really affect the final resuld :D)
# I ran a code to find all the sample that did not qualified but i'm not putting that code here
skipped_train_indices = [923, 8883, 10380, 10719, 11456, 21702, 22674, 25533, 27077, 27709, 31513, 32679, 37974, 39187, 46557, 49032, 49097, 49159, 51924, 55466, 57874, 59021, 61096, 61543, 61886, 62186, 64366, 68246, 71888, 75275, 77942]
skipped_test_indices = [4216, 13063]

X_train = [value for idx, value in enumerate(X_train) if idx not in skipped_train_indices]
X_test = [value for idx, value in enumerate(X_test) if idx not in skipped_test_indices]
Xn_train = [value for idx, value in enumerate(Xn_train) if idx not in skipped_train_indices]
Xn_test = [value for idx, value in enumerate(Xn_test) if idx not in skipped_test_indices]
y_train = [value for idx, value in enumerate(y_train) if idx not in skipped_train_indices]
y_test = [value for idx, value in enumerate(y_test) if idx not in skipped_test_indices]

# Weighted loss for imbalanced dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

y_train_tensor = torch.tensor(y_train)
class_counts = torch.bincount(y_train_tensor)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()
weights = weights.to(device)

class_weights = weights.to(device)
model = CodeT5WithNumeric('Salesforce/codet5-base', num_labels=2, num_numeric=13, class_weights=class_weights)
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base', use_fast=False)

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=256)

train_dataset = Dataset(train_encodings, Xn_train, y_train)
test_dataset = Dataset(test_encodings, Xn_test, y_test)

# Metrics
auc_score = evaluate.load("roc_auc")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1_score = evaluate.load("f1")

def compute_metrics(eval_pred, threshold=0.43):
    predictions, labels = eval_pred
    logits = predictions
    
    probabilities = softmax(logits, axis=-1)
    positive_class_probs = probabilities[:, 1]
    predicted_classes = (positive_class_probs >= threshold).astype(int)
    
    auc = auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc']
    prec = precision.compute(predictions=predicted_classes, references=labels)['precision']
    rec = recall.compute(predictions=predicted_classes, references=labels)['recall']
    f1 = f1_score.compute(predictions=predicted_classes, references=labels)['f1']
    
    return {
        "AUC": round(auc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
        "Threshold": threshold
    }

# Set up for trainer
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    report_to="wandb",
    eval_steps=2000,
    save_steps=4000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    learning_rate=7e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    save_safetensors=False,
    metric_for_best_model="F1",
    greater_is_better=True,
    seed=42,
    fp16=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

wandb.login(key='')
wandb.init(project="jitdp", name='experiment')

# Train
accelerator = Accelerator()
trainer.train()
trainer.evaluate()

  
