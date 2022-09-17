import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)


def tune_transformer(dataset_details, model_name, num_samples=4, gpus_per_trial=0, smoke_test=False):
    # data_dir_name = "./data" if not smoke_test else "./test_data"
    # data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir, 0o755)

    # Change these as needed.
    model_name = model_name
    task_name = "aggDetect"

    num_labels = 3

    config = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task=task_name
    )

    # Download and cache tokenizer, model, and features
    print("Downloading and caching Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Triggers tokenizer download to cache
    print("Downloading and caching pre-trained model")
    AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )


    aggressions = load_dataset('csv', data_files={'train': dataset_details[0]})

    aggressions.set_format(type='pandas') # Converting from HF's dataset type to pandas dataframe
    df = aggressions['train'][:]
    dt = {'NAG':0, 'CAG':1, 'OAG':2}
    df['Category'].unique()
    labels = map(lambda x: dt[x], df['Category'])
    labels = list(labels)
    labels = pd.Series(labels)

    df['Label'] = labels
    labs = np.array(labels)
    texts = list(df['Sentence'])
    labels = list(df['Label'])
    train_size=0.8

    # In the first step we will split the data in training and remaining dataset
    train_texts, X_rem, train_labels, y_rem = train_test_split(texts, labels, train_size=0.8, random_state=43)

    # Now since we want the valid and test size to be equal (10% each of overall data). 
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    valid_texts, test_texts, valid_labels, test_labels = train_test_split(X_rem, y_rem, test_size=0.5, random_state=43)
    _texts, hyp_texts, _labels, hyp_labels = train_test_split(train_texts, train_labels, test_size=0.15, random_state=43)
    print(str(np.shape(hyp_texts))), print(str(np.shape(hyp_labels)))

    _texts2, hyp_v_texts, _labels2, hyp_v_labels = train_test_split(valid_texts, valid_labels, test_size=0.5, random_state=43)
    print(str(np.shape(hyp_v_texts))), print(str(np.shape(hyp_v_labels)))

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True)
    hyp_encodings = tokenizer(hyp_texts, truncation=True, padding=True)
    hyp_v_encodings = tokenizer(hyp_v_texts, truncation=True, padding=True)

    import torch
    # tensors
    class AggressionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = AggressionDataset(train_encodings, train_labels)
    valid_dataset = AggressionDataset(valid_encodings, valid_labels)
    hyp_dataset = AggressionDataset(hyp_encodings, hyp_labels)
    hyp_v_dataset = AggressionDataset(hyp_v_encodings, hyp_v_labels)

    from datasets import load_metric

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        metric = load_metric("f1")
        return metric.compute(predictions=preds, references=labels, average="macro")



    training_args = TrainingArguments(
        output_dir=".",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
    )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=(train_dataset if dataset_details[1]=='Ours' else hyp_dataset),
        eval_dataset=(valid_dataset if dataset_details[1]=='Ours' else hyp_v_dataset),
        # compute_metrics=build_compute_metrics_fn(task_name),
        compute_metrics=compute_metrics,
    )

    tune_config = {
        "per_device_train_batch_size": tune.choice([4, 8]),
        "per_device_eval_batch_size": tune.choice([8]),
        "num_train_epochs": tune.choice([2, 3, 4, 5, 6]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration", 
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            # "weight_decay": [0.0, 0.1, 0.01],
            "warmup_ratio": [0.0, 0.1],
            # "learning_rate": [1e-3, 1e-4, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
            "weight_decay": tune.uniform(0.1, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [4]
        },
        
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_f1", "eval_loss", "epoch", "training_iteration"],
    )

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": 0, "gpu": gpus_per_trial},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 5},
        progress_reporter=reporter,
        local_dir="./ray_results_l3cube_pune_hing_mbert/",
        name="tune_transformer_pbt_"+dataset_details[1], # Stores the results in corresponding dataset folder 
        log_to_file=True,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address to use for Ray. "
        'Use "auto" for cluster. '
        "Defaults to None for local.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )

    args, _ = parser.parse_known_args()
    ##
    dataset_dict = {'Ours': ['total_ours.csv', 'Ours'],
                    'TRAC': ['total_TRAC.csv', 'TRAC'],
                    'Combined': ['total_combined.csv', 'Combined'],
                    'Code_Mixed': ['total_code_mixed.csv', 'Code_Mixed'],
                    'Non_Code_Mixed': ['total_non_code_mixed.csv', 'Non_Code_Mixed']}

    if args.smoke_test:
        ray.init()
    elif args.server_address:
        ray.init(f"ray://{args.server_address}")
    else:
        ray.init(args.ray_address, ignore_reinit_error=True)

    if args.smoke_test:
        for k, v in dataset_dict.items():
            tune_transformer(v, "l3cube-pune/hing-mbert", num_samples=5, gpus_per_trial=0, smoke_test=True)
    else:
        # You can change the number of GPUs here:
        for k, v in dataset_dict.items():
            tune_transformer(v, "l3cube-pune/hing-mbert", num_samples=10, gpus_per_trial=1)
