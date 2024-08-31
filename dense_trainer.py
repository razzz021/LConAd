import argparse
from sentence_transformers import (SentenceTransformer,
                                   models,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers.integrations import TensorBoardCallback
from datasets import Dataset, DatasetDict

from torch.utils import tensorboard

from custom_loss import InfoNCELoss

FACT_FILE = "/home/ray/suniRet/data/train_data/fact.txt"
REASON_FILE = "/home/ray/suniRet/data/train_data/reason.txt"
FACT_REASON_FILE = "/home/ray/suniRet/data/train_data/fact_reason.txt"


def build_model(model_name, pooling_mode):
    if "sentence-transformers" in model_name:
        return SentenceTransformer(model_name)

    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode=pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    for name, param in model.named_parameters():
        param.data = param.data.contiguous()

    return model

def load_samples(filepath, sample_type="A"):
    train_samples = []
    
    if sample_type == "A":
        with open(filepath, encoding="utf8") as file:
            for line in file.readlines():
                line = line.strip()
                if len(line) >= 10:
                    train_samples.append([line, line])
    
    elif sample_type == "AB":
        with open(filepath, encoding="utf8") as file:
            while True:
                line1 = file.readline().strip()
                if not line1:
                    break
                line2 = file.readline().strip()
                train_samples.append([line1, line2])
                
    elif sample_type == "ABC":
        with open(filepath, encoding="utf8") as file:
            while True:
                line1 = file.readline().strip()
                if not line1:
                    break
                line2 = file.readline().strip()
                line3 = file.readline().strip()
                train_samples.append([line1, line2, line3])
    else:
        return None 
    
    return train_samples

def build_dataset(samples):
    data_dict = {}
    for idx, col in enumerate(zip(*samples)):
        data_dict[f'sent_{idx}']=col
    dataset = Dataset.from_dict(data_dict)
    
    return dataset

def build_loss(loss_name, model):
    if loss_name == 'MNR':
        return MultipleNegativesRankingLoss(model)
    elif loss_name == "InfoNCE":
        return InfoNCELoss(model)
    else:
        return None

def build_mixture_dataset_losses(model):
    fact = load_samples(FACT_FILE, sample_type="A")
    reason = load_samples(REASON_FILE, sample_type="A")
    fact_reason = load_samples(FACT_REASON_FILE, sample_type="AB")
    fact = build_dataset(fact)
    reason = build_dataset(reason)
    fact_reason = build_dataset(fact_reason)
    dataset_dict = DatasetDict({'uf':fact, 'ur':reason, 'ufr':fact_reason})
    mnrl_loss = InfoNCELoss(model)
    losses = {"uf": mnrl_loss, "ur": mnrl_loss, "ufr": mnrl_loss}
    return dataset_dict, losses

    

def main(args):
    print("Step 1: Loading a model to fine-tune")
    model = build_model(args.model_name, args.pooling_mode)
    print(f"Model {args.model_name} loaded with pooling mode {args.pooling_mode}")

    if not args.is_data_mixture:
        print("Step 2: Loading a dataset to fine-tune on")
        train_samples = load_samples(args.filepath, args.sample_type)
        train_dataset = build_dataset(train_samples)
        print(f"Loaded {len(train_samples)} samples from {args.filepath}")

        print("Step 3: Defining a loss function")
        loss = build_loss(args.loss_name, model)
        print(f"Using {args.loss_name} loss function")
    else:
        print("Step 2: Loading a mixture dataset to fine-tune on")
        train_dataset, loss = build_mixture_dataset_losses(model)


    print("Step 4: Specifying training arguments")
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        # per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # eval_strategy="steps",
        # eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
    )
    print(f"Training arguments set with {args.num_train_epochs} epochs and batch size {args.batch_size}")

    print("Step 5: Adding TensorBoard callback")
    tensorboard_callback = TensorBoardCallback()
    print("TensorBoard callback added")

    print("Step 6: Creating a trainer and starting training")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        loss=loss,
        train_dataset=train_dataset,
        callbacks=[tensorboard_callback]
    )

    trainer.train()

    # . Save the trained model
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="sentence-transformers/paraphrase-mpnet-base-v2")
    parser.add_argument("--pooling_mode", type=str, default="cls", help="Pooling mode: cls, mean, max")
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--run_name", type=str, default="mpnet-base-all-nli-triplet")
    parser.add_argument("--sample_type", type=str, default="A", help="Sample type: A or B")
    parser.add_argument("--loss_name", type=str, default="MNR", help="Loss function: MNR or custom")
    parser.add_argument("--fp16", type=bool, default=True, help="Use FP16 precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--is_data_mixture", type=bool, default=False, help="Use mixture of datasets")
    args = parser.parse_args()

    main(args)
