import hydra
from omegaconf import DictConfig
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import jsonlines
from sklearn.model_selection import train_test_split
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import SequentialEvaluator


def load_samples(file_path):
    samples = []
    with jsonlines.open(file_path, mode='r') as reader:
        for obj in reader:
            text1 = obj["text1"]
            text2 = obj["text2"]
            label = float(obj["label"]) 

            samples.append(InputExample(texts=[text1, text2], label=label))
    return samples

def split_samples(samples, test_size=0.2, seed=42):
    train_samples, test_samples = train_test_split(samples, test_size=test_size, random_state=seed)
    return train_samples, test_samples

@hydra.main(config_path="config", config_name="cross_train_config", version_base="1.1")
def train_model(cfg: DictConfig):

    model = CrossEncoder(cfg.model_name, num_labels=1)  
    
    for name, param in model.model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()


    samples = load_samples(cfg.data_path)
    train_samples, test_samples = split_samples(samples, test_size=cfg.test_size, seed=cfg.seed)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=cfg.batch_size)
    
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name="legal-dev")
    
    optimizer_params = {
        'lr': cfg.learning_rate,           
        'weight_decay': cfg.weight_decay   
    }


    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=cfg.epochs,
        evaluation_steps=cfg.evaluation_steps,
        warmup_steps=cfg.warmup_steps,
        output_path=cfg.output_dir,
        optimizer_params=optimizer_params 
    )

if __name__ == "__main__":
    train_model()
