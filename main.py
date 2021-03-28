"""
Main file
We will run the whole program from here
"""

import torch
import hydra

from train import train, evaluate
from dataset import MyDataset
from models import MyModel
from torch.utils.data import DataLoader
from utils import main_utils, train_utils
from utils.train_logger import TrainLogger
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name='config')
def main(cfg: DictConfig) -> None:
    """
    Run the code following a given configuration
    :param cfg: configuration file retrieved from hydra framework
    """
    main_utils.init(cfg)

    logger = TrainLogger(exp_name_prefix=cfg['main']['experiment_name_prefix'], logs_dir=cfg['main']['paths']['logs'])
    logger.write(OmegaConf.to_yaml(cfg))
    # Set seed for results reproduction
    main_utils.set_seed(cfg['main']['seed'])

    # Load dataset

    image_dir = cfg['main']['paths']['train_images_dir']
    img_path = f'{image_dir}/COCO_train2014'
    questions = r'/datashare/v2_OpenEnded_mscoco_train2014_questions.json'
    target_path = cfg['main']['paths']['train']
    train_dataset = MyDataset(image_dir, questions, img_path, target_path, train=True)

    image_dir = cfg['main']['paths']['val_images_dir']
    img_path = f'{image_dir}/COCO_val2014'
    questions = r'/datashare/v2_OpenEnded_mscoco_val2014_questions.json'
    target_path = cfg['main']['paths']['validation']
    val_dataset = MyDataset(image_dir, questions, img_path, target_path, train=False)

    num_workers = cfg['main']['num_workers']
    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], num_workers=num_workers)
    eval_loader = DataLoader(val_dataset, cfg['train']['batch_size'], num_workers=num_workers)

    # Init model
    output_dim = len(train_dataset.train_lang.label2ans)
    vocab_size = train_dataset.train_lang.num_of_words
    embedding_dim = 250
    model = MyModel(embedding_dim=embedding_dim, hidden_dim=cfg['train']['num_hid'],
                    vocab_size=vocab_size, output_dim=output_dim, dropout=cfg['train']['dropout']).to(device)

    # Run model
    train_params = train_utils.get_train_params(cfg)
    train(model, train_loader, eval_loader, train_params)


def plot_graphs():
    metrics = torch.load('best_metrics.pkl', map_location=device)
    for key in metrics:
        print(key, metrics[key])
    plt.plot(metrics['train_loss'], '-o', label='Train loss')
    plt.plot(metrics['val_loss'], '-o', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss in epochs')
    plt.show()
    plt.figure()

    plt.plot(1 - torch.tensor(metrics['train_accuracy']), '-o', label='Train accuracy')
    plt.plot(1 - torch.tensor(metrics['val_accuracy']), '-o', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error in epochs')
    plt.show()


if __name__ == '__main__':
    main()
    plot_graphs()
