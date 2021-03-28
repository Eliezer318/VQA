import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.train_utils import TrainParams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams):
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95 ** x)
    # TODO try BCELoss
    criterion = nn.BCEWithLogitsLoss()
    train_accuracy_epochs, train_loss_epochs, val_accuracy_epochs, val_loss_epochs = [], [], [], []
    best_epoch = {'score': 0}
    for epoch in tqdm(range(train_params.num_epochs)):
        model.train()
        for images, questions, answers, max_label in tqdm(train_loader):
            images = images.to(device).to(device=device, dtype=dtype)
            questions = questions.to(device=device, dtype=torch.long)
            answers = answers.to(device=device)
            y_hat = model(images, questions)
            loss = criterion(y_hat, answers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        score, loss = evaluate(model, train_loader)
        train_loss_epochs.append(loss)
        train_accuracy_epochs.append(score)
        score, loss = evaluate(model, eval_loader)
        val_loss_epochs.append(loss)
        val_accuracy_epochs.append(score)

        if score > best_epoch['score']:
            best_epoch['score'] = score
            best_epoch['train_loss'] = train_loss_epochs
            best_epoch['train_accuracy'] = train_accuracy_epochs
            best_epoch['val_accuracy'] = val_accuracy_epochs
            best_epoch['val_loss'] = val_loss_epochs
            torch.save(best_epoch, 'best_metrics2.pkl')
            torch.save(model.state_dict(), 'best_weights2.pkl')
            print(f'Validation score is: {best_epoch["val_accuracy"][-1]} train score is: {best_epoch["train_accuracy"][-1]}')


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader):
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    score, loss, max_score_possible = 0, 0, 0
    for i, (images, questions, answers, max_label) in enumerate(tqdm(dataloader)):
        max_score_possible += answers.max(dim=1)[0].sum()
        images = images.to(device).to(device=device, dtype=dtype)
        questions = questions.to(device=device, dtype=torch.long)
        answers = answers.to(device=device)
        y_hat = model(images, questions)
        loss += criterion(y_hat, answers).item()
        predictions = y_hat.argmax(dim=1)
        score += answers[torch.arange(answers.shape[0]), predictions].sum().item()

    n = len(dataloader.dataset)
    return score/n, loss/n
