from torch import nn


class MyModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim, dropout: float = 0.2):
        super(MyModel, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Dropout2d(dropout), nn.Conv2d(3, 256, 3, 1), nn.LeakyReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout), nn.Conv2d(256, 128, 4, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(True),
            nn.Dropout2d(dropout), nn.Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(True),
            nn.Dropout2d(dropout), nn.Conv2d(128, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim), nn.LeakyReLU(True),
        )
        self.question_model = SentenceModel(embedding_dim, hidden_dim, vocab_size, 0.6)
        self.final_model = nn.Sequential(
            nn.Dropout(0.6), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Dropout(0.6), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, img, sentence):
        hidden = self.question_model(sentence)
        img = self.conv_model(img)
        return self.final_model(img * hidden)


class SentenceModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, dropout: float = 0.2):
        super(SentenceModel, self).__init__()
        self.question_model = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.LSTM(embedding_dim, embedding_dim, num_layers=4, dropout=dropout, bidirectional=True),
        )
        self.sentence_final_model = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim), nn.LeakyReLU(),
            nn.Linear(embedding_dim, hidden_dim), nn.LeakyReLU(),
        )

    def forward(self, sentence):
        return self.sentence_final_model(self.question_model(sentence)[0].sum(dim=1))





