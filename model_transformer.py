import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import random
from rouge import Rouge
import math
import wandb

wandb.init(project="scan-summarizer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'vocab_size': 15000,
    'embedding_dim': 300,
    'hidden_dim': 64,
    'num_layers': 1,
    'dropout': 0.1,
    'pad_idx': 0
}

max_summary_length = 100

print("checkpoint 1")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.pos_decoder = PositionalEncoding(hidden_size, dropout)
        decoder_layers = nn.TransformerDecoderLayer(hidden_size, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt) * math.sqrt(self.hidden_size)
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output

class Summarizer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, dropout=0.1):
        super(Summarizer, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = TransformerDecoder(output_size, hidden_size, num_layers, dropout)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

num_epochs = 10

nlp = spacy.load("en_core_web_sm")

print("checkpoint 2")

# Define the fields for input and output sequences
article_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, include_lengths=True, batch_first=True)
summary_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, init_token='<sos>', eos_token='<eos>', include_lengths=True, batch_first=True)

# Load the data from the CSV file
dataset = TabularDataset(
    path='data_cleaned.csv',  # assuming the CSV file is in the current directory
    format='csv',
    fields=[('article', article_field), ('summary', summary_field)]
)

# Perform train-test split
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(42))

print("checkpoint 3")

# Reduce maximum vocabulary size
article_field.build_vocab(train_data, max_size=10000)  # Adjust max_size as needed
summary_field.build_vocab(train_data, max_size=5000)  # Adjust max_size as needed

# Define the data iterators with manual iteration
batch_size = 8  # Reduced batch size
partial_train_data = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
partial_test_data = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
train_iter = [iter(BucketIterator(partial, batch_size=batch_size, sort_key=lambda x: len(x.article), sort_within_batch=True, device=device)) for partial in partial_train_data]
test_iter = [iter(BucketIterator(partial, batch_size=batch_size, sort_key=lambda x: len(x.article), sort_within_batch=True, device=device, train=False)) for partial in partial_test_data]

print("checkpoint 4")

# Define the model
input_size = len(article_field.vocab)
output_size = len(summary_field.vocab)
hidden_size = config['hidden_dim']
num_layers = config['num_layers']
dropout = config['dropout']
model = Summarizer(input_size, output_size, hidden_size, num_layers, dropout).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=summary_field.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters())

print("checkpoint 5")

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for train_iter_partial in train_iter:
        for train_batch in train_iter_partial:
            articles, article_lengths = train_batch.article
            summaries, summary_lengths = train_batch.summary

            articles = articles.to(device)
            summaries = summaries.to(device)

            optimizer.zero_grad()

            output = model(articles, summaries[:, :-1])  # Feed summary without <eos> token
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            summaries = summaries[:, 1:].contiguous().view(-1)  # Target without <sos> token

            loss = criterion(output, summaries)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    print(f'Epoch: {epoch+1}, Train Loss: {epoch_loss/len(train_iter)}')
    wandb.log({"train_loss": epoch_loss/len(train_iter)})

# Evaluate the model on the test set
model.eval()
rouge = Rouge()
all_generated_summaries = []
all_reference_summaries = []

# Manually iterate over test set
for test_iter_partial in test_iter:
    for test_batch in test_iter_partial:
        articles, article_lengths = test_batch.article
        summaries, summary_lengths = test_batch.summary

        articles = articles.to(device)
        summaries = summaries.to(device)

        generated_summaries = model(articles, summaries[:, :-1])
        generated_summaries = generated_summaries.argmax(dim=-1)
        generated_summaries = generated_summaries.tolist()

        reference_summaries = summaries[:, 1:].tolist()

        all_generated_summaries.extend(generated_summaries)
        all_reference_summaries.extend(reference_summaries)
