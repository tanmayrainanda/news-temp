import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from sklearn.model_selection import train_test_split
import spacy
import random
from rouge import Rouge
import wandb

wandb.init(project="scan-summarizer")
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureRichEncoder(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers=1, dropout=0.1):
        super(FeatureRichEncoder, self).__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embeddings = nn.ModuleList([nn.Embedding(input_size, hidden_size, padding_idx=0)
                                         for input_size in input_sizes])

        self.encoder = nn.LSTM(hidden_size * len(input_sizes), hidden_size, num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, inputs, hidden):
        embedded = [emb(inp) for emb, inp in zip(self.embeddings, inputs)]
        embedded = torch.cat(embedded, dim=-1)
        embedded = self.dropout(embedded)
        outputs, hidden = self.encoder(embedded, hidden)
        return outputs, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device))

class HierarchicalAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(HierarchicalAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.word_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.sentence_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(hidden_size + output_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
        self.generator = nn.Linear(hidden_size * 3, output_size)
        self.pointer_switch = nn.Linear(hidden_size * 3, 1)

    def forward(self, decoder_input, encoder_outputs, decoder_hidden):
        batch_size = decoder_input.size(0)
        seq_len = encoder_outputs.size(1)

        # Hierarchical attention computation
        word_attns = torch.tanh(self.word_attention(encoder_outputs))
        word_attns = word_attns.view(batch_size, seq_len, -1)
        word_attns = F.softmax(word_attns, dim=2)

        sentence_attns = torch.tanh(self.sentence_attention(encoder_outputs))
        sentence_attns = F.softmax(sentence_attns, dim=1)
        sentence_attns = sentence_attns.unsqueeze(2)

        combined_attns = word_attns * sentence_attns
        combined_attns = combined_attns.sum(dim=1)
        context_vector = torch.bmm(combined_attns.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)

        # Decoder step
        decoder_input = torch.cat((decoder_input, context_vector), dim=-1)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        # Pointer switch calculation
        combined_vector = torch.cat((decoder_output, context_vector), dim=-1)
        pointer_switch = torch.sigmoid(self.pointer_switch(combined_vector))

        # Generator and pointer distributions
        generator_output = F.log_softmax(self.generator(combined_vector), dim=-1)
        pointer_distribution = F.softmax(word_attns.view(batch_size, -1), dim=-1)

        # Combine generator and pointer
        final_distribution = pointer_switch * pointer_distribution + (1 - pointer_switch) * generator_output

        return final_distribution, decoder_hidden

class Summarizer(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Summarizer, self).__init__()
        self.encoder = FeatureRichEncoder(input_sizes, hidden_size, num_layers, dropout)
        self.decoder = HierarchicalAttentionDecoder(hidden_size, output_size, num_layers, dropout)

    def forward(self, inputs, decoder_input, decoder_hidden, encoder_hidden=None):
        batch_size = inputs[0].size(0)
        if encoder_hidden is None:
            encoder_hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden)
        final_distribution, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden)
        return final_distribution, decoder_hidden

num_epochs = 10

nlp = spacy.load("en_core_web_sm")

# Define the fields for input and output sequences
article_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, include_lengths=True, batch_first=True)
summary_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, init_token='<sos>', eos_token='<eos>', include_lengths=True, batch_first=True)

# Load the data from the CSV file
dataset = TabularDataset(
    path='data_with_summaries.csv',  # assuming the CSV file is in the current directory
    format='csv',
    fields=[('article', article_field), ('summary', summary_field)]
)

# Perform train-test split
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(42))

# Build the vocabularies
article_field.build_vocab(train_data, max_size=50000)
summary_field.build_vocab(train_data, max_size=25000)

# Create the data iterators
train_iter = BucketIterator(
    train_data,
    batch_size=32,
    sort_key=lambda x: len(x.article),
    sort_within_batch=True,
    device=device
)

test_iter = BucketIterator(
    test_data,
    batch_size=32,
    sort_within_batch=True,
    device=device
)

# Define the model
input_sizes = [len(article_field.vocab)] * len(article_field.vocab.stoi)
hidden_size = 256
output_size = len(summary_field.vocab)
model = Summarizer(input_sizes, hidden_size, output_size).to(device)  # Move model to GPU

# Define the loss function and optimizer
criterion = nn.NLLLoss(ignore_index=summary_field.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters())

# Training loop
wandb.watch(model)
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in train_iter:
        articles, article_lengths = batch.article
        summaries, summary_lengths = batch.summary

        decoder_input = summaries[:, :-1].to(device)  # Move tensors to GPU
        decoder_target = summaries[:, 1:].to(device)

        decoder_hidden = model.decoder.initHidden(batch.batch_size)
        optimizer.zero_grad()

        final_distribution, _ = model(articles.to(device), decoder_input, decoder_hidden)  # Move tensors to GPU

        loss = criterion(final_distribution.view(-1, output_size), decoder_target.contiguous().view(-1))
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
max_summary_length = 100  # Maximum length of generated summaries

def generate_batch_summaries(batch):
    articles, article_lengths = batch.article
    summaries, summary_lengths = batch.summary

    batch_size = articles.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size)

    generated_summaries = []
    with torch.no_grad():
        for i in range(batch_size):
            encoder_outputs, encoder_hidden = model.encoder([articles[i].to(device)], encoder_hidden)  # Move tensors to GPU
            decoder_input = torch.tensor([[summary_field.vocab.stoi['<sos>']]], device=device)
            decoder_hidden = encoder_hidden

            summary = []
            for _ in range(max_summary_length):
                final_distribution, decoder_hidden = model.decoder(decoder_input, encoder_outputs, decoder_hidden)
                word_id = final_distribution.argmax(dim=-1).item()

                if word_id == summary_field.vocab.stoi['<eos>']:
                    break
                else:
                    summary.append(word_id)
                    decoder_input = torch.tensor([[word_id]], device=device)

            summary = [summary_field.vocab.itos[word_id] for word_id in summary]
            generated_summaries.append(' '.join(summary))

    return generated_summaries

for batch in test_iter:
    generated_summaries = generate_batch_summaries(batch)
    reference_summaries = [' '.join(summary_field.vocab.itos[token_id] for token_id in summary) for summary in batch.summary]

    all_generated_summaries.extend(generated_summaries)
    all_reference_summaries.extend(reference_summaries)

scores = rouge.get_scores(all_generated_summaries, all_reference_summaries, avg=True)
print(scores)
wandb.log("Rouge Score:",scores)
#Save the model
torch.save(model.state_dict(), 'summarizer.pth')
wandb.save('summarizer.pth')
wandb.finish()