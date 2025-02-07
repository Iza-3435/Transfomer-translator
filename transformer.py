import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
import nltk
nltk.download('punkt_tab')



# Model parameters
embedding_size = 128
num_heads = 4
num_layers = 4
hidden_size = 128
num_epochs = 100
batch_size = 4
max_seq_len = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Remove punctuation and tokenize
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab, max_len):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_tokens = [self.src_vocab.get('<start>')] + \
                     [self.src_vocab.get(word, self.src_vocab['<unk>']) for word in preprocess_text(self.src_sentences[idx])] + \
                     [self.src_vocab.get('<end>')]
        tgt_tokens = [self.tgt_vocab.get('<start>')] + \
                     [self.tgt_vocab.get(word, self.tgt_vocab['<unk>']) for word in preprocess_text(self.tgt_sentences[idx])] + \
                     [self.tgt_vocab.get('<end>')]

        src_tokens = src_tokens[:self.max_len] + [self.src_vocab['<pad>']] * (self.max_len - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_len] + [self.tgt_vocab['<pad>']] * (self.max_len - len(tgt_tokens))

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, num_heads, num_layers, hidden_size):
        super(Transformer, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, embedding_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embedding_size, tgt_vocab_size)

    def create_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

    def forward(self, src, tgt):
        src_embed = self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_embed = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        tgt_mask = self.create_mask(tgt.size(1))

        enc_output = self.encoder(src_embed)
        dec_output = self.decoder(tgt_embed, enc_output, tgt_mask=tgt_mask)
        output = self.fc_out(dec_output)
        return output


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])

            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


def translate_sentence(model, sentence, src_vocab, tgt_vocab):
    model.eval()
    with torch.no_grad():
        # Convert sentence to tensor
        tokens = torch.tensor([[
                                   src_vocab.get('<start>')] +
                               [src_vocab.get(word, src_vocab['<unk>']) for word in preprocess_text(sentence)] +
                               [src_vocab.get('<end>')]
                               ]).to(device)

        # Add padding
        if tokens.size(1) < max_seq_len:
            padding = torch.tensor([[src_vocab['<pad>']] * (max_seq_len - tokens.size(1))]).to(device)
            tokens = torch.cat([tokens, padding], dim=1)

        # Initialize target with start token
        output_tokens = [tgt_vocab['<start>']]

        # Generate translation
        for _ in range(max_seq_len):
            tgt_tensor = torch.tensor([output_tokens]).to(device)
            output = model(tokens, tgt_tensor)

            # Get next word
            next_token = output[0, -1].argmax().item()
            output_tokens.append(next_token)

            if next_token == tgt_vocab['<end>']:
                break

        # Convert tokens to words
        rev_vocab = {v: k for k, v in tgt_vocab.items()}
        words = []
        for token in output_tokens[1:]:  # Skip start token
            word = rev_vocab[token]
            if word == '<end>':
                break
            words.append(word)

        return ' '.join(words)


def evaluate_bleu(reference, hypothesis):
    reference = [preprocess_text(reference)]
    hypothesis = preprocess_text(hypothesis)
    return sentence_bleu(reference, hypothesis)


# Vocabulary
src_vocab = {
    '<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3,
    'ich': 4, 'bin': 5, 'ein': 6, 'student': 7,
    'wie': 8, 'geht': 9, 'es': 10, 'dir': 11,
    'hallo': 12, 'danke': 13, 'bitte': 14, 'tschüss': 15,
    'ja': 16, 'nein': 17, 'gut': 18, 'schlecht': 19,
    'sprechen': 20, 'deutsch': 21, 'englisch': 22, 'lernen': 23,
    'heute': 24, 'morgen': 25, 'gestern': 26
}

tgt_vocab = {
    '<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3,
    'i': 4, 'am': 5, 'a': 6, 'student': 7,
    'how': 8, 'are': 9, 'you': 10,
    'hello': 11, 'thank': 12, 'please': 13, 'goodbye': 14,
    'yes': 15, 'no': 16, 'good': 17, 'bad': 18,
    'speak': 19, 'german': 20, 'english': 21, 'learn': 22,
    'today': 23, 'tomorrow': 24, 'yesterday': 25
}

# Training data
train_pairs = [
    ('ich bin ein Student', 'I am a student'),
    ('wie geht es dir', 'how are you'),
    ('hallo', 'hello'),
    ('danke', 'thank'),
    ('bitte', 'please'),
    ('tschüss', 'goodbye'),
    ('ja', 'yes'),
    ('nein', 'no'),
    ('gut', 'good'),
    ('schlecht', 'bad'),
    ('ich spreche deutsch', 'I speak german'),
    ('ich lerne englisch', 'I learn english'),
    ('heute', 'today'),
    ('morgen', 'tomorrow'),
    ('gestern', 'yesterday')
]

# Create dataset and dataloader
dataset = TranslationDataset(
    [pair[0] for pair in train_pairs],
    [pair[1] for pair in train_pairs],
    src_vocab, tgt_vocab, max_seq_len
)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = Transformer(
    len(src_vocab), len(tgt_vocab),
    embedding_size, num_heads,
    num_layers, hidden_size
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Interactive translation mode
while True:
    user_input = input("Enter German text: ").strip()

    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    if not user_input:
        print("Please enter some text to translate.")
        continue

    translation = translate_sentence(model, user_input, src_vocab, tgt_vocab)
    print(f"German: {user_input}")
    print(f"English (model): {translation}")

    # BLEU score
    reference = [pair[1] for pair in train_pairs if pair[0] == user_input][0]
    bleu_score = evaluate_bleu(reference, translation)
    print(f"BLEU score: {bleu_score:.4f}")
