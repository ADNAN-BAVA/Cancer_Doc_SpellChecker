import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertForMaskedLM, DistilBertTokenizer, AdamW, get_linear_schedule_with_warmup
from preprocess_corpus import preprocess_corpus

class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length, padding='max_length')
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mask_prob) * (input_ids != self.tokenizer.cls_token_id) * (input_ids != self.tokenizer.sep_token_id) * (input_ids != self.tokenizer.pad_token_id)
        input_ids[mask_arr] = self.tokenizer.mask_token_id
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def calculate_accuracy(preds, labels):
    pred_ids = torch.argmax(preds, dim=-1)
    non_masked = (labels != -100)
    correct_preds = (pred_ids == labels) & non_masked
    return correct_preds.sum().item() / non_masked.sum().item()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            accuracy = calculate_accuracy(logits, labels)
            total_accuracy += accuracy

    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def train_bert(corpus_file, output_dir, epochs=2, batch_size=8, val_split=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data, _ = preprocess_corpus(corpus_file)
    dataset = MaskedTextDataset(data, tokenizer)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        print(f"GPU is available: {gpu_name}")
    else:
        print("No GPU available.")
    
    # Use DistilBERT for Masked Language Modeling
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        total_train_loss = 0

        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
        
            if (step + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item()}")

        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {total_train_loss / len(train_loader)}")
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_bert("cancer_new.txt", "./fine_tuned_distilbert1")
