import sys
from transformers import AdamW
from transformers import BertTokenizer
import torch
import csv
import json
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from bert_models import IntentBertModel

class dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

categories = []

def read_split(path):
    texts = []
    labels = []
    with open(path, newline = '') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[1] != 'category':
                texts.append(row[0])
                labels.append(cateTonum[row[1]])
    return texts, labels

def validation(model, device, valid_loader):
    model.eval()
    loss_total = 0
    total = 0
    hit = 0
    with torch.no_grad():
        for data in tqdm(valid_loader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
            loss = outputs[1]
            loss_total += loss
            predicted = torch.argmax(outputs[0], 1)
            total += labels.size(0)
            hit += (predicted == labels).sum().item()
    #print('vala_acc = {}'.format(hit/total))
    return loss_total / len(valid_loader)
print(sys.argv[1])
train_data_file = "./data/" + sys.argv[1] + "/train_10.csv"
test_data_file = "./data/" + sys.argv[1] +"/test.csv"
val_data_file = "./data/" + sys.argv[1] + "/val.csv"
category_list = "./data/" + sys.argv[1] + "/categories.json" 


cateTonum = {}
with open(category_list, 'r') as f:
    categories = json.load(f)
for index in range(len(categories)):
    cateTonum[categories[index]] = index
f.close()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_texts, train_labels = read_split(train_data_file)

test_texts, test_labels = read_split(test_data_file)
val_texts, val_labels = read_split(val_data_file)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = dataset(train_encodings, train_labels)
val_dataset = dataset(val_encodings, val_labels)
test_dataset = dataset(test_encodings, test_labels)

model = IntentBertModel('bert-base-uncased', 0.1, len(cateTonum))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optim = AdamW(model.parameters(), lr = 5e-5)
train2_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

the_min_loss = 100
patience = 10
trigger_times = 0
candid = 0
for epoch in range(100):
    train_loss = 0
    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        print(batch)
        break
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
        loss = outputs[1]
        train_loss += loss
        #print('loss = {}'.format(loss))
        loss.backward()
        optim.step()
    break
    the_current_loss = validation(model, device, val_loader)
    #print('the_current_loss:', the_current_loss)
    if the_current_loss > the_min_loss:
        trigger_times +=1
        #print('trigger times:',trigger_times)
        if trigger_times >= patience:
            #print('Early stopping')
            break
    else:
        trigger_times = 0
        the_min_loss = the_current_loss
        torch.save(model.state_dict(), 'current.pkl'.format(candid))
        candid += 1
print('test')
hit = 0
total = 0
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
for i in range(candid):
    hit = 0
    total = 0
    model.load_state_dict(torch.load('./current.pkl'.format(i)))
    model.eval()
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
        predicted = torch.argmax(outputs[0], 1)
        total += labels.size(0)
        hit += (predicted == labels).sum().item()
    print('test_acc = {}'.format(hit/total))    
