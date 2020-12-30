import random
from transformers import AdamW
from transformers import BertTokenizer
import torch
import csv
import json
import sys
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from bert_models_multi import IntentBertModel

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

train_data_file_1 = sys.argv[1]
test_data_file_1 = "./data/banking/test.csv"
val_data_file_1 = "./data/banking/val.csv"
category_list_1 = "./data/banking/categories.json" 

train_data_file_2 = sys.argv[2]
test_data_file_2 = "./data/hwu/test.csv"
val_data_file_2 = "./data/hwu/val.csv"
category_list_2 = "./data/hwu/categories.json" 

train_data_file_3 = sys.argv[3]
test_data_file_3 = "./data/clinc/test.csv"
val_data_file_3 = "./data/clinc/val.csv"
category_list_3 = "./data/clinc/categories.json" 

def read_split(path, cateTonum):
    texts = []
    labels = []
    with open(path, newline = '') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[1] != 'category':
                texts.append(row[0])
                labels.append(cateTonum[row[1]])
    return texts, labels

def validation(model, device, valid_loader_1, valid_loader_2, valid_loader_3):
    model.eval()
    
    loss_total = 0
    total_1 = 0
    total_2 = 0
    total_3 = 0
    hit_1 = 0
    hit_2 = 0
    hit_3 = 0
    
    with torch.no_grad():
        for data in tqdm(valid_loader_1):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids_1 = input_ids, attention_mask_1=attention_mask, token_type_ids_1 = token_type_ids,intent_label_1=labels)
            loss = outputs[3]
            loss_total += loss
            predicted = torch.argmax(outputs[0], 1)
            total_1 += labels.size(0)
            hit_1 += (predicted == labels).sum().item()
    #print('Banking77: vala_acc_1 = {}'.format(hit_1/total_1))
    
    with torch.no_grad():
        for data in tqdm(valid_loader_2):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids_2 = input_ids, attention_mask_2=attention_mask, token_type_ids_2 = token_type_ids,intent_label_2=labels)
            loss = outputs[4]
            loss_total += loss
            predicted = torch.argmax(outputs[1], 1)
            total_2 += labels.size(0)
            hit_2 += (predicted == labels).sum().item()
    #print('Hwu: vala_acc_2 = {}'.format(hit_2/total_2))
    
    with torch.no_grad():
        for data in tqdm(valid_loader_3):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)
            outputs = model(input_ids_3 = input_ids, attention_mask_3=attention_mask, token_type_ids_3 = token_type_ids,intent_label_3=labels)
            loss = outputs[5]
            loss_total += loss
            predicted = torch.argmax(outputs[2], 1)
            total_3 += labels.size(0)
            hit_3 += (predicted == labels).sum().item()
    #print('Clinc: vala_acc_3 = {}'.format(hit_3/total_3))
   
    return loss_total / (len(valid_loader_1) + len(valid_loader_2) + len(valid_loader_3)), ((hit_1/total_1) + (hit_2/total_2) + (hit_3/total_3))/3

def getDataSet(train: str, test: str, val: str, cateTonum):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_texts, train_labels = read_split(train, cateTonum)
    val_texts, val_labels = read_split(val, cateTonum)    
    test_texts, test_labels = read_split(test, cateTonum)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    
    train_dataset = dataset(train_encodings, train_labels)
    val_dataset = dataset(val_encodings, val_labels)
    test_dataset = dataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset

def getCateToNum(category_list):
    cateTonum = {}
    with open(category_list, 'r') as f:
        categories = []
        categories = json.load(f)
    for index in range(len(categories)):
        cateTonum[categories[index]] = index
    f.close()
    return cateTonum
def scheduler_same_epoch(data_1, data_2, data_3):
   index_list = [1 for i in range(len(data_1))] + [2 for i in range(len(data_2))] + [3 for i in range(len(data_3))]
   random.shuffle(index_list)
   return index_list

def scheduler_same_batch(data_1, data_2, data_3):
    length = max(len(data_1), len(data_2), len(data_3))
    index_list = [1 for i in range(length)] + [2 for i in range(length)] + [3 for i in range(length)]
    random.shuffle(index_list)
    return index_list
if __name__ == "__main__":
    
    train_data_file_1 = sys.argv[1]

    train_data_file_2 = sys.argv[2]

    train_data_file_3 = sys.argv[3]


    print(sys.argv[1], sys.argv[2], sys.argv[3])
    cateTonum_1 = getCateToNum(category_list_1)
    cateTonum_2 = getCateToNum(category_list_2)
    cateTonum_3 = getCateToNum(category_list_3)

    train_dataset_1, val_dataset_1, test_dataset_1 = getDataSet(train_data_file_1, test_data_file_1, val_data_file_1, cateTonum_1)
    train_dataset_2, val_dataset_2, test_dataset_2 = getDataSet(train_data_file_2, test_data_file_2, val_data_file_2, cateTonum_2)
    train_dataset_3, val_dataset_3, test_dataset_3 = getDataSet(train_data_file_3, test_data_file_3, val_data_file_3, cateTonum_3)

    model = IntentBertModel('bert-base-uncased', 0.1, len(cateTonum_1), len(cateTonum_2), len(cateTonum_3))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    optim = AdamW(model.parameters(), lr = 5e-5)
    
    train_loader_1 = DataLoader(train_dataset_1, batch_size=16, shuffle=True)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=64, shuffle=False)
    
    train_loader_2 = DataLoader(train_dataset_2, batch_size=16, shuffle=True)
    val_loader_2 = DataLoader(val_dataset_2, batch_size=64, shuffle=False)

    train_loader_3 = DataLoader(train_dataset_3, batch_size=16, shuffle=True)
    val_loader_3 = DataLoader(val_dataset_3, batch_size=64, shuffle=False)

    #print(len(train_loader_1), len(train_loader_2), len(train_loader_3))
    #print(len(val_loader_1), len(val_loader_2), len(val_loader_3))

    the_min_loss = 100
    the_max_acc = 0
    patience = 10
    trigger_times = 0
    candid = 0
    
    i_1 = iter(train_loader_1)
    i_2 = iter(train_loader_2)
    i_3 = iter(train_loader_3)
    
    data_1 = next(i_1)
    data_2 = next(i_2)
    data_3 = next(i_3)

    for epoch in range(100):
        train_loss = 0
        #i_1 = iter(train_loader_1)
        #i_2 = iter(train_loader_2)
        #i_3 = iter(train_loader_3)

        model.train()
        
        #data_1 = next(i_1)
        #data_2 = next(i_2)
        #data_3 = next(i_3)

        schedule = scheduler_same_batch(train_loader_1, train_loader_2, train_loader_3)
        for i in tqdm(schedule):
            if i == 1:
                optim.zero_grad()
            
                input_ids = data_1['input_ids'].to(device)
                attention_mask = data_1['attention_mask'].to(device)
                token_type_ids = data_1['token_type_ids'].to(device)
                labels = data_1['labels'].to(device)
                
                outputs = model(input_ids_1 = input_ids, attention_mask_1=attention_mask, token_type_ids_1 = token_type_ids,intent_label_1=labels)
                loss = outputs[3]
                train_loss += loss
                loss.backward()
                optim.step()
                try:
                    data_1 = next(i_1)
                except StopIteration:
                    train_loader_1 = DataLoader(train_dataset_1, batch_size=16, shuffle=True)
                    i_1 = iter(train_loader_1)
                    data_1 = next(i_1)
                    pass
            elif i == 2:
                optim.zero_grad()
            
                input_ids = data_2['input_ids'].to(device)
                attention_mask = data_2['attention_mask'].to(device)
                token_type_ids = data_2['token_type_ids'].to(device)
                labels = data_2['labels'].to(device)
                
                outputs = model(input_ids_2 = input_ids, attention_mask_2=attention_mask, token_type_ids_2 = token_type_ids,intent_label_2=labels)
                loss = outputs[4]
                train_loss += loss
                loss.backward()
                optim.step()
                try:
                    data_2 = next(i_2)
                except StopIteration:
                    train_loader_2 = DataLoader(train_dataset_2, batch_size=16, shuffle=True)
                    i_2 = iter(train_loader_2)
                    data_2 = next(i_2)
                    pass
            else:
                optim.zero_grad()
            
                input_ids = data_3['input_ids'].to(device)
                attention_mask = data_3['attention_mask'].to(device)
                token_type_ids = data_3['token_type_ids'].to(device)
                labels = data_3['labels'].to(device)
                
                outputs = model(input_ids_3 = input_ids, attention_mask_3=attention_mask, token_type_ids_3 = token_type_ids,intent_label_3=labels)
                loss = outputs[5]
                train_loss += loss
                loss.backward()
                optim.step()
                try:
                    data_3 = next(i_3)
                except StopIteration:
                    train_loader_3 = DataLoader(train_dataset_3, batch_size=16, shuffle=True)
                    i_3 = iter(train_loader_3)
                    data_3 = next(i_3)
                    pass
        the_current_loss, the_current_acc = validation(model, device, val_loader_1, val_loader_2, val_loader_3)
        #print('the_current_loss:', the_current_loss)
        if the_current_acc < the_max_acc:
            trigger_times +=1
            #print('trigger times:',trigger_times)
            if trigger_times >= patience:
                #print('Early stopping')
                break
        else:
            trigger_times = 0
            the_max_acc = the_current_acc
            torch.save(model.state_dict(), 'current{}.pkl'.format(candid))
            candid+=1
    print('Test')
    
    test_loader_1 = DataLoader(test_dataset_1, batch_size=64, shuffle=False)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=64, shuffle=False)
    test_loader_3 = DataLoader(test_dataset_3, batch_size=64, shuffle=False)
    
    model.eval()
    for i in range(candid):
        loss_total = 0
        total_1 = 0
        total_2 = 0
        total_3 = 0
        hit_1 = 0
        hit_2 = 0
        hit_3 = 0
        
        model.load_state_dict(torch.load('./current{}.pkl'.format(i)))
        model.eval()
        
        with torch.no_grad():
            for data in tqdm(test_loader_1):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids_1 = input_ids, attention_mask_1=attention_mask, token_type_ids_1 = token_type_ids,intent_label_1=labels)
                loss = outputs[3]
                loss_total += loss
                predicted = torch.argmax(outputs[0], 1)
                total_1 += labels.size(0)
                hit_1 += (predicted == labels).sum().item()
        print('Banking77: test_acc_1 = {}'.format(hit_1/total_1))
    
        with torch.no_grad():
            for data in tqdm(test_loader_2):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids_2 = input_ids, attention_mask_2=attention_mask, token_type_ids_2 = token_type_ids,intent_label_2=labels)
                loss = outputs[4]
                loss_total += loss
                predicted = torch.argmax(outputs[1], 1)
                total_2 += labels.size(0)
                hit_2 += (predicted == labels).sum().item()
        print('Hwu: test_acc_2 = {}'.format(hit_2/total_2))
    
        with torch.no_grad():
            for data in tqdm(test_loader_3):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                labels = data['labels'].to(device)
                outputs = model(input_ids_3 = input_ids, attention_mask_3=attention_mask, token_type_ids_3 = token_type_ids,intent_label_3=labels)
                loss = outputs[5]
                loss_total += loss
                predicted = torch.argmax(outputs[2], 1)
                total_3 += labels.size(0)
                hit_3 += (predicted == labels).sum().item()
        print('Clinc: test_acc_3 = {}'.format(hit_3/total_3))
        print('Test_acc = {}'.format(((hit_1/total_1) + (hit_2/total_2) + (hit_3/total_3))/3))
