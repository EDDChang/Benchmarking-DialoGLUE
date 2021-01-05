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
from transformers import Trainer, TrainingArguments
from bert_models_prototype import IntentBertModel
import numpy as np
from torch.nn import functional as F
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

def sampler(n_class, n_sample):
    return np.sort(np.random.choice(np.arange(0, n_class), n_sample, replace=False))

def euclidean_dist(x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)
    return torch.pow(x-y, 2).sum(2)

def SocialDist(prototypes, Nc):
    
    a = prototypes.expand(Nc,Nc,768)
    b = torch.reshape(prototypes, (Nc,1,768)).expand(Nc,Nc,768)
    eudist = torch.pow(a-b,2).sum(2).mean()/(2*Nc)
    print(eudist)
    return torch.exp(-eudist)

def loss_fn(epoch, inputs, targets, Nc, n_support, n_query, device):
    
    n_classes = len(inputs)
    support_set = sampler(n_support+n_query, n_support)
    query_set = list(set([i for i in range(n_support+n_query)])-set(support_set))
    
    prototypes = []
    
    for classes in inputs:
        prototypes.append(classes[support_set].mean(0))    
    prototypes = torch.stack(prototypes)
   
    extra_loss = SocialDist(prototypes, Nc) if epoch < 100 else torch.tensor(0)

    query_samples = []
    for classes in inputs:
        query_samples.append(classes[query_set])
    query_samples = torch.stack(query_samples)
    query_samples = torch.reshape(query_samples, (Nc*n_query,768))
    #print('right!', query_samples.size(), prototypes.size()) 
    dists = euclidean_dist(query_samples, prototypes)
    
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    
    target_inds = torch.arange(0, n_classes).to(device)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #print('target_inds', target_inds)
       
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    #print('y_hat', y_hat)
    #print('y_hat_shape', y_hat.size())
        
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val + extra_loss , acc_val

def prototype_valid(model, device, train_loader, val_loader):
    
    with torch.no_grad():
        val_sample = np.random.choice(np.arange(0, 4), 1, replace=False)
        n_classes = len(train_loader)
        prototypes = [torch.zeros(768).to(device) for i in range(n_classes)]
        prototypes = torch.stack(prototypes)
        prototypes = torch.reshape(prototypes, (n_classes,768))
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
            prototypes[labels[0].item()] = outputs.mean(0)
        
        query = [torch.zeros(5, 768).to(device) for i in range(n_classes)]
        query = torch.stack(query)
        query = torch.reshape(query, (n_classes, 5, 768))
        sample_counter = 0
        for batch in val_loader:
            if sample_counter % 4 != val_sample[0]:
                sample_counter += 1
                continue
            sample_counter +=1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
            query[labels[0].item()] = outputs
        
        
        #print(query.size())
        query = torch.reshape(query, (n_classes*5, 768)) 
        #print(query.size(), prototypes.size())
        dists = euclidean_dist(query, prototypes)
    
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    
        target_inds = torch.arange(0, n_classes).to(device)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, 5, 1).long()
        #print('target_inds', target_inds)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        #print('y_hat', y_hat)
        #print('y_hat_shape', y_hat.size())
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val

if __name__ == "__main__":
    
    print(sys.argv[1])    
    train_data_file = "./data/" + sys.argv[1] + "/" + sys.argv[2]
    test_data_file = "./data/" + sys.argv[1] +"/test.csv"
    val_data_file = "./data/" + sys.argv[1] + "/val.csv"
    category_list = "./data/" + sys.argv[1] + "/categories.json" 

    n_support = 5 if sys.argv[2] == 'train_10.csv' else 3
    n_query = 5 if sys.argv[2] == 'train_10.csv' else 2
    Nc = 30
    
    categories = []
    cateTonum = {}
    cate_list = []

    with open(category_list, 'r') as f:
        categories = json.load(f)
        cate_list = categories
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

    train_loader = DataLoader(train_dataset, batch_size=10 if sys.argv[2] == 'train_10.csv' else 5, shuffle=False)
    #tmp_sample = sampler(len(cate_list), Nc)
    #print(tmp_sample)
    #for batch in train_loader:
    #    if batch['labels'][0] in tmp_sample:
    #        print(batch['labels'])
    #sys.exit(0)
    optim = AdamW(model.parameters(), lr = 5e-5)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

    the_min_loss = 100
    patience = 50
    trigger_times = 0
    candid = 0
    
    Pwarm_up = 100
    for epoch in range(1000):
        train_loss = 0
        model.train()
        sample_list = []
        labels_list = []
        tmp_sample = sampler(len(cate_list), Nc)
        #print(tmp_sample)
        optim.zero_grad()
        for batch in tqdm(train_loader):
            if batch['labels'][0].item() in tmp_sample:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
                sample_list.append(outputs)
                labels_list.append(labels)
        loss, acc = loss_fn(epoch,sample_list, labels_list, Nc, n_support, n_query, device)
        #sys.exit(0)
        print('train loss:', loss.item(), 'train acc:', acc.item())
        loss.backward()
        optim.step()
        if epoch < Pwarm_up:
            continue
        the_current_loss, the_current_acc = prototype_valid(model, device, train_loader, val_loader)
        print('valid loss:', the_current_loss.item(), 'valid acc', the_current_acc.item())
        if the_current_loss > the_min_loss:
            trigger_times +=1
            print('trigger times:',trigger_times)
            if trigger_times >= patience:
                print('Early stopping')
                break
        else:
            trigger_times = 0
            the_min_loss = the_current_loss
            torch.save(model.state_dict(), 'current.pkl')
            candid = 1
        
    print('test')
    hit = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if candid == 1:
        model.load_state_dict(torch.load('./current.pkl'))
    model.eval()

    wrong_predict_dict = [{} for j in range(len(cateTonum))]
    acc_list = [0.0 for i in range(len(cate_list))]

    current_label = 0
    current_hit = 0
    current_total = 0
    with torch.no_grad():    
        
        prototypes = [torch.zeros(768).to(device) for i in range(len(train_loader))]
        prototypes = torch.stack(prototypes)
        prototypes = torch.reshape(prototypes, (len(train_loader),768))
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
            prototypes[labels[0].item()] = outputs.mean(0)
        
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            if labels[0] != current_label and current_total != 0:
                acc_list[current_label] = current_hit/current_total
                current_hit = 0
                current_total=0
                current_label = labels[0]
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids,intent_label=labels)
            outputs = outputs.expand(len(train_loader),768)
            predicted = torch.argmin(torch.pow(outputs-prototypes, 2).sum(1))
            total += 1
            current_total+=1
            #print(predicted, labels)
            if predicted.item() == labels[0].item():
                hit += 1 
                current_hit += 1
            else:
                if cate_list[predicted.item()] in wrong_predict_dict[labels[0]]:
                    wrong_predict_dict[labels[0]][cate_list[predicted.item()]] = wrong_predict_dict[labels[0]][cate_list[predicted.item()]] + 1
                else:
                    wrong_predict_dict[labels[0]][cate_list[predicted.item()]] = 1
                    #print('predict label: {} real label: {}'.format(predicted.item(), labels[0].item()))
        acc_list[current_label] = current_hit/current_total
        
        for i in range(len(wrong_predict_dict)):
            print("{:<30}{:.3f}{:<5}{}".format(cate_list[i], acc_list[i],"",sorted(wrong_predict_dict[i].items(), key=lambda x:x[1], reverse=True)))
        print('test_acc = {}'.format(hit/total))    
    sys.exit(0)
