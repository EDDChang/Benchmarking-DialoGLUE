import sys
import torch
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss 
from torch.nn import Dropout
from torch.nn import CosineSimilarity
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Any

class BertPretrain(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str):
        super(BertPretrain, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained(model_name_or_path)

    def forward(self, 
                input_ids: torch.tensor,
                mlm_labels: torch.tensor):
        outputs = self.bert_model(input_ids, masked_lm_labels=mlm_labels)
        return outputs[0]

class IntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 labels_info):
        super(IntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        
        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.labels_info = labels_info
        self.alpha = torch.tensor(0.2)
    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None,
                device='cuda'):

        pooled_output = self.bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[1]

        dp = self.dropout(pooled_output) 
        intent_logits = self.intent_classifier(dp)
        eudist = torch.tensor(0.0).to(device)
        cossim_sum = torch.tensor(0.0).to(device)
        if intent_label is not None:
            cos = nn.CosineSimilarity(dim = 0, eps=1e-6)
            labels_embedding_list = self.bert_model(input_ids=self.labels_info['input_ids'].to(device),
                                            attention_mask=self.labels_info['attention_mask'].to(device),
                                            token_type_ids=self.labels_info['token_type_ids'].to(device))[1]
        
            labels_embedding_list = self.dropout(labels_embedding_list) 
        
            #target_embedding = []
            #for label in intent_label:
            #    target_embedding.append(labels_embedding_list[label])
            
            #target_embedding = torch.stack(target_embedding)
            #eudist = torch.pow(dp - target_embedding, 2).sum(1).mean()
            #eudist = eudist
            for i in range(len(input_ids)):
                index = intent_label[i]
                cossim_sum += cos(labels_embedding_list[index], dp[i])
                #eudist += (torch.pow(labels_embedding_list[index] - dp[i], 2).sum()/len(input_ids))
            #eudist /= 25
            cossim_sum /= len(input_ids)
            labels_logits = self.intent_classifier(labels_embedding_list)
        else:
            eudist = torch.tensor(0.0)
            
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
            labels_loss = loss_fct(labels_logits.view(-1, self.num_intent_labels), self.labels_info['labels'].to(device).type(torch.long))
            print(intent_loss, cossim_sum, labels_loss)
        else:
            labels_loss = torch.tensor(0.0)
            intent_loss = torch.tensor(0.0)
        
        return intent_logits , (1-self.alpha)*intent_loss + self.alpha*torch.log(1/cossim_sum)

class SlotBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_slot_labels: int):
        super(SlotBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)
        self.dropout = Dropout(dropout)
        self.num_slot_labels = num_slot_labels
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)

    def encode(self,
               input_ids: torch.tensor,
               attention_mask: torch.tensor,
               token_type_ids: torch.tensor):
        hidden_states, _ = self.bert_model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids)
        return hidden_states

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                slot_labels: torch.tensor = None):
        hidden_states = self.encode(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        return slot_logits, slot_loss

class JointSlotIntentBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path: str,
                 dropout: float,
                 num_intent_labels: int,
                 num_slot_labels: int):
        super(JointSlotIntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_classifier = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(self.bert_model.config.hidden_size, num_slot_labels)

    def forward(self,
                input_ids: torch.tensor,
                attention_mask: torch.tensor,
                token_type_ids: torch.tensor,
                intent_label: torch.tensor = None,
                slot_labels: torch.tensor = None):
        hidden_states, pooled_output = self.bert_model(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)


        intent_logits = self.intent_classifier(self.dropout(pooled_output))
        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        # Compute losses if labels provided
        if slot_labels is not None:
            loss_fct = CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = loss_fct(active_logits, active_labels.type(torch.long))
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1).type(torch.long))
        else:
            slot_loss = torch.tensor(0).cuda() if torch.cuda.is_available() else torch.tensor(0)

        # Compute losses if labels provided
        if intent_label is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label.type(torch.long))
        else:
            intent_loss = torch.tensor(0)

        return intent_logits, slot_logits, intent_loss + slot_loss
