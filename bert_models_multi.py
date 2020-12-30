import torch
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss 
from torch.nn import Dropout
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
                 num_intent_labels_1: int,
                 num_intent_labels_2: int,
                 num_intent_labels_3: int):
        super(IntentBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_intent_labels_1 = num_intent_labels_1
        self.num_intent_labels_2 = num_intent_labels_2
        self.num_intent_labels_3 = num_intent_labels_3
        
        self.intent_classifier_1 = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels_1)
        self.intent_classifier_2 = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels_2)
        self.intent_classifier_3 = nn.Linear(self.bert_model.config.hidden_size, num_intent_labels_3)

    def forward(self,
                input_ids_1: torch.tensor = None,
                attention_mask_1: torch.tensor = None,
                token_type_ids_1: torch.tensor = None,
                
                input_ids_2: torch.tensor = None,
                attention_mask_2: torch.tensor = None,
                token_type_ids_2: torch.tensor = None,
                
                input_ids_3: torch.tensor = None,
                attention_mask_3: torch.tensor = None,
                token_type_ids_3: torch.tensor = None,
                
                intent_label_1: torch.tensor = None,
                intent_label_2: torch.tensor = None,
                intent_label_3: torch.tensor = None):
        intent_logits_1 = None
        intent_logits_2 = None
        intent_logits_3 = None

        intent_loss_1 = torch.tensor(0)
        intent_loss_2 = torch.tensor(0)
        intent_loss_3 = torch.tensor(0)

        if input_ids_1 is not None: 
            pooled_output_1 = self.bert_model(input_ids=input_ids_1,
                                        attention_mask=attention_mask_1,
                                        token_type_ids=token_type_ids_1)[1]
            intent_logits_1 = self.intent_classifier_1(self.dropout(pooled_output_1))

        if input_ids_2 is not None:
            pooled_output_2 = self.bert_model(input_ids=input_ids_2,
                                        attention_mask=attention_mask_2,
                                        token_type_ids=token_type_ids_2)[1]
            intent_logits_2 = self.intent_classifier_2(self.dropout(pooled_output_2))
        
        if input_ids_3 is not None:
            pooled_output_3 = self.bert_model(input_ids=input_ids_3,
                                        attention_mask=attention_mask_3,
                                        token_type_ids=token_type_ids_3)[1]
            intent_logits_3 = self.intent_classifier_3(self.dropout(pooled_output_3))
        # Compute losses if labels provided
        if intent_label_1 is not None:
            loss_fct_1 = CrossEntropyLoss()
            intent_loss_1 = loss_fct_1(intent_logits_1.view(-1, self.num_intent_labels_1), intent_label_1.type(torch.long))
        else:
            intent_loss_1 = torch.tensor(0)
        
        if intent_label_2 is not None:
            loss_fct_2 = CrossEntropyLoss()
            intent_loss_2 = loss_fct_2(intent_logits_2.view(-1, self.num_intent_labels_2), intent_label_2.type(torch.long))
        else:
            intent_loss_2 = torch.tensor(0)
        
        if intent_label_3 is not None:
            loss_fct_3 = CrossEntropyLoss()
            intent_loss_3 = loss_fct_3(intent_logits_3.view(-1, self.num_intent_labels_3), intent_label_3.type(torch.long))
        else:
            intent_loss_3 = torch.tensor(0)
        intent_loss = intent_loss_1 + intent_loss_2 + intent_loss_3
        return intent_logits_1, intent_logits_2, intent_logits_3, intent_loss_1, intent_loss_2, intent_loss_3

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
