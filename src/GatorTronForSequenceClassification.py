import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class GatorTronForSequenceClassification(PreTrainedModel):
    def __init__(self, pretrained_model_name="UFNLP/gatortron-base", num_labels=2):
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.num_labels = num_labels
        super().__init__(config)

        self.gatortron = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        if self.num_labels == 1:
            self.loss_fct = BCEWithLogitsLoss()
        else:
            self.loss_fct = CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs
    ):
        outputs = self.gatortron(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return HF-compatible ModelOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )