import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerModel, LongformerLMHead, LongformerMaskedLMOutput

class LongformerForMaskedLM(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.label_yes_id = config.label_yes
        self.label_no_id = config.label_no
        self.mask_token_id = config.mask_token_id

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> import torch
            >>> from transformers import LongformerForMaskedLM, LongformerTokenizer

            >>> model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096')
            >>> tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

            >>> SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
            >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

            >>> attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
            ...                        # check ``LongformerModel.forward`` for more details how to set `attention_mask`
            >>> outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            >>> loss = outputs.loss
            >>> prediction_logits = output.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # sequence_output = outputs[0]
        sequence_output = outputs[0][input_ids == self.mask_token_id]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        prediction_scores = prediction_scores[:,self.label_yes_id] - prediction_scores[:,self.label_no_id]
        prediction_scores = prediction_scores.view(input_ids.shape[0],-1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return LongformerMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

class CodeAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_size
        self.embed_dim = config.hidden_size
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.dense_ff = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm_ff = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.dropout_ff = nn.Dropout(config.hidden_dropout_prob)

        self.layer_id = layer_id


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads,  self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, code_embeddings, doc_embeddings):
        mixed_query_layer = self.query(code_embeddings)
        key = self.transpose_for_scores(self.key(doc_embeddings))
        value = self.transpose_for_scores(self.value(doc_embeddings))
        query = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        code_embeddings = self.dense_ff(code_embeddings)
        code_embeddings = self.dropout_ff(code_embeddings)
        code_embeddings = self.LayerNorm_ff(code_embeddings + context_layer)
        return code_embeddings
class ConvAttnPrompt(nn.Module):

    def __init__(self, Y, dicts, args=None):
        super(ConvAttnPrompt, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = LongformerConfig.from_pretrained("whaleloops/KEPTlongformer-PMM3")
        config.num_hidden_layers = args.num_tran_layers
        config.attention_window = [32 for x in range(config.num_hidden_layers)]
        self.tran_model = LongformerModel.from_pretrained("whaleloops/KEPTlongformer-PMM3", config=config)

        self.code_reasoning_layers = nn.ModuleList()
        for layer_id in range(args.reasoning_layer_num):
            self.code_reasoning_layers.append(CodeAttention(args, layer_id))
        self.code_prob_layer = nn.Linear(args.hidden_size, 1)
        self.code_descriptions = []
        for code_id in dicts['c2ind']:
            self.code_descriptions.append(dicts['desc'][code_id])

    def _code_embedding_update(self):
        with torch.no_grad():
            update_batch_size = 50
            rounds = math.ceil(len(self.code_descriptions)/update_batch_size)
            new_code_embeddings = []
            for i in range(rounds):
                c_descriptions = self.code_descriptions[i*update_batch_size:(i+1)*update_batch_size]
                batch_data = self.tokenizer(c_descriptions, padding=True, return_tensors='pt', truncation=True, max_length=4000)
                batch_ids = batch_data['input_ids'].to(self.device)
                batch_mask = batch_data['attention_mask'].to(self.device)
                global_mask = torch.ones_like(batch_mask)
                outs = self.tran_model(input_ids=batch_ids, attention_mask=batch_mask, global_attention_mask=global_mask).pooler_output
                new_code_embeddings.append(outs)
            new_code_embeddings = torch.cat(new_code_embeddings, dim=0).detach()
            self.code_embeddings = new_code_embeddings

    def _code_embedding_propogate(self, code4training_index, code4training_index_total):
        c2p = {}
        for position, codeid in enumerate(code4training_index_total):
            c2p[codeid] = position
        c_descriptions = np.array(self.code_descriptions)[code4training_index_total].tolist()
        batch_data = self.tokenizer(c_descriptions, padding=True, return_tensors='pt', truncation=True, max_length=4000)
        batch_ids = batch_data['input_ids'].to(self.device)
        batch_mask = batch_data['attention_mask'].to(self.device)
        global_mask = torch.ones_like(batch_mask)
        outs = self.tran_model(input_ids=batch_ids, attention_mask=batch_mask, global_attention_mask=global_mask).pooler_output
        outs_matrix = []
        #outs [Nt, E]
        for i in range(len(code4training_index)):
            temp = []
            for v in code4training_index[i]:
                temp.append(outs[c2p[v[1]]])
            temp = torch.stack(temp, 0)
            outs_matrix.append(temp)
        outs_matrix = torch.stack(outs_matrix, 0)

        return outs_matrix


    def _prompt_inquiry(self, code_embeddings, doc_embeddings):
        if len(code_embeddings.shape) < 3:
            code_embeddings = code_embeddings.unsqueeze(0)
        for code_reasoning_layer in self.code_reasoning_layers:
            code_embeddings = code_reasoning_layer(code_embeddings, doc_embeddings)
        return code_embeddings

    def _get_context_loss(self, yhat, target):
        # calculate the BCE
        loss = F.binary_cross_entropy_with_logits(yhat, target)
        return loss

    def _select_traget_codes(self, target, yhat):
        # select the codes that need to be trained
        target = target.detach().cpu()
        yhat = yhat.detach().cpu()
        index_tp = target > 0
        topk = max(torch.sum(index_tp, dim=1).max().item()*2, 50)
        p_log, index_p = torch.topk(yhat, k=topk, dim=1)
        index_t = index_tp.clone()
        for i in range(len(index_t)):
            k = 0
            while torch.sum(index_t[i]).item() < topk:
                index_t[i][index_p[i][k]] = True
                k += 1
        code4training_index = torch.nonzero(index_t).view(index_t.shape[0], -1, 2)
        code4training_index_total = torch.nonzero(index_t.sum(dim=0)).squeeze(-1)
        code4training_target = torch.zeros(code4training_index.shape[0:2], dtype=torch.float32)
        for bid, bindexs in enumerate(code4training_index):
            for lid, lindexs in enumerate(bindexs):
                if index_tp[lindexs[0], lindexs[1]] == True:
                    code4training_target[bid, lid] = 1
                else:
                    code4training_target[bid, lid] = 0
        return code4training_index.numpy(), code4training_index_total.numpy(), code4training_target

    def forward(self,
        input_ids=None,
        attention_mask=None,):
        # update code embeddings
        self._code_embedding_update()
        # get embeddings and apply dropout
        outs = self.tran_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        contextualized_code_embeddings = self._prompt_inquiry(self.code_embeddings, outs)
        #yhat_full = self.final.weight.mul(contextualized_code_embeddings).sum(dim=2).add(self.final.bias)
        yhat_full = self.code_prob_layer(contextualized_code_embeddings).squeeze(-1)
        loss_context = self._get_context_loss(yhat_full, target)
        code4training_index, code4training_index_total, target_training = self._select_traget_codes(target, yhat_full)
        code4training_embeddings = self._code_embedding_propogate(code4training_index, code4training_index_total)
        contextualized_code_embeddings = self._prompt_inquiry(code4training_embeddings, outs.detach())
        #yhat_train = self.final.weight.mul(contextualized_code_embeddings).sum(dim=2).add(self.final.bias)
        yhat_train = self.code_prob_layer(contextualized_code_embeddings).squeeze(-1)
        loss_coding = self._get_context_loss(yhat_train, target_training.to(yhat_train.device))
        loss = loss_coding+loss_context
        yhat_full = torch.sigmoid(yhat_full)
        return yhat_full, loss, loss_context
