import re
import torch

def insertion_predict(sent, model, tokenizer, device, k=1):
    # sent: # 'i [MASK] looking forward [MASK] seeing you'
    tokenized_ids = tokenizer.encode(sent, add_special_tokens=True)
    masked_indices =  [i for i, idx in enumerate(tokenized_ids) if idx == tokenizer.mask_token_id]
    input_ids = torch.tensor(tokenized_ids, device=device).unsqueeze(0)
    # segments_tensors = torch.zeros_like(input_ids, device=device)
    with torch.no_grad():
        prediction_scores = model(input_ids)[0] # batch_size=1 x tokenized_sequence_length x vocab_size
    predicted_indices = prediction_scores[0, masked_indices, :].topk(k)[1].tolist()[0] # greedy if k = 1
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

    # In RoBERTa, predicted masked token will be added a special character "Ġ" in the beginning of the word, 
    # e.g. "i 'm looking forward seeing <mask> you" -> "i 'm looking forward seeing Ġfrom you""
    # Removing this character to keep it human readable
    predicted_tokens = [re.sub("Ġ", "", token) for token in predicted_tokens]
    
    return predicted_tokens

def get_insert_cand_list(text, res_dict, k=1, only_insert_articles_or_preps=False, lower_if_insert_at_beginning=False):
    # text:  ['I', "'m", 'looking', 'forward', 'seeing', 'you']
    cand_list_insert = []
    for i in range(len(text)+1):
        masked_text = text[:]
        
        if lower_if_insert_at_beginning:
            if i == 0:
                masked_text[0]= masked_text[0].lower()

        masked_text.insert(i, res_dict["bert_tokenizer"].mask_token)

        predicted_tokens = insertion_predict(" ".join(masked_text), res_dict["bert_model"], res_dict["bert_tokenizer"], res_dict["device"], k)
        for predicted_token in predicted_tokens:
            if only_insert_articles_or_preps and predicted_token not in res_dict["det"] and predicted_token not in res_dict["prep"]:
                # this could avoid inserting tokens such as "not", "very" that may change the semantic meaning
                # however, it may also cause system not able to correct missing words such as "and", "or", that are not in the pre-loaded dictionary
                # Maybe add common missing words from the test set could improve the performance
                continue

            masked_text[i] = predicted_token
            pred_out_text = " ".join(masked_text)
            # pred_out_text = re.sub("Ġ", "", pred_out_text).capitalize().split()
            pred_out_text = pred_out_text.capitalize().split()
            cand_list_insert.append(pred_out_text)

    return cand_list_insert