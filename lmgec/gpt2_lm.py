import torch

def gpt2_score(sent, res_dict, do_evaluation=False):
    sent = sent[0].upper() + sent[1:]
    if do_evaluation:
        sent = res_dict["detokenizer"].detokenize(sent.split()) # "I 'm Charlie ." -> "I'm Charlie."
    # sent here shouldn't be tokenized and lowercased, or it will result in different tokenization result
    tokenized_ids = res_dict["gpt2_tokenizer"].encode(sent, add_special_tokens=True)
    input_ids = torch.tensor(tokenized_ids, device=res_dict["device"]).unsqueeze(0)  # Batch size 1
    with torch.no_grad():
        outputs = res_dict["gpt2_model"](input_ids, labels=input_ids)
    loss = outputs[0]
    return - loss.item()