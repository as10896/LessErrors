from truecaser.truecase import truecase

from .sentence_differ import sentence_differ
from .insert_cand_gen import get_insert_cand_list
from .gpt2_lm import gpt2_score
from .nli import entailment

def single_sentence_corrector(args, res_dict, sent, threshold=0.98, threshold_insert=0.92, use_truecase=True, addPeriod=False, generate_insertion_candidates=True, use_nli=True, nli_enable_neutral=False):
    if not args.do_evaluation and addPeriod and sent.strip()[-1] not in ".,!?;":
        sent += "."

    # Keep track of all upper case sentences and lower case them for the LM.
    if args.recapitalize:
        # sent = truecase(sent)
        sent = sent[0].upper() + sent[1:]

    upper = False
    if sent.isupper():
        # sent = sent.lower()
        sent = sent.capitalize()
        upper = True

    if args.do_evaluation:
        origin_sent = res_dict["detokenizer"].detokenize(sent.split()) # "I 'm Charlie ." -> "I'm Charlie."
    else:
        origin_sent = sent # "I'm Charlie."

    if args.print_correction_process:
        if args.use_gpt2:
            origin_prob = gpt2_score(sent, res_dict, do_evaluation=args.do_evaluation)
        else:
            origin_proc_sent = processWithSpacy(sent, res_dict["nlp"], do_evaluation=args.do_evaluation)
            origin_proc_tokenized_str = " ".join([tok.text for tok in origin_proc_sent])
            origin_prob = res_dict["lm"].score(origin_proc_tokenized_str, bos=True, eos=True) / len(origin_proc_sent)
        
        print()
        print(sent, round(origin_prob, 4))
        multi_step_results = []
        # multi_step_results.append((sent, round(origin_prob, 2), "--", "--"))
        multi_step_results.append((origin_proc_tokenized_str, round(origin_prob, 2), "--", "--"))

    # Search for and make corrections while has_errors is true
    has_errors = True
    # Iteratively correct errors one at a time until there are no more.
    accum_prob_improve = 0
    prev_sent = origin_proc_tokenized_str
    # prev_sent = origin_sent
    # prev_sent = truecase(origin_sent)
    while has_errors:
        sent, hpy_prob, prob_improve, correct_error_type, has_errors = processSent(args, sent, res_dict, threshold, threshold_insert, origin_sent=origin_sent, generate_insertion_candidates=generate_insertion_candidates, use_nli=use_nli, nli_enable_neutral=nli_enable_neutral)
        if has_errors and args.print_correction_process:
            print(sent, round(hpy_prob, 4))

            # Converting to diff+ format (for frontend usage)
            # e.g. "I am looking forward to [-see-] {+seeing+} you ."
            sent_tmp = processWithSpacy(sent, res_dict["nlp"], do_evaluation=False)
            sent_tmp = " ".join([tok.text for tok in sent_tmp])
            if use_truecase:
                sent_tmp = truecase(sent)
            else:
                sent_tmp = sent[0].upper() + sent[1:]
            if upper: sent_tmp = sent_tmp.upper()
            differ_sent = sentence_differ(prev_sent, sent_tmp)
            multi_step_results.append((differ_sent, round(hpy_prob, 2), round(prob_improve, 2), correct_error_type))
            accum_prob_improve += prob_improve
            prev_sent = sent_tmp

    if args.print_correction_process:
        print()

    # Join all the tokens back together and upper case first char
    if use_truecase:
        sent = truecase(sent)
    else:
        sent = sent[0].upper() + sent[1:]

    # Convert all upper case sents back to all upper case.
    if upper: sent = sent.upper()

    # return sent
    return list(enumerate(multi_step_results)), (len(multi_step_results)-1, sent, round(accum_prob_improve, 2))

# Input 1: The sentence we want to correct; i.e. a list of token strings
# Input 2: A dictionary of useful resources
# Input 3: Command line args
# Output 1: The input sentence or a corrected sentence
# Output 2: Boolean whether the sentence needs more corrections
def processSent(args, sent, res_dict, threshold, threshold_insert, origin_sent=None, generate_insertion_candidates=True, use_nli=True, nli_enable_neutral=False):
    # Process sent with spacy
    proc_sent = processWithSpacy(sent, res_dict["nlp"], do_evaluation=args.do_evaluation)
    proc_tokenized = [tok.text for tok in proc_sent] # 'I have an apple .'
    
    # Calculate avg token prob of the sent so far.
    if args.use_gpt2:
        orig_prob = gpt2_score(sent, res_dict, do_evaluation=args.do_evaluation)
    else:
        orig_prob = res_dict["lm"].score(" ".join(proc_tokenized), bos=True, eos=True) / len(proc_sent)

    # Store all the candidate corrected sentences here
    cand_dict = {}
    # Process each token.
    for tok in proc_sent:
        # SPELLCHECKING
        # Spell check: tok must be alphabetical and not a real word.
        if tok.text.isalpha() and not res_dict["gb"].spell(tok.text):
            cands = res_dict["gb"].suggest(tok.text)
            # Generate correction candidates
            if cands: cand_dict.update(generateCands(tok.i, cands, proc_tokenized, threshold, "SPELL"))
        # MORPHOLOGY
        if tok.lemma_ in res_dict["gb_infl"]:
            cands = res_dict["gb_infl"][tok.lemma_]
            cand_dict.update(generateCands(tok.i, cands, proc_tokenized, threshold, "MORPH"))
        # RV error
        if args.use_rv and tok.lemma_ in res_dict["gb_rv"]:
            cands = res_dict["gb_rv"][tok.lemma_]
            cand_dict.update(generateCands(tok.i, cands, proc_tokenized, threshold, "RV"))
        # DETERMINERS
        if tok.text in res_dict["det"]:
            cand_dict.update(generateCands(tok.i, res_dict["det"], proc_tokenized, threshold, "DET"))
        # PREPOSITIONS
        if tok.text in res_dict["prep"]:
            cand_dict.update(generateCands(tok.i, res_dict["prep"], proc_tokenized, threshold, "PREP"))
        # PUNCTUATIONS
        # if tok.text in [".", ",", "!", "?"]:
            # cands = [".", ",", "!", "?"]
        if tok.text in [".", "!"]:
            cands = ["?"]
            cand_dict.update(generateCands(tok.i, cands, proc_tokenized, threshold, "PUNC"))
        # Temporaily not handling question mark, since it would usually change the question mark into a period (false negative rather than flase positive)
        # if tok.text in ["?"]:
        #     cands = ["."]
        #     cand_dict.update(generateCands(tok.i, cands, proc_tokenized, threshold, "PUNC"))

    # get all the hypotheses predicted by BERT
    # if args.generate_insertion_candidates:
    if generate_insertion_candidates:
        cand_list_insert = get_insert_cand_list(proc_tokenized, res_dict, k=1, only_insert_articles_or_preps=False)

    # Keep track of the best sent if any
    best_prob = float("-inf")
    best_sent = []
    
    # Loop through the candidate edits; edit[-1] is the error type weight
    for edit, cand_sent in cand_dict.items():
        # Score the candidate sentence
        if args.use_gpt2:
            cand_prob = gpt2_score(" ".join(cand_sent), res_dict, do_evaluation=True)
        else:
            cand_prob = res_dict["lm"].score(" ".join(cand_sent), bos=True, eos=True)/len(cand_sent)
        
        # Compare cand_prob against weighted orig_prob and best_prob
        if cand_prob > edit[-1] * orig_prob and cand_prob > best_prob:
            if use_nli and not entailment(predictor=res_dict["nli_model"], hypothesis=res_dict["detokenizer"].detokenize(cand_sent), premise=origin_sent, nli_threshold_type=args.nli_threshold_type, nli_threshold=args.nli_threshold, enable_neutral=nli_enable_neutral):
                continue
            best_prob = cand_prob
            best_sent = cand_sent
            correct_error_type = edit[2]

    # Keep checking out if there's any BERT-predicted result's lm_prob better than the current best_sent
    if generate_insertion_candidates:
        for cand_sent in cand_list_insert:
            if args.use_gpt2:
                cand_prob = gpt2_score(" ".join(cand_sent), res_dict, do_evaluation=True)
            else:
                cand_prob = res_dict["lm"].score(" ".join(cand_sent), bos=True, eos=True)/len(cand_sent)

            if cand_prob > threshold_insert * orig_prob and cand_prob > best_prob:
                if use_nli and not entailment(predictor=res_dict["nli_model"], hypothesis=res_dict["detokenizer"].detokenize(cand_sent), premise=origin_sent, nli_threshold_type=args.nli_threshold_type, nli_threshold=args.nli_threshold, enable_neutral=nli_enable_neutral):
                    continue
                best_prob = cand_prob
                best_sent = cand_sent
                correct_error_type = "INSERT"

    prob_improve = best_prob - orig_prob

    # Return the best sentence and a boolean whether to search for more errors
    # if best_sent: return best_sent, best_prob, True
    # else: return sent, best_prob, False
    if best_sent:
        if args.do_evaluation:
            return " ".join(best_sent), best_prob, prob_improve, correct_error_type, True
        else:
            return res_dict["detokenizer"].detokenize(best_sent), best_prob, prob_improve, correct_error_type, True
    else: 
        return sent, best_prob, 0, None, False

# Input 1: A token index indicating the target of the correction.
# Input 2: A list of candidate corrections for that token.
# Input 3: The current sentence as a list of token strings.
# Input 4: An error type weight
# Output: A dictionary. Key is a tuple: (tok_id, cand, weight),
# value is a list of strings forming a candidate corrected sentence.
def generateCands(tok_id, cands, sent, threshold, error_type):
    contractions = {"n't": ["not"], "'m": ["am"], "'s": ["is", "has"], "'re": ["are", "were"], "'ve": ["have"], "'ll": ["will", "shall"], "'d": ["had", "would", "did"]}
    # Save candidates here.
    edit_dict = {}
    # Loop through the input alternative candidates
    for cand in cands:
        # Copy the input sentence
        new_sent = sent[:]
        # Change the target token with the current cand

        # Avoid correcting contractions such as "She does n't know ." -> "She does not know ."
        if sent[tok_id] in contractions:
            if cand in contractions[sent[tok_id]]:
                continue

        new_sent[tok_id] = cand
        # Remove empty strings from the list (for deletions)
        new_sent = list(filter(None, new_sent))

        if cand != "":
            edit_id = (tok_id, cand, error_type, threshold)
        else:
            edit_id = (tok_id, cand, "DELETE", threshold)

        # Save non-empty sentences
        if new_sent: edit_dict[edit_id] = new_sent

    return edit_dict

# # Input 1: A list of token strings.
# # Input 2: A spacy processing object
# # Output: A spacy processed sentence.
def processWithSpacy(sent, nlp, do_evaluation=False):
    if do_evaluation:
        # sent: "I 'm looking forward to seeing you ."
        proc_sent = nlp.tokenizer.tokens_from_list(sent.split())
        nlp.tagger(proc_sent)
    else:
        # sent: "I'm looking forward to seeing you."
        proc_sent = nlp(sent)
    return proc_sent
