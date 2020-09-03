import kenlm
import os
import re
import spacy
from hunspell import Hunspell

from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

from allennlp.predictors.predictor import Predictor
import allennlp_models.nli


# Input: Command line args.
# Output: A dictionary of useful resources.
def loadResources(args):
    # Get base working directory.
    basename = os.path.dirname(os.path.realpath(__file__))

    print(vars(args))

    # Load spaCy
    print("Load spaCy.....", end="", flush=True)
    nlp = spacy.load("en")
    print("Done.")
    print("Load TreebankWordDetokenizer.....", end="", flush=True)
    d = TreebankWordDetokenizer()
    print("Done.")
    # Hunspell spellchecker: https://pypi.python.org/pypi/CyHunspell
    # CyHunspell seems to be more accurate than Aspell in PyEnchant, but a bit slower.
    print("Load Hunspell.....", end="", flush=True)
    gb = Hunspell("en_GB-large", hunspell_data_dir=basename+'/resources/spelling/')
    print("Done.")
    # Inflection forms: http://wordlist.aspell.net/other
    print("Load AGID.....", end="", flush=True)
    gb_infl = loadWordFormDict(basename+"/resources/agid-2016.01.19/infl.txt")
    print("Done.")
    # Synonym sets
    print("Load RV.....", end="", flush=True)
    gb_rv = loadSynonymSet(os.path.join(basename, "resources/synonym_set.txt"))
    print("Done.")
    # List of common determiners
    det = {"", "the", "a", "an"}
    # List of common prepositions
    prep = {"", "about", "at", "by", "for", "from", "in", "of", "on", "to", "with"}

    if args.generate_insertion_candidates:
        print("Load BERT.....", end="", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_bert_model, do_lower_case=args.tokenizer_do_lower_case)
        model = AutoModelWithLMHead.from_pretrained(args.pretrain_bert_model)
        model.eval()
        print("Done.")

    if args.use_gpt2:
        print("Load GPT-2.....", end="", flush=True)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt2_model.eval()
        print("Done.")
    else:
        # Language model built by KenLM: https://github.com/kpu/kenlm
        print("Load KenLM.....", end="", flush=True)
        lm = kenlm.Model(os.path.join(basename, "resources/1b.bin"))
        print("Done.")

    if args.cuda_device > -1:
        torch.cuda.set_device(args.cuda_device)
        device = torch.device("cuda")
        if args.generate_insertion_candidates:
            model.cuda()
            # model.to(device)
        if args.use_gpt2:
            gpt2_model.cuda()
            # gpt2_model.to(device)
    else:
        device = torch.device("cpu")

    if args.use_nli:
        print("Load AllenNLP NLI model...", end="", flush=True)
        if args.pretrain_nli_model == "snli":
            predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli-roberta-large-2020.02.27.tar.gz", predictor_name="textual-entailment", cuda_device=args.cuda_device)
        elif args.pretrain_nli_model == "mnli":
            predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/mnli-roberta-large-2020.02.27.tar.gz", predictor_name="textual-entailment", cuda_device=args.cuda_device)
        print("Done.")

    # Save the above in a dictionary:
    res_dict = {"nlp": nlp,
                "gb": gb,
                "gb_infl": gb_infl,
                "gb_rv": gb_rv,
                "det": det,
                "detokenizer": d,
                "prep": prep,
                "device": device}

    if args.generate_insertion_candidates:
        res_dict["bert_tokenizer"] = tokenizer
        res_dict["bert_model"] = model

    if args.use_nli:
        res_dict["nli_model"] = predictor

    if args.use_gpt2:
        res_dict["gpt2_tokenizer"] = gpt2_tokenizer
        res_dict["gpt2_model"] = gpt2_model
    else:
        res_dict["lm"] = lm

    return res_dict

# Input: Path to Automatically Generated Inflection Database (AGID)
# Output: A dictionary; key is lemma, value is a set of word forms for that lemma
def loadWordFormDict(path):
    entries = open(path).read().strip().split("\n")
    form_dict = {}
    for entry in entries:
        entry = entry.split(": ")
        key = entry[0].split()
        forms = entry[1]
        # The word lemmax
        word = key[0]
        # Ignore some of the extra markup in the forms
        forms = re.sub("[012~<,_!\?\.\|]+", "", forms)
        forms = re.sub("\{.*?\}", "", forms).split()
        # Save the lemma and unique forms in the form dict
        form_dict[word] = set([word]+forms)
    return form_dict

def loadSynonymSet(path):
    synonym_set = {}
    with open(path) as f:
        for x in f:
            word, synonyms = x.strip().split("\t")
            synonyms = synonyms.split()
            synonym_set[word] = synonyms
    return synonym_set