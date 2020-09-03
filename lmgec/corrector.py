import os
from .lmgec import single_sentence_corrector
from .loadResources import loadResources

class AttrDict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

args = AttrDict()
basename = os.path.dirname(os.path.realpath(__file__))

args.pretrain_bert_model = "roberta-large"

args.use_rv = False # Replace-Verb Errors

args.tokenizer_do_lower_case = False

# args.cuda_device = -1
args.cuda_device = 0

args.use_nli = True
args.nli_threshold_type = "label"
# args.nli_threshold_type = "prob"
args.nli_threshold = 0.85 # used only when `args.use_nli` is "prob"
args.pretrain_nli_model = "snli"
# args.pretrain_nli_model = "mnli"

args.use_gpt2 = False

args.recapitalize = True

args.print_correction_process = True

args.do_evaluation = False

args.generate_insertion_candidates = True

res_dict = loadResources(args)

def corrector(sent, threshold=0.98, threshold_insert=0.92, sent_segmentation=False, use_truecase=True, addPeriod=False, generate_insertion_candidates=True, use_nli=True, nli_enable_neutral=False):
    sent = sent.replace("’", "'") # I don’t know. -> I don't know.
    
    # Currently we only support single sentence correction
    # TO-DO: Iterative correction after sentence segmentation
    correct_process, correct_result = single_sentence_corrector(args, res_dict, sent, threshold, threshold_insert, use_truecase, addPeriod, generate_insertion_candidates, use_nli, nli_enable_neutral)

    return correct_process, correct_result

"""Example input/output
    - sent: 'I am looking forway see you.'
    - correct_process: [(0, ('I am looking forway see you .', -3.45, '--', '--')), 
                         (1, ('I am looking [-forway-] {+forward+} see you .', -2.44, 1.01, 'SPELL')), 
                         (2, ('I am looking forward {+to+} see you .', -1.55, 0.89, 'INSERT')), 
                         (3, ('I am looking forward to [-see-] {+seeing+} you .', -1.21, 0.34, 'MORPH'))]
    - correct_result: (3, 'I am looking forward to seeing you .', 2.24)}
"""