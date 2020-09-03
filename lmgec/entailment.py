from allennlp.predictors.predictor import Predictor

def entailment(predictor: Predictor, hypothesis: str, premise: str):
    return predictor.predict(hypothesis=hypothesis, premise=premise)["label"] == "entailment"

    # premise = "Two women are wandering along the shore drinking iced tea."
    # hypothesis = "Two women are sitting on a blanket near some rocks talking about politics."
    # result = predict(hypothesis=hypothesis, premise=premise)
    # {'label': 'contradiction',
    #  'logits': [-4.040728569030762, 3.5561299324035645, 1.1318645477294922],
    #  'probs': [0.0004609781608451158, 0.9182355403900146, 0.08130345493555069]}