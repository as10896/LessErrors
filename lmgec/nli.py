def entailment(predictor, hypothesis, premise, nli_threshold_type="label", nli_threshold=0.85, enable_neutral=False):
    if nli_threshold_type == "label":
        if enable_neutral:
            return predictor.predict(hypothesis=hypothesis, premise=premise)["label"] != "contradiction"
        else:
            return predictor.predict(hypothesis=hypothesis, premise=premise)["label"] == "entailment"
    elif nli_threshold_type == "prob":
        return predictor.predict(hypothesis=hypothesis, premise=premise)["probs"][0] >= nli_threshold

    # premise = "Two women are wandering along the shore drinking iced tea."
    # hypothesis = "Two women are sitting on a blanket near some rocks talking about politics."
    # result = predict(hypothesis=hypothesis, premise=premise)
    # {'label': 'contradiction',
    #  'logits': [-4.040728569030762, 3.5561299324035645, 1.1318645477294922],
    #  'probs': [0.0004609781608451158, 0.9182355403900146, 0.08130345493555069]}