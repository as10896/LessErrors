from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from .corrector import corrector
import json

@csrf_exempt
def correct(request):
    if request.method == "POST":
        if request.body:
            json_data = json.loads(request.body)
            sent = json_data["sent"]
            threshold = json_data["threshold"]
            threshold_insert = json_data["threshold_insert"]
            sent_segmentation = json_data.get("sent_segmentation", False)
            use_truecase = json_data.get("use_truecase", True)
            addPeriod = json_data.get("addPeriod", False)
            generate_insertion_candidates = json_data.get("generate_insertion_candidates", True)
            use_nli = json_data.get("use_nli", True)
            nli_enable_neutral = json_data.get("nli_enable_neutral", False)

            _, correct_result = corrector(sent=sent, threshold=threshold, threshold_insert=threshold_insert, sent_segmentation=sent_segmentation, use_truecase=use_truecase, addPeriod=addPeriod, generate_insertion_candidates=generate_insertion_candidates, use_nli=use_nli, nli_enable_neutral=nli_enable_neutral)
            
            print("json_data:", json_data)
            print("correct_result:", correct_result)
            response = correct_result[1]

            print("response:", response)

            return HttpResponse(response)

        else:
            return HttpResponse("Please add something to your request body.")
    else:
        return HttpResponse("Please use POST.")


@csrf_exempt
def correct_frontend(request):
    if request.method == "POST":
        if request.body:
            json_data = json.loads(request.body)
            sent = json_data["sent"]
            threshold = json_data["threshold"]
            threshold_insert = json_data["threshold_insert"]
            sent_segmentation = json_data.get("sent_segmentation", False)
            use_truecase = json_data.get("use_truecase", True)
            addPeriod = json_data.get("addPeriod", True)
            generate_insertion_candidates = json_data.get("generate_insertion_candidates", True)
            use_nli = json_data.get("use_nli", True)
            nli_enable_neutral = json_data.get("nli_enable_neutral", False)
            
            correct_process, correct_result = corrector(sent=sent, threshold=threshold, threshold_insert=threshold_insert, sent_segmentation=sent_segmentation, use_truecase=use_truecase, addPeriod=addPeriod, generate_insertion_candidates=generate_insertion_candidates, use_nli=use_nli, nli_enable_neutral=nli_enable_neutral)
            print("json_data:", json_data)
            print("correct_result:", correct_result)

            response = {"proc": correct_process, "result": correct_result}
            print("response:", response)

            return JsonResponse(response, safe=False)

        else:
            return HttpResponse("Please add something to your request body.")
    else:
        return HttpResponse("Please use POST.")