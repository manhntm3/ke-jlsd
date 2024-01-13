from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect

from wiki.keyword_extraction.keyword_extraction import KeywordExtractor

def index(request):
    return HttpResponse("Hello, world. You're at the wiki index.")

extractor = KeywordExtractor()

def get_keywords(document):

    keywords = extractor.inference(document)
    return keywords

@csrf_exempt 
def get_wiki_keywords(request):
    if request.method == 'POST':
        document = request.body.decode("utf-8")
        document = json.loads(document)['data']
        print('doc:', document)

        with open("/colab/jlsd/sample.txt", "w") as f:
            f.write(document)

        data = {
            'keywords': get_keywords(document),
        }
        print('json-data to be sent: ', data)
        # Process your data here
        return JsonResponse(data)
    else:
        # This is not a POST request
        # Render your page here
        return render(request, JsonResponse())

