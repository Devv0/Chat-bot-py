import pandas
import aiml
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import PIL.Image
import numpy as np
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring
#opening xml file
kernel = aiml.Kernel()
kernel.bootstrap(learnFiles="std-startup.xml")
kernel.respond("load aiml b")
#declaring arrays
questions = []
answers = []
#open cnn
model = keras.models.load_model('./')
#importing knowledge base
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
#checks integeraty of knowledge base
dummy = read_expr('dummy(dummy)')
result = ResolutionProver().prove(dummy, kb, verbose = False)
if (result):
    print("Error")
    quit()
#keys for azure
cog_key = 'ccec02e6f50c4ad6b205b1ba3682e8a3'
cog_endpoint = 'https://n0801907.cognitiveservices.azure.com/'
cog_region = 'eastus'

# Create a function that makes a REST request to the Text Translation service
def translate_text(cog_region, cog_key, text, to_lang='fr', from_lang='en'):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]


#while true loop keeps bot running of reply to user message
while True:
    text_to_translate = input("Enter text")
    #libraries 
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.textanalytics import TextAnalyticsClient
    #connect to azure
    credential = AzureKeyCredential("ccec02e6f50c4ad6b205b1ba3682e8a3")
    endpoint="https://n0801907.cognitiveservices.azure.com/"
    text_analytics_client = TextAnalyticsClient(endpoint, credential)
    #array to user input
    documents = []
    documents.append(text_to_translate)
    #detects language of user input
    response = text_analytics_client.detect_language(documents)
    result = [doc for doc in response if not doc.is_error]
    #translate users input to english
    for doc in result:
        translation = translate_text(cog_region, cog_key, text_to_translate, to_lang='en', from_lang=doc.primary_language.iso6391_name)
    user_input = translation
    #removes user input from array
    documents.pop()
    
    #list of image names
    images = ["image01.jpg", "image02.jpg", "image03.jpg", "image04.jpg", "image05.jpg"]
    #loop to detect if user wants to check image
    image = user_input.lower().split()
    for word in image:
        if word in images:
            #classes of images in cnn
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
                        
            image = word

            import cv2 as cv
            #finding and resizing image
            img = cv.imread(image)
            img = cv.resize(img, (32,32))
            # Create a batch
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) 
            #using cnn model to predict image
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            #output prediction
            print(
                "This image most likely belongs to {}"
                .format(class_names[np.argmax(score)])
            )

    #gets response from aiml
    
    response = kernel.respond(user_input)
    cmd = 0
    
    if response[0] == '#':
        params = response[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
    #aiml response
    if cmd == 0:
        #gives translated response
        translation = translate_text(cog_region, cog_key, response, to_lang=doc.primary_language.iso6391_name, from_lang='en')
        print(translation)

    #closes program
    elif cmd == 1:
        print("Thank you for using MTG Bot")
        time.sleep(1)
        break

    elif cmd == 31:
        #lets user add to kb
        object,subject=params[1].split(' is ')
        expr=read_expr(subject + '(' + object + ')')
        #check for contadiction 
        kb.append(expr) 
        dummy = read_expr('dummy(dummy)')
        result = ResolutionProver().prove(dummy, kb, verbose = False)
        if result:
            kb.pop()
            print("Not possible")
        else:
            print('OK, I will remember that',object,'is', subject)
        
    elif cmd == 32: #checks if user input is in kb
        object,subject=params[1].split(' is ')
        expr=read_expr(subject + '(' + object + ')')
        answer=ResolutionProver().prove(expr, kb, verbose = False)
        if answer:
           print('Correct.')
        #checks is user input is possible
        else:
            print('It may not be true. Let me check')
            kb.append(expr) 
            dummy = read_expr('dummy(dummy)')
            result = ResolutionProver().prove(dummy, kb, verbose = False)
            if result:
                kb.pop()
                print("Not possible")
            else:
                kb.pop()
                print('This could be true')

    else:
        #puts content of csv into array
        with open ('mtg.csv', newline='') as csvfile:
            data = list(csv.reader(csvfile))
            for row in data:
                questions.append(row[0])
                answers.append(row[1])
        #adds user input to arrat
        questions.append(user_input)
        #convers array to tf/idf
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(questions)
        #store highest cosine value
        answer = 0
        #loops through array to find most accurate response for question
        for i in range(len(questions) - 1):
            cosine = cosine_similarity(x[i], x[-1])
            if cosine > answer:
                answer = cosine
                location = i

        if answer > 0.01:        
            #gives translated response
            translation = translate_text(cog_region, cog_key, answers[location], to_lang=doc.primary_language.iso6391_name, from_lang='en')
            print(translation)
        else:
            print ("I couldnt find anything for that")
        #removes user input from array
        questions.pop()