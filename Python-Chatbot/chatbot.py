import nltk
#nltk.download('punkt')

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle
import webbrowser
from googlesearch import search

#with open("intents.json") as file:
with open('my_intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tensorflow.reset_default_graph()
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
  model.load("model.tflearn")
except:
  model.fit(training, output, n_epoch=700, batch_size=10, show_metric=True)
  model.save("model.tflearn")

#Uncomment these lines in case model is not saved and comment the try expect block for model
#model.fit(training, output, n_epoch= 700, batch_size= 10, show_metric=True)
#model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def searchit(inp):
  option= input('Enter 0 to paste links here or 1 to open a tab: ')
  if option == '0':
    print('Ok here are some links: ')
    for i in search(inp, tld='com', lang='en', num=5, stop=5, pause=2.0):
      print(i)
    else:
      webbrowser.open('https://google.com/search?q=' + inp)
  return

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.5:       
          if tag == 'search':
              searchit(inp)
          for tg in data["intents"]:
            #tag = 'greetings'
            if tg['tag'] == tag:
              #print(tag)
              responses = tg['responses']
              print("Rawbot: ", random.choice(responses))
        else: 
          print("Rawbot: Sorry i didnt quite understand what u talking about. \n\tI'm just a simple chatbot :[ \n\tplease try again")

chat()

