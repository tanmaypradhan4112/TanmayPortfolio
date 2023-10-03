---
title: "Mental Health ChatBot : AURA"
description: "Implemented a Chatbot leveraging Deep Learning and Natural Language Processing techniques, utilizing PyTorch and NLTK libraries. The primary objective was to educate teenagers about various Mental Health Disorders in an engaging and informative manner."
pubDate: "Oct 2, 2023"
heroImage: "https://images.pexels.com/photos/17483868/pexels-photo-17483868/free-photo-of-an-artist-s-illustration-of-artificial-intelligence-ai-this-image-represents-how-machine-learning-is-inspired-by-neuroscience-and-the-human-brain-it-was-created-by-novoto-studio-as-par.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
badge: "Latest"
---

## Introduction

***Description:***

Implemented a Chatbot leveraging Deep Learning and Natural Language Processing (NLP) techniques, utilizing PyTorch and NLTK libraries. The primary objective was to educate teenagers about various Mental Health Disorders in an engaging and informative manner.

***Key Contributions:***

* Designed and developed a Chatbot framework from scratch, focusing on providing accurate and accessible information about Mental Health Disorders.
* Utilized PyTorch, a powerful deep learning library, to build the neural network architecture of the Chatbot, enabling it to understand and respond to user queries effectively.
* Integrated the NLTK (Natural Language Toolkit) library to enhance the NLP capabilities of the Chatbot, ensuring it could process and comprehend natural language input.
* Customized the Chatbot's training data to encompass a wide range of Mental Health Disorders, tailoring responses to be informative, empathetic, and age-appropriate for teenagers.
* Implemented a user-friendly interface for seamless interaction, allowing teenagers to ask questions and receive informative responses about Mental Health topics.
* Conducted rigorous testing and validation to ensure the Chatbot provided accurate and reliable information, with a focus on maintaining user trust and confidence.

***Outcome:***

The Chatbot has proven to be an effective educational tool, empowering teenagers with essential knowledge about Mental Health Disorders. Its user-friendly interface and informative responses have made it a valuable resource for teenagers seeking information and support in this critical area.

This project showcases proficiency in Deep Learning, NLP, and utilization of PyTorch and NLTK libraries for developing practical applications addressing mental health awareness among teenagers. Additionally, it highlights a commitment to leveraging technology for the betterment of mental health education and awareness.

## Getting Started

### **Prerequisites for Building the Mental Health Chatbot:**

**Python Programming Knowledge:** A solid understanding of Python is essential as it serves as the primary programming language for this project.

**Familiarity with Deep Learning and NLP:** Basic knowledge of Deep Learning concepts and Natural Language Processing (NLP) techniques is crucial for building the chatbot.

**Understanding of PyTorch and NLTK:** Familiarity with the PyTorch library for implementing neural networks and NLTK (Natural Language Toolkit) for NLP tasks is required.

**Basic Web Development Skills (Optional):** If planning to deploy the chatbot on a web platform, basic web development knowledge may be needed for interface creation.

### Tech Stack Used

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.tertiarycourses.com.sg%2Fmedia%2Fcatalog%2Fproduct%2Fcache%2F1%2Fimage%2F512x%2F040ec09b1e35df139433887a97daa66f%2Fp%2Fy%2Fpytorch.png&f=1&nofb=1&ipt=71066b00971ed30a6a0025d45111ea22e14f0df32a6ecc824cf2de9cd4ed3ab4&ipo=images)

* **Python:** Primary programming language for the project, offering a wide range of libraries and tools for Deep Learning, NLP, and general software development.
* **PyTorch:** Deep Learning library used to build the neural network architecture of the chatbot for processing natural language input and generating responses.
* **NLTK (Natural Language Toolkit):** Library providing a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.
* **Data Collection Tools:** Depending on the sources, tools for web scraping, APIs for data retrieval, or manual data collection methods may be utilized.
* **Development Environment:** IDEs like Jupyter Notebook or Visual Studio Code for writing, testing, and debugging code.

## Let's Code

Open your IDE to get started to build your ChatBot, Here I will be using VS Codium

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse3.mm.bing.net%2Fth%3Fid%3DOIP.ToQKWi3R5A00mTQjlBIp2AHaEf%26pid%3DApi&f=1&ipt=7ceaf067bccc9899f4d8f084525038bf3f0cc67c81b9dbbd3f1b0470dcb1ad09&ipo=images)

Create **intents.json** file to help your Bot understand the intents. We now define conversational intents using json

```
{
  "intents": [
      {
          "tag": "greeting",
          "patterns": [
              "Hi",
              "Hey",
              "How are you",
              "How are you doing",
              "Is anyone there?",
              "What's up?",
              "Hello",
              "Good day",
              "What's popping"
          ],
          "responses": [
              "Hey! Welcome to Aura.",
              "Hi there, I'm doing well! My name's Aura, what can I do for you?",
              "Hey there, my name's Aura, your mental health friend. How can I help you today?"
          ]
      },
      {
          "tag": "goodbye",
          "patterns": ["Bye", "See you later!", "Goodbye!", "Thank you"],
          "responses": [
              "See you later! Take care!",
              "Have a lovely day! Take care and stay safe!",
              "Take care of yourself! I'm always here to support you! Feel free to come back at any time!"
          ]
      },
      {
          "tag": "thanks",
          "patterns": ["Thanks", "Thank you", "That's helpful", "Thank's a lot!"],
          "responses": ["Happy to help!", "Any time!", "My pleasure"]
      },
      {
          "tag": "about",
          "patterns": [
              "What do you do?",
              "Who are you?",
              "What are you here for?",
              "What can you help me with?",
              "Is anyone there?",
              "Are you a real person?",
              "Tell me about yourself!"
          ],
          "responses": [
              "Love that question! My name's Aura, your mental health friend! I've been trained to support you through any of the issues and things that you're going through in life right now. I'm all ears and want to support you through your ups and downs. Technically, I'm a computer that's been trained by a human, but I like to think of myself as human!",
              "Hey! I love that question! My name is Aura, and I want to be your mental health friend and support you! My fellow human friends have trained me to be a compassionate listener and support buddy when things are going well, and particularily when things aren't going so well."
          ]
      },
      {
          "tag": "anxiety",
          "patterns": [
              "I think I have anxiety",
              "What is anxiety?",
              "Tell me about anxiety",
              "Can I have some information about anxiety?",
              "How do I support a loved one with anxiety?",
              "How do I fix my anxiety?"
          ],
          "responses": [
              "I see that you want to learn more about anxiety, and how to support yourself or your loved ones. At the higest level, anxiety is your body's natural response to stress. It's a feeling of fear or apprehension about what's to come. The first day of school, going to a job interview, or giving a speech may cause most people to feel fearful and nervous. We all get anxious sometimes, but anxiety becomes a problem when it starts affecting your daily life drastically. The best thing that you can do to help reduce anxiety is to take a few deep breaths and talk to someone about what you're feeling. Some natural remedies that you can try include: getting enough sleep meditating, staying active and exercising, eating a healthy diet, staying active and working out, avoiding alcohol, avoiding caffeine, and quitting smoking cigarettes. Support your friends and loved ones by checking up on them and just listening. Whatever you're feeling is valid."
          ]
      },
      {
          "tag": "depression",
          "patterns": [
              "I think I have depression",
              "What is depression?",
              "Tell me about depression",
              "Can I have some information about depression?",
              "How do I support a loved one with depression?",
              "How do I fix my depression?"
          ],
          "responses": [
              "I see that you want to learn more about depression, and how to support yourself or your loved ones. At the higest level, depression is a mood disorder that causes a persistent feeling of sadness and loss of interest. Also called major depressive disorder or clinical depression, it affects how you feel, think and behave and can lead to a variety of emotional and physical problems. You may have trouble doing normal day-to-day activities, and sometimes you may feel as if life isn't worth living. All of us experience sadness and periods of time where things aren't that great. However, the difference in sadness and depression lies in the duration and magnitude of the problem. The best recommended course of treatment is to talk a therapist or psychiatrist who can help you find the best path for you. Everyone is different and what might work for one person might not work for another one. Some natural remedies that you can try include: getting enough sleep, meditating, staying active and exercising, eating a healthy diet, staying active and working out, avoiding alcohol, avoiding caffeine, and quitting smoking cigarettes. Support your friends and loved ones by checking up on them and just listening. Whatever you're feeling is valid."
          ]
      },
      {
          "tag": "schizophrenia",
          "patterns": [
              "I think I have schizophrenia",
              "What is schizophrenia?",
              "Tell me about schizophrenia",
              "Can I have some information about schizophrenia?",
              "How do I support a loved one with schizophrenia?",
              "How do I fix my schizophrenia?"
          ],
          "responses": [
              "I see that you want to learn more about schizophrenia, and how to support yourself or your loved ones. At the higest level, Schizophrenia is a serious mental disorder in which people interpret reality abnormally. Schizophrenia may result in some combination of hallucinations, delusions, and extremely disordered thinking and behavior that impairs daily functioning, and can be disabling. People with schizophrenia require lifelong treatment. Early treatment may help get symptoms under control before serious complications develop and may help improve the long-term outlook. If you believe you have symptoms of schizophrenia, and/or you have various risk factors that increase the likelihood of schizophrenia, please consult a family doctor and/or psychiatrist for a diagnosis and treatment plan specific for the individual."
          ]
      },
      {
          "tag": "funny",
          "patterns": [
              "Tell me a joke!",
              "Tell me something funny!",
              "Do you know a joke?"
          ],
          "responses": [
              "Of course! Why did the hipster burn his mouth? He drank the coffee before it was cool.",
              "What did the buffalo say when his son left for college? Bison!",
              "I invented a new word! Plagiarism!"
          ]
      }
  ]
}
```

Every intent contains:

* a tag (a label/name that you choose)
* patterns (sentence patterns for the text classifier in the neural network)
* responses (the answer that you would like the machine to give when done)

After completing the definition of conversational intents we will now focus on NLP processing

```
#nltk_utls.py
import numpy as np
import nltk
#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
  # stem each word
  sentence_words = [stem(word) for word in tokenized_sentence]
  # initialize bag with 0 for each word
  bag = np.zeros(len(words), dtype=np.float32)
  for idx, w in enumerate(words):
    if w in sentence_words:
      bag[idx] = 1
  return bag
```

This code provides essential functions for text preprocessing in NLP tasks, including tokenization (breaking text into words) and stemming (reducing words to their base form). The bag-of-words function is a fundamental step in text classification and information retrieval tasks.

In the next step, we focus on building Deep Learing Model with Pytorch.

```
#models.py
import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
```

this code defines a feedforward neural network architecture with three hidden layers. The activation function used is ReLU, and it's designed to take an input of size `input_size`, pass it through the hidden layers, and produce an output of size `num_classes`.

The purpose of this code down below is to load the intents.json file, apply the natural language processing code, create the training data, and begin training the model.

```
#train.py
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
with open('intents.json', 'r') as f:
    intents = json.load(f)
all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))
# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)
X_train = np.array(X_train)
y_train = np.array(y_train)
# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')
```

The next step is to build a framework for the chatbot.

```
#model.py
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
bot_name = "Aura"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."
```

After building the framework for the responses of chatbot, we willnow create a GUI for the Mental Health Chatbot similar to ChatGPT.

```
#chat.py
from tkinter import *
from chat import get_response, bot_name
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOUR = "#EAECEE"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
    
    def run(self):
        self.window.mainloop()
    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=600, height=750, bg=BG_COLOR)
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOUR,
                            text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        #tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        #text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOUR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        #bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        #message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOUR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        #send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command= lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\\n\\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        msg2 = f"{bot_name}: {get_response(msg)}\\n\\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
```

All right, we are now set to use the Mental Health Chatbot. You can also change the configuration like widht, height and other styling aspect you need. Focus on the intents.json as the responses of the chatbot is based on it.

**HAPPY CODING**......

