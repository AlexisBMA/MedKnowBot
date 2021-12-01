# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import time 
from typing import Any, Text, Dict, List
from transformers import pipeline
classifier = pipeline("zero-shot-classification")

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionHelloWorld(Action):
    
    def obtain_top_labels(userI, limit, labels):
        ans = classifier(userI, labels)
        ans_top = {}
        for i in range(limit):
            ans_top[ans['labels'][i]] = ans['scores'][i]
        print(ans_top)

    def name(self) -> Text:
        return "action_return_prob"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text=f"These are your chief complaints related to your symptoms")
        chiefSigns = open("chiefSigns.txt", "r")
        candlabels = chiefSigns.readlines()
        userInput = tracker.latest_message['text']
        print(userInput)
        sLabels = classifier(userInput, candlabels)
        prob = {}
        for i in range(10):
            res = sLabels['labels'][i] + ": " + str(sLabels['scores'][i])
            dispatcher.utter_message(text=f"{res}")

        return []



