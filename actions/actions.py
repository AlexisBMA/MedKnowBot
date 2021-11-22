# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
import datetime as dt
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

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        chiefSigns = open("chiefSigns.txt", "r")
        candlabels = chiefSigns.readlines()
        ans = classifier(tracker.latest_message['text'], candlabels)
        prob = {}
        for i in range(10):
            prob[ans['labels'][i]] = ans['scores'][i]
        dispatcher.utter_message(text=f"{prob}")

        return []



