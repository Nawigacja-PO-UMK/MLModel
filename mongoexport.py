import pymongo
import json

client = pymongo.MongoClient('mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000')
db = client['skany']
collection = db['skany_pozycji']
document = list(collection.find())

with open('baza_pozycji.json', 'w') as file:
    json.dump(document, file)
