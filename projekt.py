import spacy
import torch
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from flask_cors import CORS

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

app = Flask(__name__)
CORS(app)

nlp = spacy.load('pl_core_news_lg')

device = "cuda" if torch.cuda.is_available() else "cpu"

sbert = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=device
)

TRESHOLD = 0.8
model = torch.load(r"C:\Users\Admin\Desktop\studia\nlp\projekt\pjn_model.pth", weights_only=False, map_location=device)
model.eval()

def tokenize(sentence):
    with torch.no_grad():
        text = sbert.encode(sentence, convert_to_tensor=True)
    return text

def getCategory(token):
    with torch.no_grad():
        logits = model(token)
        prob = torch.softmax(logits, dim=0)
        max_prob, pred = torch.max(prob, dim=0)
    if max_prob < TRESHOLD:
        return "Brak zdarzenia"
    elif pred == 0:
        return "biznes"
    elif pred == 1:
        return "katastrofa"
    else:
        return "przestępstwo"
    
def extract_subject_predicate_object(sentence):
    doc = nlp(sentence)
    subject = None
    predicate = None
    object_ = None

    for token in doc:
        if "subj" in token.dep_:
            subject = token.text
        if "VERB" in token.pos_:
            predicate = token.text
        if "obj" in token.dep_:
            object_ = token.text

    return subject, predicate, object_


@app.route("/getResult", methods=["POST"])
def analyze():
    data = request.get_json()
    sentence = data.get('sentence')
    kategoria = getCategory(tokenize(sentence))
    subject, predicate, obj = extract_subject_predicate_object(sentence)
    return jsonify({"kategoria":kategoria, "kto":subject, "trigger":predicate, "co":obj})



if __name__ == "__main__":
    app.run(debug=True)
