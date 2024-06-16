import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class AuthorClassifier(nn.Module):
    def __init__(self, mode, output_size, hidden_size, vocab_size, embedding_length):
        super(AuthorClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # [CLS] token
        output = self.fc(cls_output)
        return output

model = AuthorClassifier(mode='lstm', output_size=50, hidden_size=300, vocab_size=30522, embedding_length=100)
model.load_state_dict(torch.load('author_classifier_model.pth', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_and_predict(text):
    inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

st.title("Author Classifier")
st.write("Enter some text and the model will predict the author.")

user_input = st.text_area("Text input")
if st.button("Predict"):
    prediction = preprocess_and_predict(user_input)
    st.write(f"The predicted author class is: {prediction}")

# Start ngrok and Streamlit
from pyngrok import ngrok
public_url = ngrok.connect(port='8501')
print(f'Public URL: {public_url}')
