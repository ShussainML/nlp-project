import streamlit as st
import torch
from transformers import BartForSequenceClassification, BartTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Dictionary mapping class index to author names
dictOfAuthors = {
    0: 'AaronPressman', 1: 'AlanCrosby', 2: 'AlexanderSmith', 3: 'BenjaminKangLim', 4: 'BernardHickey',
    5: 'BradDorfman', 6: 'DarrenSchuettler', 7: 'DavidLawder', 8: 'EdnaFernandes', 9: 'EricAuchard',
    10: 'FumikoFujisaki', 11: 'GrahamEarnshaw', 12: 'HeatherScoffield', 13: 'JanLopatka', 14: 'JaneMacartney',
    15: 'JimGilchrist', 16: 'JoWinterbottom', 17: 'JoeOrtiz', 18: 'JohnMastrini', 19: 'JonathanBirt',
    20: 'KarlPenhaul', 21: 'KeithWeir', 22: 'KevinDrawbaugh', 23: 'KevinMorrison', 24: 'KirstinRidley',
    25: 'KouroshKarimkhany', 26: 'LydiaZajc', 27: 'LynneODonnell', 28: 'LynnleyBrowning', 29: 'MarcelMichelson',
    30: 'MarkBendeich', 31: 'MartinWolk', 32: 'MatthewBunce', 33: 'MichaelConnor', 34: 'MureDickie',
    35: 'NickLouth', 36: 'PatriciaCommins', 37: 'PeterHumphrey', 38: 'PierreTran', 39: 'RobinSidel',
    40: 'RogerFillion', 41: 'SamuelPerry', 42: 'SarahDavison', 43: 'ScottHillis', 44: 'SimonCowell',
    45: 'TanEeLyn', 46: 'TheresePoletti', 47: 'TimFarrand', 48: 'ToddNissen', 49: 'WilliamKazer'
}

# Function to load model and tokenizer from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "facebook/bart-large-cnn"  # Example model name, replace with your correct model name
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Load model
    model = BartForSequenceClassification.from_pretrained('sajid227/nlp-project-author-identifcation')
    
    return tokenizer, model

# Function to predict author name
def predict_author_name(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()
    predicted_author = dictOfAuthors.get(predicted_class, "Unknown")  # Get author name from dictionary
    return predicted_author

# Function to evaluate model performance using actual data
def evaluate_model_performance(model, tokenizer, test_data):
    y_true = []
    y_pred = []
    
    for text, true_label in test_data:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        y_true.append(true_label)
        y_pred.append(predicted_class)
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=list(dictOfAuthors.values()), output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)
    
    return report, confusion

# Streamlit app
def main():
    st.title("Author Identification")
    tokenizer, model = load_model_and_tokenizer()
    st.write("Model and tokenizer loaded successfully!")
    
    text_input = st.text_input("Enter text to classify:", "")
    
    if st.button("Predict"):
        if text_input:
            st.write("Predicting author...")
            predicted_author = predict_author_name(model, tokenizer, text_input)
            st.write(f"Predicted author: {predicted_author}")
    
    # Example test data: replace with actual test dataset
    test_data = [
        (""Britain's Ladbroke Group Plc Monday concluded a long-awaited global alliance with Hilton Hotels Corp., reuniting the Hilton brand worldwide for the first time in 32 years.
Despite the tie-up, which covers 400 hotels in 49 countries, the two companies denied there was a hidden agenda to progress toward a full merger of the two groups.
""We're taking a first step in bringing the hotels business much closer. We're putting the relationship back together and who knows what may come of it,"" said Ladbroke Chief Executive Peter George.
The companies said the deal covering sales and marketing, frequent customer programmes and development would fuel top-line growth and dispel the confusion that has surrounded the two groups in recent years.
Hilton has operated separately within and outside the U.S. market since 1964, when Hilton International was spun off from HHC.
""From our customers' point of view this will look as if it's one company for the first time in 32 years. For many years we've confused our guests ... They'll now no longer see a difference between a London Hilton and a Hilton in New York,"" said Stephen Bollenbach, chief executive of HHC.
""These two companies will have such a close alliance and so many points of contact with each other that I believe you will get all the benefits of one legal entity whether or not you become one legal entity.""
The reunification would also help both groups capitalise on industry growth and make them more resilient in cyclical downturns, George said.
Announcing the signing of the agreements for the hotel deal, which was first unveiled last summmer, the companies also said a worldwide loyalty programme -- Hilton HHonors Worldwide -- would start Feb. 1.
That would be the single most important marketing link between the two, George said, adding that there was scope for further cooperation in other areas such as gaming.
""There is a gaming provision as well whereby we're looking at doing some gaming projects together,"" said George.
""It's not a deal that involves a great deal of cost savings but the primary benefit is the top line -- it'll be driving revenues. It can be valued in the tens of millions of dollars to each company.""
As expected, the agreements provide for taking cross shareholdings of up to 20 percent. Hilton said it intended to acquire a 5 percent stake in Ladbroke in due course.
Mutual participation in future hotel development will focus primarily on management contracts and franchises, but the companies said it was no longer proposed that each make minority investments in the other's hotel real estate.
"
", 44),
        (""One of the hottest topics at a recent Internet trade show was so-called ""push technology,"" which directly broadcasts customised news to computer users hooked up to the Net -- but is also seen as the next area ripe for a shakeout     Internet broadcasters are led by companies
like privately held PointCast Inc., which announced a major deal with software giant Microsoft Corp. early last month.
Cupertino, Calif.-based PointCast and several other start-up companies deliver customised news -- or specific Web sites, depending on the service -- to consumer or corporate computer users, up to several times a day.
In other words, instead of searching the Web for the information they want, computer users have it sent directly to them.
The Yankee Group, a Boston research firm, predicts that by the year 2000 Internet broadcasting will be worth $5.7 billion in revenues. It now represents about $10 million in revenues a year.
""Making the Internet a broadcast medium is a direction that a lot of companies are going in, but that's already really crowded,"" said Ted Julian, an analyst with IDC Corp.
At the Internet World trade show, four of the best-known push technology companies exhibited and demonstrated their services, trying to differentiate themselves amid all the hype about direct Internet broadcasting.
But analysts said there are anywhere from 20 to 30 start-ups in this burgeoning area, as many companies are betting the concept will further change the World Wide Web and gain new advertiser interest.
""This is the new Web,"" said Steve Harmon, senior investment analyst at MecklerMedia Corp., host of the Internet World show.
PointCast was the first company to develop what are called personalized broadcast systems, and it now has 1.7 million subscribers. It delivers news from The New York Times, Boston Globe, Reuters Holdings Plc., and now MSNBC, the cable television-online news venture of Microsoft and General Electric Co.'s NBC.
PointCast and most of the other start-up companies operate on the same advertising-based model. They sell advertising space on their service and offer the service for free to subscribers.
By building up lots of subscribers, they can gain more advertising revenue.
The deal PointCast signed with Microsoft has the potential to significantly expand its subscriber base because PointCast will become part of Microsoft's next-generation computer desktop. Microsoft plans to incorporate its Internet Explorer Web browser in its next version of its Windows operating system, which dominates its market.
In a study last month, the Yankee Group said the Internet broadcasting market will significantly affect the way companies collect advertising dollars on the Web.
""The Web is hitting a wall"" in terms of people signing up for paid subscriptions for information, said analyst Melissa Bain of the Yankee Group. ""The Wall Street Journal (Interactive Edition) is successful, but not every content provider has had success. Users can get the information elsewhere for free.""
Bain said that, according to her report, 51 percent of all online users surveyed said that they typically access or use the same Web sites on a daily basis, making the ""push"" model attractive.
But with many more companies arriving on the scene and Microsoft and Netscape Communications Corp. getting into the market, consolidation is on the horizon, because media companies will not want to develop a channel for each delivery system, most of which, for now, are proprietary.
""There is no way I am going to download all four of these, and I am in the industry,"" said Jerry Yang, one of the founders of search engine power Yahoo! Corp. ""This is a huge issue.""
But Yang said that Yahoo! and other Internet search engines are looking at push technology because it will enhance a user's experience with the Web.
""Look for Yahoo to do a lot of development with one of these companies, or a Yahoo!-branded push channel,"" Yang said in a recent interview. ""We will all experiment.""
At Internet World, StarWave Corp., one of Microsoft co-founder Paul Allen's many investments, demonstrated a service it plans to introduce sometime next year. Patrick Naughton, StarWave's chief technology officer, said the service will be free and demonstrated a television-like advertisement for Levi's on ""StarWave TV.""
Even the players themselves are predicting a shakeout.
Harmon of Meckler predicts that push technology will fuel a hot new wave of Internet initial public offerings. But unlike the search engine fever, when a bevvy of small, unprofitable companies went public -- only to see their stock prices sharply decline -- consolidation will come first, he said.
Another player, Freeloader, was purchased last year by Individual Inc. of Burlington, Mass. Freeloader lets subcribers choose what Web sites they want delivered to their computers and obtain news via Individual.
""Clearly, there will be a big shakeout,"" said Freeloader President Sunil Paul, adding that there is a lot of chaos right now. ""PointCast clearly stands out, and we will be one of them (the survivors).
Another much-talked-about start-up, IFusion, is offering television-like content with video and audio in its service, which will be introduced early this year and called Arrive.
The company that created some of the recent frenzy about ""push"" technology is Marimba Inc., founded by several former Sun Microsystems Inc. employees.
Marimba did not exhibit at Internet World, but the Palo Alto, Calif.-based company, which develops software tools to create these ""channels"" using Sun's Java programming language, was mentioned by everyone. Marimba, along with PointCast, were seen as the potential early IPO candidates of this sector.
"
", 46),
        # Add more test samples
    ]
    
    if st.button("Evaluate Model Performance"):
        st.write("Evaluating model performance...")
        report, confusion_matrix = evaluate_model_performance(model, tokenizer, test_data)
        st.write("Model performance evaluation:")
        st.write("Classification Report:")
        st.write(report)
        st.write("Confusion Matrix:")
        st.write(confusion_matrix)

if __name__ == "__main__":
    main()
