from flask import Flask, render_template, request, redirect, url_for
import main
import sanction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/individual_query', methods=['GET', 'POST'])
def individual_query():
    if request.method == "POST":

        name = request.form['name']
        gender = request.form['gender']
        dob = request.form['dob']
        nationality = request.form['nationality']
        sanctions_list = request.form['sanctions-list']

        if name: 
            individual_dict, model_output = main.run(name=name, nationality=nationality, gender=gender, dob=dob)
            
            ## background tile
            name = individual_dict['name']
            gender = individual_dict['gender'] if individual_dict['gender'] else 'Not Available'
            nationality = individual_dict['nationality'] if individual_dict['nationality'] else 'Not Available'
            dob = dob if dob else 'Not Available'


            no_of_articles = len(model_output)
            average_confidence = str(round(model_output['confidence_score'].mean(), 2)) + ' / 100'

            topic_count = list(topic_count_df(model_output).values())
            print(topic_count)

            if sanctions_list == 'yes':
                in_sanction = sanction.sanction_screening(name)
            else:
                in_sanction = 'Not Checked'

            print(in_sanction)

            return render_template('dashboard.html', name=name, nationality=nationality, gender=gender, dob=dob, model_output=model_output, no_of_articles=no_of_articles, average_confidence=average_confidence, topic_count=topic_count, in_sanction=in_sanction)

    return render_template('individual_query.html')

@app.route('/dashboard')
def dashboard():
		return render_template('dashboard.html')  # render a template

def topic_count_df(df):
    temp_dict = {
        "Financial Crimes" : len(df[df['sentiment_lstm'] == 'Financial Crime']),
        "Serious Crimes" : len(df[df['sentiment_lstm'] == 'Serious Crime']),
        "General News (Positive)" : len(df[df['sentiment_lstm'] == 'General News (Positive)']),
        "General News (Neutral)" : len(df[df['sentiment_lstm'] == 'General News (Neutral)']),
    }
    return temp_dict