from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField,TextAreaField
from wtforms.validators import InputRequired, Email, Length, AnyOf, DataRequired, URL
from flask_bootstrap import Bootstrap
import re
from predict_flask import classify

app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = 'DontTellAnyone'

class HomepageForm(FlaskForm):
	url = StringField('URL', validators=[URL(require_tld=True, message="Sahi URL daal bsdk")])
	headline = TextAreaField('Headline', validators=[InputRequired(), DataRequired()])
	body = TextAreaField('Body', validators=[InputRequired(), DataRequired()])


@app.route('/', methods=['GET', 'POST'])
def index():
	form = HomepageForm()
	if form.validate_on_submit():
		global document
		SITE = do_regex_match(form.url.data)
		document = [form.headline.data, form.body.data]
		relevance_data, score = classify(document)
		document.append(SITE)
		document.append(relevance_data)
		document.append(score)
		return redirect('result')
	return render_template('index.html', form=form)

def do_regex_match(t2s, pat = "(https?://(www.)?)([a-zA-Z]+)(\.[a-zA-Z0-9./-]+)", grp = 3):
    pattern = re.compile(pat, re.I)
    matches = pattern.finditer(t2s)
    
    for match in matches:
        print(match.group(grp))
        return match.group(grp)

@app.route('/result')
def result():
    return render_template('result.html', headline = document[0], body = document[1], url = document[2], relevance = document[3], score = document[4])

if __name__ == '__main__':
	app.run(debug=True)