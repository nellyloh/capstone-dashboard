import individual_query 
import webscraper_confidence_score 
import sentiment_model

def run(name=None, nationality=None, gender=None, dob=None):
	individual_dict = individual_query.preprocess_input_to_dict(name, nationality=nationality, gender=gender, dob=dob)
	print(individual_dict)
	articles = webscraper_confidence_score.search_articles_on_individual(individual_dict, no_of_articles=10, additional_keywords=None)
	print(articles)
	model_output = sentiment_model.sentiment_model(articles)
	print(model_output)
	return individual_dict, model_output