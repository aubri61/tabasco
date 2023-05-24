import pandas as pd
import random
import json
import os
import openai
import dotenv
from datetime import datetime

import threading
import concurrent.futures


dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OUTPUT_PATH = './output/output.json'
ERROR_PATH = './output/error.json'


# managing data (save, store)
class DataManager:
    def __init__(self):
        dataset_dir_path = './datasets/'
        self.rotten_movies = pd.read_csv(dataset_dir_path+'rotten_movies.csv')
        self.rotten_reviews = pd.read_csv(
            dataset_dir_path+'original_rotten_reviews.csv')
        self.reviews_by_critic = pd.read_csv(
            dataset_dir_path+'reviews_by_critic.csv')

        self.drop_unamed_from_df()
        self.groupby_critic()

        self.output = None

        self.check_output()

    def drop_unamed_from_df(self):
        if 'Unnamed: 0' in self.rotten_movies.columns:
            self.rotten_movies = self.rotten_movies.drop('Unnamed: 0', axis=1)

        if 'Unnamed: 0' in self.rotten_reviews.columns:
            self.rotten_reviews = self.rotten_reviews.drop(
                'Unnamed: 0', axis=1)

        if 'Unnamed: 0' in self.reviews_by_critic.columns:
            self.reviews_by_critic = self.reviews_by_critic.drop(
                'Unnamed: 0', axis=1)

    def groupby_critic(self):
        self.grouped_critic = self.reviews_by_critic.groupby('critic_name').filter(
            lambda x: len(x) > 5 and (x['review_score'] > 0.7).any()).groupby('critic_name')

    def choose_gt_movies(self, reviews):
        # k =
        over_ratings = [r for r in reviews if r['review_score'] > 0.7]
        num_over = min(len(over_ratings), 10)
        ratings = random.sample(over_ratings, num_over)
        keys = ['movie_title', 'rotten_tomatoes_link',
                'review_score', 'review_content']
        ratings = [{k: v for k, v in r.items() if k in keys} for r in ratings]

        return ratings

    def choose_5_ratings(self, reviews, gt):
        ratings = [r for r in reviews if r['movie_title'] != gt['movie_title']]
        ratings = random.sample(ratings, k=5)
        keys = ['movie_title', 'review_score', 'review_content']
        ratings = [{k: v for k, v in r.items() if k in keys} for r in ratings]

        return ratings

    def init_output_file(self):
        columns = ['gt_movie', 'persona', 'critic_name',
                   'conversation', 'used_ratings', 'turn_num', 'is_casual']
        df = pd.DataFrame(columns=columns)

        for critic, group in self.grouped_critic:
            # print(f'index:  {index}, critic: {critic}')
            reviews = group.to_dict(orient='records')
            gt_movies = self.choose_gt_movies(reviews)

            for gt in gt_movies:
                turn_num = random.randint(10, 20)
                is_casual = random.choice([0, 1])
                ratings = self.choose_5_ratings(reviews, gt)

                el = {
                    # 'index': count,
                    'gt_movie': json.dumps(gt, ensure_ascii=False),
                    'persona': None,
                    'critic_name': critic,
                    'conversation': None,
                    'used_ratings': json.dumps(ratings, ensure_ascii=False),
                    'turn_num': turn_num,
                    'is_casual': is_casual
                }

                new_df = pd.DataFrame(el, index=[0])

                df = pd.concat([df, new_df], ignore_index=True)

        return df

    def check_output(self):

        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, 'r') as file:
                data = json.load(file)
                self.output = data

        else:
            output = self.init_output_file()
            output.reset_index(inplace=True)
            # output.to_json(OUTPUT_PATH, orient='records')
            self.output = output.to_dict(orient='records')
            with open(OUTPUT_PATH, "w") as file:
                json.dump(self.output, file)


class Generator:
    def __init__(self, data_manager: DataManager):
        self.groups = data_manager.grouped_critic
        self.data_manager = data_manager
        openai.api_key = OPENAI_API_KEY
        current_date = datetime.now().date()
        self.current_date = current_date.strftime('%Y-%m-%d')

    def request_openai_api(self, prompt, is_dialog=False):
        # \nYou are making movie-recommending conversation.
        # system_cont_dial =  "You are generating movie-recommending conversation between seeker and recommender, and have tremendous knowledges about movies.\nFollow the instruction as exactly as possible."
        system_cont_dial = "You are a movie-recommending conversation generator between seeker and recommender, who has tremendous knowledges about movies.\nAnswer as concisely as possible and follow the instruction as exactly as possible."
        system_cont_persona = "You are figuring out the peson's movie tastes and preferences and have tremendous knowledges about movies.\nAnswer as concisely as possible and follow the instruction as exactly as possible."
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[
                                                    {"role": "system", "content": system_cont_dial if is_dialog else system_cont_persona},
                                                    {"role": "user",
                                                        "content": prompt},
                                                ], temperature=0.2, max_tokens=3000,
                                                frequency_penalty=1,
                                                )

        result = response['choices'][0]['message']['content']
        return result

    def generate_persona_prompt(self, gt, ratings):
        prompt = f"""Your task is to write a person's tastes and preferences based in terms of movie selection.
Here are the past seen movies of the person:
{ratings}

The person's favorite movie:
{gt}

Give me short and lively summary of this person's movie tastes in 6 sentences. Include the person's preferred genres, themes, styles, and any notable patterns or preferences that emerge from the movie-watching history. Don't mention movie titles nor review scores nor comments. Start with 'This person', never say 'based on~'.
  """
        return prompt

    def generate_dialog_prompt(self, gt, persona, turn_num, is_casual):

        return f"""Generate a movie recommending dialogue between the 'Seeker' and the 'Recommender'. The Seeker has a specific movie taste, and the goal is to recommend a movie that aligns with their preferences. You will be provided with the movie taste of the Seeker and the movie that should be recommended at the end. The dialogue should flow naturally, with the Recommender leading the conversation to uncover the Seeker's preferences and eventually make the perfect movie recommendation. Each utterance should be less than 15 words. Feel free to include additional dialogue paths, open-ended questions, movie watching history, and responses that encourage the Seeker to reveal their tastes, mood, or specific movie preferences.

The Seeker's Movie Taste:
{persona}

Movie to Recommend:
"{gt}"

 ATTENTION! When referring to movies, enclose the `title (released year)` in square brackets. ex) [Parasite (2019)]. Never enclose utterances in quotation marks. ATTENTION! The conversation should have at least 10 exchanges between the Seeker and the Recommender. ATTENTION! The conversation does not consider time sequences which means that the conversation is a single session that concludes right here."""

    def save_output_file(self):
        with open(OUTPUT_PATH, 'w') as file:
            json.dump(self.data_manager.output, file)

        return

    def record_error_point(self, index, error):
        if os.path.isfile(ERROR_PATH):
            # Load existing data from the file
            with open(ERROR_PATH, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append({"index": index, "error_message": error})

        with open(ERROR_PATH, 'w') as file:
            json.dump(data, file)

    def process_item(self, item):
        try:
            print(f'working on {item["index"]}...')

            gt = item['gt_movie']
            ratings = item['used_ratings']
            persona_prompt = self.generate_persona_prompt(gt, ratings)
            persona = self.request_openai_api(persona_prompt, is_dialog=False)
            turn_num = item['turn_num']
            is_casual = item['is_casual']

            gt = json.loads(gt)
            
            
            dialog_prompt = self.generate_dialog_prompt(
                gt['movie_title'], persona, turn_num, is_casual)

            dialog = self.request_openai_api(dialog_prompt, is_dialog=True)

            item['persona'] = persona
            item['conversation'] = dialog

            return item

        except Exception as e:
            error_message = str(type(e)) + str(e)
            print(f"error occurred: {error_message}")
            self.record_error_point(item["index"], error_message)

    def generate_dialog(self):
        count = 0
        save_interval = 10
        max_workers = 3 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()  # Change to a set
            

            for item in self.data_manager.output:
                if item['conversation'] is None:
                    future = executor.submit(self.process_item, item)
                    futures.add(future)
                    # thread = threading.Thread(
                    #     target=self.process_item, args=(item,))
                    # thread.start()

                    count += 1

                    if count % save_interval == 0:
                        concurrent.futures.wait(futures)
                        # thread.join()
                        self.save_output_file()
                        
                     # Wait until there are fewer than max_workers futures remaining
                    while len(futures) >= max_workers:
                        done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                        
            concurrent.futures.wait(futures)

        return 1
