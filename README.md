The project is tested on python 3.6.9, and it may work on later versions as well.

Create a virtual environment using the command
```
python -m venv deep_tweets_workspace
```

Clone this project inside the environment.

## Dependencies
* Tensorflow
* Numpy
* Pandas
* PyTorch
* Rake
* Transformers
* Wandb (Weights and Biases)

Dependencies can be installed using pip:

```
pip install -r requirements.txt
```

Note, you'll need free a Weights and Biases account to run the GPT-2 models. The LSTM models can be run without it. Getting started: https://wandb.ai/

## LSTM models

To run the inference on the LSTM model, navigate to the `./models/LSTM` folder, and run the command

```
python inference.py [twitter_handle] [prompt]
```

To train the LSTM model from scratch, run the following command from the same directory
```
python training.py [twitter_handle]
```

`twitter_handle` can be either `realDonaldTrump` or `JoeBiden`, and `prompt` is the starting prompt of the sentence, e.g., `I like`

Note: Training the LSTM models take a significant amount of time. If you just want to obtain the results, run the inference.


## GPT-2 models

Navigate to the directory GPT2, and run the file `deeptweets_prompt.ipynb`. for the DeepTweets-Prompt model and the file `deeptweets_context.ipynb`. for the DeepTweets-Context model. You can change the following line in each file to replace the handle:

```
handle = [twitter_handle]
```

