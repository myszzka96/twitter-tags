# Slang

_Text understanding._

Slang uses RASA, an open source dialogue agent framework. Slang allows you to
map sentences to intents and parse out certain target words. For example, you
can create a new intent in markdown format like so:

```markdown
## intent:pick_up
- pick up a [banana](object)
- go get me a [monster](object)
- grab a new brick of [bustello](object)
```

### Configuration

The config file can take either the pipeline of `tensorflow_embedding` or
`spacy_sklearn`. `spacy` is good for small configs because it uses pre-trained
word vectors from GloVe and fastText, so it'll interpolate between similar
words. `tensorflow_embedding` creates an all-new model, so it's good for large
data sets, or data sets for which common words have different meaninings.
Tensorflow is also fine if you don't expect a lot of variance in the sentence
structure. For example, if you're listening for pick and place commands, it's
probably going to be in the structure of

- "Bring a(n) `<object>` to (the) `<location>`."
- "Pick up a(n) `<object>` and place it at (the) `<location>`."

We could probably parse this with a grammar, but constructing the trained model
allows us a little more flexibility.

The configuration file should look like this:

```yml
# my_config.yml
language: en
pipeline: tensorflow_embedding
```

Or alternatively this:

```yml
# my_config.yml
language: en
pipeline: spacy_sklearn
```

## Generating a New Model

You can write a markdown file as above, but the fastest way to create a bunch
of data is to use [Chatito](https://rodrigopivi.github.io/Chatito/). See
`./chatitos/bringLocations.chatito` for an example. Use the
[online editor](https://rodrigopivi.github.io/Chatito/) to create new data and
then use the "Generate Dataset" button. Enforce the "RASA NLU" Dataset format.
That will give you a `.json` file to train on.

## Training a New Model

You'll need to install all the dependencies first. Create a virtual environment
and install the `requirements.txt`

```bash
$ python --version
2.7.12
$ python -m virtualenv env    # "env" is the name of the virtualenv you're creating
$ source env/bin/activate
(env) $ type python
/path/to/your/env/bin/python
(env) $ python --version
2.7.12
(env) $ type pip
/path/to/your/env/bin/pip
(env) $ pip install -r requirements.txt
```

Once you have the data (either an `.md` or `.json` file), you can train like so:

```bash
(env) $ python -m rasa_nlu.train -c <config.yml> --data <examples.md> -o <output-dir> --fixed_model_name <project-name> --project <project-name> --verbose
```

## ROS Service

In order to run the service, you'll need the virtual env. The order is

```bash
$ source env/bin/activate
(env) $ source ~/rasa_ws/devel/setup.bash
(env) $ rosrun slang translator.py
```

This ensures that you're falling back on the right packages. Thanks python.

### Non-Python Dependencies
```bash
$ sudo apt install mpg123
```

### Add this line to your .bashrc
```
export GOOGLE_APPLICATION_CREDENTIALS='<path_to_google_cloud_texttospeech_credentials.json>'
```

### For now the credentials file path is:
```
export GOOGLE_APPLICATION_CREDENTIALS='<path_to_slang>/Palpi-project.json'
```
