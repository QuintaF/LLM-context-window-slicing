# LLM-project
Second assignment of the NLP course. Use a LLM to either accomplish (CASE 1) or (CASE 2).<br>
Informations on the LLM:

| Model | Training Data | Params | Context-length | GroupedQueryAttention | Tokens | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA 2 | A new mix of <br>publicly available <br>online data | 70B | 4k | yes | 2.0T | 1.5 x 10^-4 |

more in the official paper: https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/


## Before Usage

An API key is needed for the program to work properly otherwise an authentication error is shown: 
```
replicate.exceptions.ReplicateError: You did not pass an authentication token
```

Get your token from the replicate API website and put it in the [__replicate_api.py__](main/src/replicate_api.py) file at the __14th__ row inside the quotation marks:
```
os.environ['REPLICATE_API_TOKEN'] = 'your_key' # set your key 
```

## Usage
Start the algorithm:
```
usage: main.py [-h] [--file FILE]
```
Usage options:
```
options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  given a path to a file it uses it as input text
```
A default execution(wihtout -f) allows for a continous exchange with the LLM (it's on the user to end the program), although it is important to observe that the LLM lacks memory of past interactions, as the API does not incorporate a way for managing history.

### llama-2-70b-chat API usage
Folowing some informations about the model parameters used when sending the user prompt to the LLM. For further insights on the usage of the model refer to the [official README](https://replicate.com/meta/llama-2-70b-chat/readme).<br>
Inputs:
* __prompt(string)__:\
  Prompt sent to the model.

* __system_promp(string)__\
  System prompt sent to the model. This is prepended to the prompt and helps guide system behavior.

* __max_new_tokens(integer)__\
  Maximum number of tokens to generate. 

* __temperature(number)__\
  Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.
  
* __top_p(number)__\
  When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens

## Repository Structure

main/\
├── src/\
│&emsp;&emsp;├── arg_parser.py&emsp;...python parser for command line arguments.\
│&emsp;&emsp;├── main.py&emsp;...optional file reading, call to the model API.\
│&emsp;&emsp;└── replicate_api.py&emsp;...prompts to the API and prompt engineering functions\
│\
└── test/\
&emsp;&emsp;&emsp;└── ...test files.



## Assignment
In this assignment, candidates are either required to:<br> 
(CASE 1) implement an algorithm to generate slicing of excessive context window for a LLM<br>
(CASE 2) implement a hierarchical system for summarization in the same system. 
The implementation for both choices are to be either in Python, while using NLTK, or in Java while using OpenNLP.

CASE 1:
The method is based on the following pipeline:

* When the input is below the standard size of the context window (4k tokens) is then passed "as it is" to the LLM;

* When the input is above the standard size is subdivided in a finite number of slices each of a size that
   fits the context window and such that they sum to a number N greater than or equal to the size of the input length;

* The criteria to generate a coverage as provided above are:

      ** Two slices can overlap;
      ** No slice is included in another one;
      ** When two adjacent slices are settled, the two slices have to be different "enough".

Ideal solutions will be based on the comparison of two slices based on cosine distance of bag of words constructed by the usual pipeline of stopword elimination, stemming/lemmatization and count of occurrences weighted on the length of the document after the steps above.<br> 
The setup of the threshold for distance is empirical, no need to settle it by experiments (use reasonable threshold like 20%).

Once the prompt engineering algorithm has been run, we shall collect the results and use them as they are, so the assignment does not require ex-post filtering.

The assignment does not require the evaluation of the performances.

---