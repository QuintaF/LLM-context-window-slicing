import os
import numpy
import math
import nltk
from nltk.corpus import stopwords

#local methods
from arg_parser import parse_args
args = parse_args()

#for LLM API usage: "https://replicate.com/meta/llama-2-70b-chat/api?tab=python"
#if there is an authentication error get a new token
import replicate
os.environ['REPLICATE_API_TOKEN'] = 'your_key' # set your key 
REPLICATE_CONTEXT_WINDOW = 4096


def tokenization(text):
    '''
    transforms a text into a list of tokens(punctuation is removed)

    :returns: list of tokens
    '''

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [tkn.lower() for tkn in tokens]

    return tokens


def cosine_similarity(a, b):
    '''
    computes cosine similarity between vector a and b

    :returns: similarity value
    '''

    return numpy.dot(a, b)/(numpy.linalg.norm(a)* numpy.linalg.norm(b))


def similarity(slice1, slice2):
    '''
    creates BoWs for two slices
    and words value vectors
    
    :returns: cosine similarity
    '''

    slices = [slice1 , slice2]

    #lemmatizing
    lemmatized_tokens = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for token_list in slices:
        lst = [lemmatizer.lemmatize(token) for token in token_list]
        lemmatized_tokens.append(lst)

    #Bag of Words
    BoWs = []
    for token_list in lemmatized_tokens:
        BoWs.append(nltk.FreqDist(token_list))

    #creation of word vectors
    slice_1_tokens = BoWs[0].elements()
    slice_2_tokens = BoWs[1].elements()
    vectors = {}

    words = set()
    for token in slice_1_tokens:
        words.add(token)

    for token in slice_2_tokens:
        words.add(token)
        
    for token in words:
        c = 0
        tf_1 = math.log10(1 + BoWs[0].get(token) if BoWs[0].get(token) is not None else 1)
        tf_2 = math.log10(1 + BoWs[1].get(token) if BoWs[1].get(token) is not None else 1)

        if tf_1 > 0:
            c += 1
        if tf_2 > 0:
            c += 1

        idf = 1 + math.log10(2/c)

        vectors[token] = [tf_1*idf, tf_2*idf]

    matrix = []
    for val in vectors.values():
        matrix.append(val)

    matrix = numpy.asarray(matrix)
    matrix_t = numpy.transpose(matrix) # to get vector of words values

    return cosine_similarity(matrix_t[0], matrix_t[1])


def slicing(tokens):
    '''
    generates slices of the input text that meet this constraints:
        - each fits the context window
        - their sum length is >= to input length
        - two slices can overlap, but no slice is included in another one;
        - two by two the slices have to be different "enough"(20% similarity threshold).

    :returns: slices
    '''

    slices = []
    prompt1 = [token for token in tokens[:REPLICATE_CONTEXT_WINDOW]]

    c = 0
    slices.append(prompt1)
    while len(prompt1) >= REPLICATE_CONTEXT_WINDOW:
        #the first slice starts shortly after the first one and goes to a whole context window length
        prompt2 = [token for token in tokens[REPLICATE_CONTEXT_WINDOW*c + REPLICATE_CONTEXT_WINDOW//4 : 1+ REPLICATE_CONTEXT_WINDOW*(c+1) + REPLICATE_CONTEXT_WINDOW//4]]

        # checks similarity between slices until they're different enough, or just before losing input information by slicing
        similar = 4
        while similarity(prompt1, prompt2) >= 0.2:
            prompt2 = [token for token in tokens[REPLICATE_CONTEXT_WINDOW*c + (REPLICATE_CONTEXT_WINDOW//similar) : 1 + REPLICATE_CONTEXT_WINDOW*(c+1) + (REPLICATE_CONTEXT_WINDOW//similar)]]
            if similar == 1:
                break

            similar -= 1
            
        slices.append(prompt2)
        prompt1 = prompt2
        c += 1

    return slices


def check_length(input_text):
    '''
    checks that the input fits inside the 
    context_window of the LLM.
    '''

    #tokenized text
    tokenized_text = tokenization(input_text)
    
    #stopword elimination
    nltk.download("stopwords", quiet="True")
    stop_words = stopwords.words("english")

    filtered_list = []
    for token in tokenized_text:
        if token.casefold() not in stop_words:
            filtered_list.append(token)


    if len(input_text) > REPLICATE_CONTEXT_WINDOW:
        slices = slicing(tokenized_text)
        detokenizer = nltk.TreebankWordDetokenizer()
        for prompt_slice in slices:
            prompt(detokenizer.detokenize(prompt_slice) )
    else:
        prompt(input_text)


def prompt(prompt = None):
    '''
    runs the model with the given prompt
    '''

    if not prompt:
        prompt = "..."
    elif args.file:
        print("\n\nYou: ", end="")
        print(prompt)


    # The meta/llama-2-70b-chat model can stream output as it's running.
    print("\nLLaMA: ", end="")
    for event in replicate.stream(
                            "meta/llama-2-70b-chat",
                            input={
                                "top_p": 0.9,
                                "prompt": "[INST]" + prompt + "[/INST]",
                                "temperature": 0.5,
                                "system_prompt": """You are a helpful, respectful and honest assistant. 
                                                    Always answer as helpfully as possible, while being safe.
                                                    If a question does not make any sense, or is not factually coherent, 
                                                    explain why instead of answering something not correct.
                                                    """,
                                "max_new_tokens": 500,
                            },):
        print(str(event), end="")

    if not args.file:
        print("\nYou: ", end="")