import argparse

def parse_args():
    '''
    builds a parser for 
    command line arguments

    :returns: args values
    '''
    parser = argparse.ArgumentParser()
    
    #default mode is input from keyborad
    parser.add_argument("--file", "-f", type=str, help="given a path to a file it uses it as input text")

    return parser.parse_args()