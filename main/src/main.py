# local methods
from arg_parser import parse_args
from replicate_api import prompt
from replicate_api import check_length


def main():

    prompt()
    if not args.file:
        while True:
            input_text = str(input())
            check_length(input_text)
    else:
        file = open(args.file, "r")
        input_text = file.read()
        file.close()
        check_length(input_text)
    
    return 0

if __name__ == '__main__':
    args = parse_args()
    main()