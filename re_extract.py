import re
import os


if __name__ == '__main__':
    for file in os.listdir('slurms'):
        text = open(f'slurms/{file}', encoding='ISO-8859-1').read()
        matches = re.findall(r"{'ev.*?}", text, re.DOTALL)
        with open(f'outs/fixed/{file.replace(".out", "-fixed.out")}', 'w') as wr:
            for match in matches:
                wr.write(match+'\n')
