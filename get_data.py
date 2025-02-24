import csv
from collections import deque

def get_data(filnamn:str):
    data = []
    with open(filnamn, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        temp = deque()
        for i, line in enumerate(csv_reader):
            temp.append(line[1])
            if i == 9:
                break
        for line in csv_reader:
            temp.append(line[1])
            data.append(([*temp][:-1], temp[-1]))
            temp.popleft()
    return data