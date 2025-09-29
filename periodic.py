from datetime import date, timedelta
from platformdirs import user_config_dir, user_data_dir
from os import path
from sys import argv
import re
import numpy as np
from scipy.stats import norm

APP = 'periodic'
DATA_DIR = user_data_dir(APP)
CONFIG_DIR = user_config_dir(APP)
OLD_DATA_COEFF = 0.9
PROB_TRESHOLD = 0.05

def computeStats(rawdata: list[int]):
    data = np.array(rawdata).astype(np.float64)
    weights = np.array([OLD_DATA_COEFF ** i for i in range(len(data))]).astype(np.float64)
    mean = np.average(data, weights=weights)
    V1 = np.sum(weights)
    V2 = np.sum(weights ** 2)
    sd = np.sqrt(np.sum(weights * (data - mean) ** 2) / (V1 - V2 / V1))
    return mean, sd

def drawGraph(firstDay: date, mean: np.floating, sd: np.floating):
    probs = []
    for i in range(100):
        p = norm.cdf((i - mean + 0.5) / sd) - norm.cdf((i - mean - 0.5) / sd)
        probs.append(p)
    start = int(np.round(norm.ppf(PROB_TRESHOLD) * sd + mean))
    end = int(np.round(norm.ppf(1 - PROB_TRESHOLD) * sd + mean))
    highest = max(probs)
    scores = [round(i / highest * 20) for i in probs]
    for i in range(start, end + 1):
        print(firstDay + timedelta(days=i), str(round(probs[i] * 100, 1)) + '%', scores[i] * '\u25A0')

class DataHandler:
    def __init__(self):
        self.events: list[tuple[str, date]] = []
        try:
            self.load(path.join(DATA_DIR, 'data'))
        except FileNotFoundError:
            pass

    def sortEvents(self):
        self.events.sort(key = lambda x: x[1])

    def write(self):
        self.sortEvents()
        self.save(path.join(DATA_DIR, 'data'))

    def load(self, filename):
        self.events = []
        with open(filename, 'r') as fh:
            for line in fh:
                line = line.split('#')[0].strip()
                if not line: continue
                t, d = line.split()
                d = date(*map(int, d.split('-')))
                self.events.append((t, d))
        self.sortEvents()

    def save(self, filename):
        with open(filename, 'w') as fh:
            for t, d in self.events:
                fh.write(f'{t} {d}\n')

    def getCycles(self):
        cycles = []
        start = None
        for t, d in self.events:
            if t == 's':
                start = d
            elif start is not None:
                cycles.append([start, d])
        return cycles

    def getCycleLengths(self) -> list[int]:
        res = []
        cycles = self.getCycles()
        for i in range(len(cycles) - 1):
            d = (cycles[i + 1][0] - cycles[i][1]).days
            if d > 80:
                res += [d // 3] * 3
            elif d > 50:
                res += [d // 2] * 2
            else:
                res.append(d)

        return res

    def getPeriodLengths(self) -> list[int]:
        cycles = self.getCycles()
        return [(e - s).days for s, e in cycles]

    def showCondition(self):
        cycles = self.getCycles()
        if len(cycles) <= 2:
            print('Not enough data was collected in order to make predictions about period cycle')
            return
        status, day = self.events[-1]

        if status == 'e':
            print('Period is not active')
            print('The start of period can be expected on:')
            print()
            data = self.getCycleLengths()

        else:
            print('Period is active')
            print('The end of period can be expected on:')
            print()
            data = self.getPeriodLengths()

        drawGraph(day, *computeStats(data))

def parseDate(s: str):
    args = re.split(r'\D+', s)
    assert len(args) == 3, 'date must have 3 numbers'
    return date(*map(int, args))

def getDate(args: list[str]) -> date:
    if len(args) == 0:
        return date.today()

    try:
        return parseDate(' '.join(args))
    except ValueError as err:
        assert 0, err

    return date.today()

def main():
    try:
        data = DataHandler()
        _, *args = argv
        if args:
            cmd, *args = args
        else:
            cmd = 'c'
        cmd = cmd[0].lower()
        datearg = getDate(args)

        if cmd in ('s', 'e'):
            data.events.append((cmd, datearg))
            longcmd = {"s": "start", "e": "end"}[cmd]
            print(f'Period {longcmd}ed on {datearg}.')
            data.write()
            data.showCondition()

        elif cmd == 'd':
            l1 = len(data.events)
            data.events = [i for i in data.events if i[1] != datearg]
            data.write()
            l2 = len(data.events)
            if l1 == l2:
                print('No entry was deleted')
            else:
                print('Entry deleted')

        elif cmd == 'c':
            data.showCondition()

        elif cmd == 'l':
            for t, d in data.events:
                longcmd = {"s": "start", "e": "end"}[t]
                print(d, longcmd)

        elif cmd == 'h':
            print(
f"""
Usage: {APP} [command] [date]

These are availible commands
    Condition       Show the current periodic cycle condition
    Help            Show this message
    List            List the periodic cycle history
    Start [date]    Add entry: the period started
    End [date]      Add entry: the period ended
    Delete [date]   Delete the entry at given date

It is sufficient to type the first letter of command.
Date is always optional. If it is not supplied, today's date will be used.
""")

    except AssertionError as err:
        print(err)
        print(f'Use "{APP} h" for help')

if __name__ == '__main__':
    main()
