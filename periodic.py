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
    start = max(0, int(np.round(norm.ppf(PROB_TRESHOLD) * sd + mean)))
    end = int(np.round(norm.ppf(1 - PROB_TRESHOLD) * sd + mean))
    for i in range(start, end + 1):
        p = norm.cdf((i - mean + 0.5) / sd) - norm.cdf((i - mean - 0.5) / sd)
        probs.append(p)
    highest = max(probs)
    scores = [round(i / highest * 20) for i in probs]
    for i, day in enumerate(range(start, end + 1)):
        print(firstDay + timedelta(days=day), f'{round(probs[i] * 100, 1):>5.1f}% ', scores[i] * '\u25A0')

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

    def showFertility(self):
        # https://www.demographic-research.org/volumes/vol3/5/3-5.pdf
        cycleData = self.getCycleLengths()
        periodData = self.getPeriodLengths()

        mCyc, sCyc = computeStats(cycleData)
        mPer, sPer = computeStats(periodData)

        t, d = self.events[-1]
        nextStart = d.toordinal() + mCyc
        nextStartDev = sCyc
        if t == 's':
            nextStart += mPer
            nextStartDev = np.hypot(nextStartDev, sPer)

        BBTDay = nextStart - 14
        BBTDev = np.hypot(nextStartDev, 2)

        print('The BBT day can be expected on:')
        print()
        drawGraph(date.today() - timedelta(days=50), BBTDay - date.today().toordinal() + 50, BBTDev)

    def predictMore(self, n: int):
        cycleData = self.getCycleLengths()
        periodData = self.getPeriodLengths()

        mCyc, sCyc = computeStats(cycleData)
        mPer, sPer = computeStats(periodData)

        t, d = self.events[-1]
        nextStart = d.toordinal() + mCyc
        nextStartDev = sCyc
        if t == 's':
            nextStart += mPer
            nextStartDev = np.hypot(nextStartDev, sPer)

        nextStart += (mPer + mCyc) * n
        nextStartDev = np.hypot(nextStartDev, np.hypot(sCyc, sPer) * np.sqrt(n))

        print(f'Start of next {n+1}-th period be expected on:')
        print()
        drawGraph(date.today(), nextStart - date.today().toordinal(), nextStartDev)

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

        if cmd in ('s', 'e'):
            datearg = getDate(args)
            data.events.append((cmd, datearg))
            longcmd = {"s": "start", "e": "end"}[cmd]
            print(f'Period {longcmd}ed on {datearg}.')
            data.write()
            data.showCondition()

        elif cmd == 'd':
            datearg = getDate(args)
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

        elif cmd == 'f':
            data.showFertility()

        elif cmd == 'p':
            try:
                n = int(args[0]) if args else 1
                data.predictMore(n - 1)
            except ValueError as err:
                assert 0, err

        elif cmd == 'l':
            for t, d in data.events:
                longcmd = {"s": "start", "e": "end"}[t]
                print(d, longcmd)

        elif cmd == 'h':
            print(
f"""
Usage: {APP} [command] [date/n]

These are availible commands
    Condition       Show the current periodic cycle condition
    Fertility       Show the fertility summary
    Help            Show this message
    List            List the periodic cycle history
    Start [date]    Add entry: the period started
    End [date]      Add entry: the period ended
    Delete [date]   Delete the entry at given date
    Predict [n]     Predict the start and end of n-th next period

It is sufficient to type the first letter of command.
Date is always optional. If it is not supplied, today's date will be used.
""")
        else:
            assert False, f'Unknown command: {cmd}'

    except AssertionError as err:
        print(err)
        print(f'Use "{APP} h" for help')

if __name__ == '__main__':
    main()
