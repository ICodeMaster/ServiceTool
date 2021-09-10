import csv
import os
from typing import List
import numpy as np
import datetime
from pandas.core.frame import DataFrame
from pandas.core.tools.datetimes import to_time
import pickle
import sklearn as sk
from sklearn.svm import SVR
import sklearn.preprocessing as pre
import pandas as pd
import numpy as np

def importCsvFile(filename: str) -> list:
    with open(filename, newline='') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        outputList = []
        for row in csvReader:
            outputList.append(row)
        csvfile.close()
        return outputList
class Menu:

    def __init__(self):
        self.menuItems = []
    class MenuItem:
        def __init__(self, name, key):
            self.name = name
            self.key = key
    def AddMenuItem(self, item: MenuItem ):
        self.menuItems.append(item)
    def RemoveMenuItem(self, item: MenuItem):
        self.menuItems.remove(item)
    def CountMenuItems(self) -> int:
        return self.menuItems.count

class ServiceRecord:
    def __init__(self, date: datetime.date, day, covers, menu: Menu):
        self.menu = menu
        self.date = date
        self.day = day
        self.covers = covers
        self.record = {}
    def AddMenuCount(self, counts: dict[str, int]):
        for menuItem in self.menu.menuItems:
            if any(recordItem == menuItem.key for recordItem in counts):
                self.record[menuItem.key] = counts[menuItem.key]
            else:
                self.record[menuItem.key] = np.nan
    def GetMenuItemName(self, key: str) -> str:
        for menuItem in self.menu.menuItems:
            if menuItem.key == key:
                return menuItem.name
def importServiceRecords(menu:Menu) -> list[ServiceRecord]:
    location = "service_records"
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, location)
    records = []
    for file in os.listdir(directory):
        fileDir = os.path.join(directory, file)
        record = importCsvFile(fileDir)
        header = record.pop(0)
        date = datetime.date.fromisoformat(file[:10])
        serviceRecord = ServiceRecord(date, header[0], header[1], menu)
        counts = {}
        for row in record:
            counts[row[0]] = row[1]
        serviceRecord.AddMenuCount(counts)
        records.append(serviceRecord)
    return records
def printServiceRecords(records: list[ServiceRecord]):
    csvOut = []
    for record in records:
        date = record.date.isoformat()
        header = [record.day, date, record.covers]
        csvOut.append(header)
        for key, item in record.record.items():
            name = record.GetMenuItemName(key)
            row = [name, item]
            csvOut.append(row)
        csvOut.append("")
    path = os.path.dirname(__file__)
    path = os.path.join(path, "service_report.csv")
    with open(path, 'w') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerows(csvOut)
        csvfile.close()
def parseMenu() -> Menu:
    menuList = importCsvFile("E:\ServicePredictorTool\menu.csv")
    menu = Menu()
    for row in menuList:
        menu.AddMenuItem(Menu.MenuItem(row[0], row[1]))
    return menu
def appendCategoriesForMenu(list: list, menu: Menu) -> list:
    for menuItem in menu.menuItems:
        list.append(menuItem.key)
    return list
def marshalDataIntoFrame(categories: list, records: list[ServiceRecord], menu: Menu) -> pd.DataFrame:
    categories = appendCategoriesForMenu(categories, menu)
    list_for_array = []
    for record in records:
        today = datetime.date.today()
        daysService = (today - record.date).total_seconds()
        daysService = np.floor_divide(daysService, 86400)
        row = [record.day, record.date, daysService, record.covers]
        for _, item in record.record.items():
            row.append(item)
        list_for_array.append(row)
    array = np.array(list_for_array)
    df = pd.DataFrame(array, columns=categories).set_index('Date')
    df.iloc[:,2:] = df.iloc[:, 2:].astype(float)
    return df, categories

def encodeWeekDays(data: pd.DataFrame, categories: list):
    weekdays = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    oe = pre.OrdinalEncoder( categories=[weekdays])
    outData = oe.fit_transform(data[['Day']])[:]
    data["dayEncode"] = outData
    data = dropCategorical(data, categories)
    return data
def dropCategorical(data: pd.DataFrame, categories: list):
    to_remove = [feature for feature in categories if feature in ["Day"]]
    to_keep = [ col for col in data.columns if col not in to_remove ]
    return data[to_keep]

def sliceDataFrame(data: pd.DataFrame, menu: Menu) -> list[pd.DataFrame]:
    baseFeatures= ["Days Since Service", "Covers", "dayEncode"]
    dataFrames = []
    for menuItem in menu.menuItems:

        ## Use list() due to mutable type
        features = list(baseFeatures)
        features.append(menuItem.key)
        itemData = data.loc[:, features]
        indecies = itemData[itemData.iloc[:,3] == -1].index
        itemData.drop(indecies, inplace=True)
        dataFrames.append(itemData)
    return dataFrames
def findAverageMenuItemCount(df: DataFrame) -> DataFrame:

    countFrame = df.iloc[:, 2:-1]
    df["countAv"] = countFrame.mean(1)
    return df


def trainModel(menu: Menu):
    categories = ["Day", "Date", "Days Since Service", "Covers"]
    records = importServiceRecords(menu)
    printServiceRecords(records)
    df,categories = marshalDataIntoFrame(categories, records, menu)
    eDf = encodeWeekDays(df, categories)
    eDf = findAverageMenuItemCount(eDf)
    ### This is broken idk. Need to import direct into Pandas
    eDf['Days Since Service'].astype(float)
    print(eDf)
    svr_poly = SVR(kernel = 'linear', gamma='auto')
    featureCols = ['dayEncode']
    X = eDf.loc[:, featureCols]
    Y = eDf['Covers']
    weight = eDf['Days Since Service']
    max = weight.max()
    weight = weight.sub(max).multiply(-1)
    print(X.shape, Y.shape, weight.shape)
    svr_poly.fit(X, Y, sample_weight=weight)
    print("Fit with last three points:  ",svr_poly.score(X.iloc[-3:].to_numpy(), Y.iloc[-3:]))
    path = os.path.dirname(__file__)
    path = os.path.join(path, "svr_pickle.pkl")
    svrFile = open(path, 'ab')
    pickle.dump(svr_poly, svrFile)
    svrFile.close()
    trainModelForEachItem(menu, eDf, weight)
def trainModelForEachItem(menu: Menu, df: DataFrame, weight) -> list[SVR]:
    frame = df.iloc[:, 2:-2]
    for (columnName, columnData) in frame.iteritems():
        itemFrame = pd.DataFrame(columnData, dtype=float)
        itemFrame['Covers'] = df['Covers']
        itemFrame['dayEncode'] = df['dayEncode']
        itemFrame['weight'] = weight
        itemFrame = itemFrame.dropna()
        print(itemFrame)
        svr_item = SVR(kernel='linear', gamma='auto')
        #svr_item.fit(X, columnData.values, sample_weight=weight)
menu = parseMenu()
trainModel(menu)
