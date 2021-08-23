import csv
import os
from typing import List
import numpy as np
import datetime
from pandas.core.tools.datetimes import to_time
import sklearn as sk
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
                self.record[menuItem.key] = -1
    def GetMenuItemName(self, key: str) -> str:
        for menuItem in self.menu.menuItems:
            if menuItem.key == key:
                return menuItem.name
def importServiceRecords(menu:Menu) -> list[ServiceRecord]:
    location = "E:\ServicePredictorTool\service_records"
    dirname = os.path.dirname(__file__)
    directory = os.path.join(dirname, location)
    os.chdir(location)
    records = []
    for file in os.listdir(directory):
        record = importCsvFile(file)
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
    with open("../service_report.csv", 'w') as csvfile:
        csvWriter = csv.writer(csvfile)
        csvWriter.writerows(csvOut)
        csvfile.close()
def parseMenu() -> Menu:
    menuList = importCsvFile("menu.csv")
    menu = Menu()
    for row in menuList:
        menu.AddMenuItem(Menu.MenuItem(row[0], row[1]))
    return menu

categories = ["Day", "Date", "Days Since Service", "Covers"]
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


menu = parseMenu()
records = importServiceRecords(menu)
printServiceRecords(records)
df,categories = marshalDataIntoFrame(categories, records, menu)
eDf = encodeWeekDays(df, categories)
print(eDf)
itemFrames = sliceDataFrame(eDf, menu)
print(itemFrames)
