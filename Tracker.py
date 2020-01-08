# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class Tracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.numbers = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.rects = OrderedDict()
        self.missing = []
    def register(self, centroid,number,rects):
        self.objects[self.nextObjectID] = centroid
        self.rects[self.nextObjectID] = rects
        self.disappeared[self.nextObjectID] = 0
        self.numbers[self.nextObjectID] = number
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.numbers[objectID]
        del self.rects[objectID]

    def update(self, rects, texts):
        self.missing=[]
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):

                self.missing.append(self.rects[objectID])
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects, self.numbers, self.rects, self.missing

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cY,cX)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i],texts,rects)

        else:

            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectNumbers = list(self.numbers.keys())
            objectRects = list(self.rects.keys())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)


            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()


            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue


                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.numbers[objectID] = texts[col]
                self.rects[objectID] = rects[col]
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.missing.append(self.rects[objectID])
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col],texts[col],rects[col])

        return self.objects, self.numbers, self.rects, self.missing