import csv

def getNormalizedData(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        return getVector(csv_reader)


def getVector(data):
    input = []

    for row in data:
        try:
            normilized_row = [int(row[1])/10,int(row[2])/10,int(row[3])/10,int(row[4])/10,
                          int(row[5])/10,int(row[6])/10,int(row[7])/10,int(row[8])/10,int(row[9])/10]
            out = [0,0]

            if(int(row[10]) == 2):
                out[0] = 1
            else:
                out[1] = 1
            
            input.append({"input":normilized_row, "output":out})
        except:
            continue
    return input


if __name__ == "__main__":
    getNormalizedData('data/TestData.csv')