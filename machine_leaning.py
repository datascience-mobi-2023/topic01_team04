# This is where the code belongs

#ich bin verwirrt:

print('test')

test = []

for n in 0,5:
    test += [n]
    print(test)

filepath = "C:\Users\Ole Decker\Documents\GitHub\topic01_team04\Data\fashion-mnist_train.csv"


def load_csv(C:\Users\Ole Decker\Documents\GitHub\topic01_team04\Data\fashion-mnist_train.csv):
    data =  []
    col = []
    checkcol = False
    with open(C:\Users\Ole Decker\Documents\GitHub\topic01_team04\Data\fashion-mnist_train.csv) as f:
        for val in f.readlines():
            val = val.replace("\n","")
            val = val.split(',')
            if checkcol is False:
                col = val
                checkcol = True
            else:
                data.append(val)
    df = pd.DataFrame(data=data, columns=col)
    return df
