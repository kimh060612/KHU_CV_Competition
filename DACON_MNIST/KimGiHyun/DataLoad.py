import numpy as np
import matplotlib.pyplot as plt

train_file_path = "./train.csv"
test_file_path = "./test.csv"

DigitIndex = ["0","1","2","3","4","5","6","7","8","9"]
LetterIndex = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

train = True

if train :
    with open(train_file_path, "r") as File:
        List = File.readlines()
        TargetDigit = []
        TargetLetter = []
        Image = []
        for i in range(1, len(List)):
            Col = List[i].split(",")
            Digit = np.zeros(10)
            Letter = np.zeros(26)
            Digit[DigitIndex.index(Col[1])] = 1
            Letter[LetterIndex.index(Col[2])] = 1
            img = Col[3:]
            img = np.array(img).reshape(28, 28, 1)
            TargetDigit.append(Digit)
            TargetLetter.append(Letter)
            Image.append(img)
        Image = np.array(Image, dtype="float64")
        TargetDigit = np.array(TargetDigit, dtype="float64")
        TargetLetter = np.array(TargetLetter, dtype="float64")
        print(Image.shape)
    np.save("./Image", Image)
    np.save("./TargetDigit", TargetDigit)
    np.save("./TargetLetter", TargetLetter)

else :
    with open(test_file_path, "r") as File:
        List = File.readlines()
        TargetLetter = []
        Image = []
        for i in range(1, len(List)):
            Col = List[i].split(",")
            Letter = np.zeros(26)
            Letter[LetterIndex.index(Col[1])] = 1
            img = Col[2:]
            img = np.array(img).reshape(28, 28, 1)
            TargetLetter.append(Letter)
            Image.append(img)
        Image = np.array(Image, dtype="float64")
        TargetLetter = np.array(TargetLetter, dtype="float64")
        print(Image.shape)
    np.save("./ImageTest", Image)
    np.save("./TargetLetterTest", TargetLetter)

        
    

