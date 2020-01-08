class Syntatic_analysator():
    def check(self, number):

        if len(number) == 7 or len(number) == 8:
            #    print(number)
            if(len(number)) == 8:
                for i in range(0, 2):
                    if number[i] == 8:
                        number[i] = 11
                    if number[i] == 0:
                        number[i] = 13
                    if number[i] == 1:
                        number[i] = 18
                    if number[i] == 0:
                        number[i] = 24
                    if number[i] == 2:
                        number[i] = 34

            if(len(number)) == 7:
                for i in range(0, 1):
                    if number[i] == 8:
                        number[i] = 11
                    if number[i] == 0:
                        number[i] = 13
                    if number[i] == 1:
                        number[i] = 18
                    if number[i] == 0:
                        number[i] = 24
                    if number[i] == 2:
                        number[i] = 34

            if len(number) == 8 and number[0] > 9 and number[1] > 9 and number[2] > 9:
                for i in range(3, 7):
                    if number[i] == 11:
                        number[i] = 8
                    if number[i] == 13:
                        number[i] = 0
                    if number[i] == 18:
                        number[i] = 1
                    if number[i] == 24:
                        number[i] = 0
                    if number[i] == 34:
                        number[i] = 2
                if number[3] < 10 and number[4] < 10 and number[5] < 10 and number[6] < 10 and number[7] < 10:
                    return number, 2
                if number[3] > 9 and number[4] > 9 and number[5] < 10 and number[6] < 10 and number[7] < 10:
                    return number, 2
                if number[3] > 9 and number[4] < 10 and number[5] < 10 and number[6] < 10 and number[7] < 10:
                    return number, 2

            elif len(number) == 7 and number[0] > 9 and number[1] > 9 and number[2] > 9:
                for i in range(3, 6):
                    if number[i] == 11:
                        number[i] = 8
                    if number[i] == 13:
                        number[i] = 0
                    if number[i] == 18:
                        number[i] = 1
                    if number[i] == 24:
                        number[i] = 0
                    if number[i] == 34:
                        number[i] = 2
                if number[3] > 9 and number[4] < 10 and number[5] < 10 and number[6] < 10:
                    return number, 2
                if number[3] < 10 and number[4] < 10 and number[5] > 9 and number[6] > 9:
                    return number, 2
                if number[3] < 10 and number[4] > 9 and number[5] > 9 and number[6] < 10:
                    return number, 2
                if number[3] < 10 and number[4] > 9 and number[5] < 10 and number[6] < 10:
                    return number, 2

            elif len(number) == 7 and number[0] > 9 and number[1] > 9:
                for i in range(2, 6):
                    if number[i] == 11:
                        number[i] = 8
                    if number[i] == 13:
                        number[i] = 0
                    if number[i] == 18:
                        number[i] = 1
                    if number[i] == 24:
                        number[i] = 0
                    if number[i] == 34:
                        number[i] = 2
                if number[2] < 10 and number[3] < 10 and number[4] < 10 and number[5] < 10 and number[6] < 10:
                    return number, 2
                if number[2] < 10 and number[3] < 10 and number[4] < 10 and number[5] < 10 and number[6] > 9:
                    return number, 2
                if number[2] < 10 and number[3] < 10 and number[4] < 10 and number[5] > 9 and number[6] > 9:
                    return number, 2
                if number[2] < 10 and number[3] < 10 and number[4] > 9 and number[5] < 10 and number[6] < 10:
                    return number, 2
                if number[2] < 10 and number[3] < 10 and number[4] > 9 and number[5] > 9 and number[6] < 10:
                    return number, 2



            return number, 1

        else:

            return number, 0


