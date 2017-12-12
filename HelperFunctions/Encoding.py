from Utils.LoggerUtil import LoggerUtil


class Encoding:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    @staticmethod
    def encode_3_len_vector(vector, x):
        output = ""
        if vector[x: x + 3] == '000':
            output += 'A'
        if vector[x: x + 3] == '001':
            output += 'B'
        if vector[x: x + 3] == '010':
            output += 'C'
        if vector[x: x + 3] == '011':
            output += 'D'
        if vector[x: x + 3] == '100':
            output += 'E'
        if vector[x: x + 3] == '101':
            output += 'F'
        if vector[x: x + 3] == '110':
            output += 'G'
        if vector[x: x + 3] == '111':
            output += 'H'
        return output

    @staticmethod
    def encode_2_len_vector(vector, length, x):
        output = ""
        if vector[x:length] == '00':
            output += 'I'
        if vector[x:length] == '01':
            output += 'J'
        if vector[x:length] == '10':
            output += 'K'
        if vector[x:length] == '11':
            output += 'L'
        return output

    @staticmethod
    def encode_single_len_vector(vector, length, x):
        output = ""
        if vector[x:length] == '0':
            output += 'M'
        if vector[x:length] == '1':
            output += 'N'
        return output

    def encode(self, vector):
        """
        Takes a vector in the form of '010001111' and the converts it by encoding 3-bits each.
        If k = 3 then 000 -> 0, 001 -> 1, ..., 111 -> 7 and the encoded values represent variables from A to H.
        If k = 2 then 00 -> 8, 01 -> 9, 10 -> 10, 11 -> 11 and the encoded values represent variables from I to L.
        If k = 1 then 0 -> 12, 1 -> 13 and the encoded values represent variables M, N.
        :param vector:
        :return:
        """
        x, length = 0, len(vector)
        output = ""
        while x <= length - 3:
            output += self.encode_3_len_vector(vector, x)
            x += 3

        if (length - x) % 3 == 1:
            output += self.encode_single_len_vector(vector, length, x)
        elif (length - x) % 3 == 2:
            output += self.encode_2_len_vector(vector, length, x)
        return output

    def _test(self):
        vector = "1100011010101"
        output = self.encode(vector)
        print(output)


if __name__ == '__main__':
    encoder = Encoding()
    encoder._test()
