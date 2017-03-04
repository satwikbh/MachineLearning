import pickle as pi


class DBLookUp():
    """
    This class will be used for the db lookup
    """

    def main(self):
        f = open("fingerprint/mycsvfile.csv")
        names_list = list()
        for lines in f.readlines():
            split = lines.split(",")
            names_list.append(split[0])
        pi.dump(names_list, open("names_db.txt", "w"))


if __name__ == '__main__':
    dblookup = DBLookUp()
    dblookup.main()

