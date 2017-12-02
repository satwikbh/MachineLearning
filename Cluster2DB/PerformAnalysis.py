from Utils.LoggerUtil import LoggerUtil


class PerformAnalysis:
    """
    This class invokes the scripts neccessary for the behavioral analysis of the malware.
    The malware executable's path is picked from the collection analysis is done.
    It appends to the success or failure collections based on the execution status.
    This class is Asynchronous and doesn't respond to any class.
    """

    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()

    def submit_2_cuckoo(self, file_path):
        """
        The executable is submitted to cuckoo sandbox for dynamic analysis using the file_path.
        It returns success or failure based on cuckoo's response.
        :param file_path:
        :return:
        """
        if True:
            return "success"
        else:
            return "failure"

    def consume(self, channel_name):
        # TODO : Insert code to read from channel and get the contents of message. Replace none with that method call.
        message = dict()
        self.log.info("Message from Producer : {}".format(message))
        return message

    def to_book_keeping_collection(self, message, status):
        """
        Inserts into success or failure collection based on the status.
        :param message:
        :param status:
        :return:
        """
        if status == "success":
            message["success_count"] += 1
        else:
            message["failure_count"] += 1
        self.log.info("")

    def consumer(self, channel_name):
        """
        This consumer consumes from the channel and gets the executable's md5 and its file path.
        Once the analysis is done,
        :param channel_name:
        :return:
        """
        message = self.consume(channel_name)
        status = self.submit_2_cuckoo(file_path=message["md5_path"])
        self.to_book_keeping_collection(message, status)

    def main(self):
        pass


if __name__ == '__main__':
    perform_analysis = PerformAnalysis()
    perform_analysis.main()
