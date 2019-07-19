class ParseException(Exception):

    def __init__(self, msg, file=None):
        super().__init__(self, msg)
        self.file = file

    def __str__(self):
        msg = super().__str__()
        if self.file:
            return 'Error occured while parsing the file {}: {}.'.format(
                self.file, msg
            )
        else:
            return 'Error occured while parsing: {}.'.format(msg)