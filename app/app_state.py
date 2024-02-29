class SingletonState():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SingletonState, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    def rem_data(self):
        self.data = {}
    data = {}
    