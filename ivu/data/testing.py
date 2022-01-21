from ivu.data.training import TrainInputData


class TestInputData(TrainInputData):
    def __init__(self, x, y):
        super().__init__(x, y)
