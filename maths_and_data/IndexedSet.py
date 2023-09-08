
class IndexedSet(list):

    def __init__(self):
        super().__init__()

    def addItem(self, item):

        if item not in self:
            self.append(item)
            return item
        return None


