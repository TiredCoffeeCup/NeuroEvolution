
class IndexedSet(list):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def addItem(self, item):

        if item not in self:
            self.append(item)
            return item
        return None


