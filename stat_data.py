class Stat():
    """class about data statics"""
    def __init__(self, data):
        self.data = data
    
    def n_data(self):
        """
        The number of transactions
        """
        return len(self.data)
    
    def get_trans(self):
        trans = []
        for d in self.data:
            trans.append(list(d[1]))
        return trans
    
    def get_itemset(self):
        """
        return set of item
        """
        itemset = set()
        for d in self.data:
            itemset |= d[1]
        return itemset
    
    def n_item(self, itemset=None):
        """
        The number of items
        """
        if itemset is None:
            itemset = self.get_itemset()
        return len(itemset)

    def a_item(self):
        """
        Average number of item in a transaction
        """
        mean = 0
        for d in self.data:
            mean += len(d[1])
        mean /= self.n_data()
        return mean
    
    def get_usrset(self):
        """
        return set of user
        """
        usrset = set()
        for d in self.data:
            usrset |= {d[0]}
        return usrset
    
    def n_usr(self, usrset=None):
        """
        The number of user
        """
        if usrset is None:
            usrset = self.get_usrset()
        return len(usrset)
    
    def show(self):
        """
        print all statics
        """
        print(f"""
        # transactions: {self.n_data()}
        # items       : {self.n_item()}
        Ave item      : {self.a_item()}
        # users       : {self.n_usr()}
        """)
