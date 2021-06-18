from .dfscode import DFSedge
from .dfscode import DFScode

class TrieNode:

    def __init__(self, dfsedge):
        self.dfsedge = dfsedge

        self.is_end = False

        self.children = {}


class Trie(object):

    def __init__(self):

        self.root = TrieNode(None)

    def insert(self, dfscode):

        node = self.root

        for dfsedge in dfscode:
            if dfsedge in node.children:
                node = node.children[dfsedge]
            else:

                new_node = TrieNode(dfsedge)
                node.children[dfsedge] = new_node
                node = new_node

        node.is_end = True

    def search(self, dfscode):

        node = self.root

        for dfsedge in dfscode:
            if dfsedge in node.children:
                node = node.children[dfsedge]
            else:
                return False
        return True