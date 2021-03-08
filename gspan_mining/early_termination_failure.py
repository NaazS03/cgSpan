from __future__ import absolute_import
from graph import VACANT_VERTEX_LABEL

class EarlyTerminationFailureTree():
    def __init__(self):
        self.children = dict()

    def add_node(self, vevlb, obj):
        self.children[vevlb] = obj

    def add_nodes_rmpath(self, DFScode, rmpath, reversed_rmpath):
        path = list()
        vlbs = dict()
        for e in DFScode:
            if e.vevlb[0] != VACANT_VERTEX_LABEL:
                vlbs[e.frm] = e.vevlb[0]
            if e.vevlb[2] != VACANT_VERTEX_LABEL:
                vlbs[e.to] = e.vevlb[2]
        node = None
        for i, rmpath_i in enumerate(reversed(rmpath)):
            e = DFScode[rmpath_i]
            evevlb = (e.vevlb[0], e.vevlb[1], e.vevlb[2])
            if evevlb[0] == VACANT_VERTEX_LABEL:
                evevlb = (vlbs[e.frm], evevlb[1], evevlb[2])
            if evevlb[2] == VACANT_VERTEX_LABEL:
                evevlb = (evevlb[0], evevlb[1], vlbs[e.to])
            path.append(evevlb)
            if i == 0:
                if not evevlb in self.children:
                    close_graphs = set()
                    node = EarlyTerminationFailureTreeNode(close_graphs)
                    self.children[evevlb] = node
                else:
                    node = self.children[evevlb]
                continue

            if evevlb not in node.children:
                child_node = EarlyTerminationFailureTreeNode(set())
                node.add_node(evevlb, child_node)
                node = child_node
            else:
                node = node.children[evevlb]

        for i, rmpath_i in enumerate(reversed_rmpath):
            e = DFScode[rmpath_i]
            evevlb = (e.vevlb[2], e.vevlb[1], e.vevlb[0])
            if evevlb[0] == VACANT_VERTEX_LABEL:
                evevlb = (vlbs[e.to], evevlb[1], evevlb[2])
            if evevlb[2] == VACANT_VERTEX_LABEL:
                evevlb = (evevlb[0], evevlb[1], vlbs[e.frm])
            path.append(evevlb)
            if node is None:
                if not evevlb in self.children:
                    close_graphs = set()
                    node = EarlyTerminationFailureTreeNode(close_graphs)
                    self.children[evevlb] = node
                else:
                    node = self.children[evevlb]
                continue

            if evevlb not in node.children:
                child_node = EarlyTerminationFailureTreeNode(set())
                node.add_node(evevlb, child_node)
                node = child_node
            else:
                node = node.children[evevlb]

        return path

    def add_path_close_graphs(self, path, graph):
        node = None
        for evevlb in path:
            if node is None:
                node = self.children[evevlb]
            else:
                node = node.children[evevlb]
        node.close_graphs.add(graph)

    def should_allow_early_terminations(self, DFScode, rmpath, graph):
        vlbs = dict()
        for e in DFScode:
            if e.vevlb[0] != VACANT_VERTEX_LABEL:
                vlbs[e.frm] = e.vevlb[0]
            if e.vevlb[2] != VACANT_VERTEX_LABEL:
                vlbs[e.to] = e.vevlb[2]
        node = None
        for i, rmpath_i in enumerate(reversed(rmpath)):
            e = DFScode[rmpath_i]
            evevlb = (e.vevlb[0], e.vevlb[1], e.vevlb[2])
            if evevlb[0] == VACANT_VERTEX_LABEL:
                evevlb = (vlbs[e.frm], evevlb[1], evevlb[2])
            if evevlb[2] == VACANT_VERTEX_LABEL:
                evevlb = (evevlb[0], evevlb[1], vlbs[e.to])
            if node is None:
                if evevlb in self.children:
                    node = self.children[evevlb]
                    continue
                else:
                    for node_evevlb in self.children.keys():
                        close_graphs = self.children[node_evevlb].close_graphs
                        if graph in close_graphs:
                            if (evevlb[0] < node_evevlb[0]) \
                                    or (evevlb[0] == node_evevlb[0] and evevlb[1] < node_evevlb[1]) \
                                    or (evevlb[0] == node_evevlb[0] and evevlb[1] == node_evevlb[1] and evevlb[2] < node_evevlb[2]):
                                return False
                    return True
            else:
                if evevlb in node.children:
                    node = node.children[evevlb]
                    continue
                else:
                    for node_evevlb in node.children.keys():
                        close_graphs = node.children[node_evevlb].close_graphs
                        if graph in close_graphs:
                            if (evevlb[0] < node_evevlb[0]) \
                                    or (evevlb[0] == node_evevlb[0] and evevlb[1] < node_evevlb[1]) \
                                    or (evevlb[0] == node_evevlb[0] and evevlb[1] == node_evevlb[1] and evevlb[2] <
                                        node_evevlb[2]):
                                return False
                    return True
        return False


class EarlyTerminationFailureTreeNode():
    def __init__(self, close_graphs):
        self.close_graphs = close_graphs
        self.children = dict()

    def add_node(self, vevlb, child):
        self.children[vevlb] = child