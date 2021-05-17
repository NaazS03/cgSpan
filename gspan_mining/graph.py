"""Definitions of Edge, Vertex and Graph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import enum


VACANT_EDGE_ID = -1
VACANT_VERTEX_ID = -1
VACANT_EDGE_LABEL = -1
VACANT_VERTEX_LABEL = -1
VACANT_GRAPH_ID = -1
AUTO_EDGE_ID = -1



class EdgeDirection(enum.Enum):
    Forward = 1
    Backward = 2

class Edge(object):
    """Edge class."""

    def __init__(self,
                 eid=VACANT_EDGE_ID,
                 frm=VACANT_VERTEX_ID,
                 to=VACANT_VERTEX_ID,
                 elb=VACANT_EDGE_LABEL):
        """Initialize Edge instance.

        Args:
            eid: edge id.
            frm: source vertex id.
            to: destination vertex id.
            elb: edge label.
        """
        self.eid = eid
        self.frm = frm
        self.to = to
        self.elb = elb


class Vertex(object):
    """Vertex class."""

    def __init__(self,
                 vid=VACANT_VERTEX_ID,
                 vlb=VACANT_VERTEX_LABEL):
        """Initialize Vertex instance.

        Args:
            vid: id of this vertex.
            vlb: label of this vertex.
        """
        self.vid = vid
        self.vlb = vlb
        self.edges = dict()

    def add_edge(self, eid, frm, to, elb):
        """Add an outgoing edge."""
        self.edges[to] = Edge(eid, frm, to, elb)


class Graph(object):
    """Graph class."""

    def __init__(self,
                 gid=VACANT_GRAPH_ID,
                 is_undirected=True,
                 eid_auto_increment=True):
        """Initialize Graph instance.

        Args:
            gid: id of this graph.
            is_undirected: whether this graph is directed or not.
            eid_auto_increment: whether to increment edge ids automatically.
        """
        self.gid = gid
        self.is_undirected = is_undirected
        self.vertices = dict()
        self.set_of_elb = collections.defaultdict(set)
        self.set_of_vlb = collections.defaultdict(set)
        self.eid_auto_increment = eid_auto_increment
        self.counter = itertools.count()
        self.num_edges = 0

    def get_num_vertices(self):
        """Return number of vertices in the graph."""
        return len(self.vertices)

    def add_vertex(self, vid, vlb):
        """Add a vertex to the graph."""
        if vid in self.vertices:
            return self
        self.vertices[vid] = Vertex(vid, vlb)
        self.set_of_vlb[vlb].add(vid)
        return self

    def add_edge(self, eid, frm, to, elb):
        """Add an edge to the graph."""
        if (frm is self.vertices and
                to in self.vertices and
                to in self.vertices[frm].edges):
            return self
        if self.eid_auto_increment:
            eid = next(self.counter)
        self.vertices[frm].add_edge(eid, frm, to, elb)
        self.set_of_elb[elb].add((frm, to))
        if self.is_undirected:
            self.vertices[to].add_edge(eid, to, frm, elb)
            self.set_of_elb[elb].add((to, frm))
        self.num_edges = self.num_edges + 1
        return eid

    def display(self):
        """Display the graph as text."""
        display_str = ''
        print('t # {}'.format(self.gid))
        for vid in self.vertices:
            print('v {} {}'.format(vid, self.vertices[vid].vlb))
            display_str += 'v {} {} '.format(vid, self.vertices[vid].vlb)
        for frm in self.vertices:
            edges = self.vertices[frm].edges
            for to in edges:
                if self.is_undirected:
                    if frm < to:
                        print('e {} {} {}'.format(frm, to, edges[to].elb))
                        display_str += 'e {} {} {} '.format(
                            frm, to, edges[to].elb)
                else:
                    print('e {} {} {}'.format(frm, to, edges[to].elb))
                    display_str += 'e {} {} {}'.format(frm, to, edges[to].elb)
        return display_str

    def plot(self):
        """Visualize the graph."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except Exception as e:
            print('Can not plot graph: {}'.format(e))
            return
        gnx = nx.Graph() if self.is_undirected else nx.DiGraph()
        vlbs = {vid: v.vlb for vid, v in self.vertices.items()}
        elbs = {}
        for vid, v in self.vertices.items():
            gnx.add_node(vid, label=v.vlb)
        for vid, v in self.vertices.items():
            for to, e in v.edges.items():
                if (not self.is_undirected) or vid < to:
                    gnx.add_edge(vid, to, label=e.elb)
                    elbs[(vid, to)] = e.elb
        fsize = (min(16, 1 * len(self.vertices)),
                 min(16, 1 * len(self.vertices)))
        plt.figure(3, figsize=fsize)
        pos = nx.spectral_layout(gnx)
        nx.draw_networkx(gnx, pos, arrows=True, with_labels=True, labels=vlbs)
        nx.draw_networkx_edge_labels(gnx, pos, edge_labels=elbs)
        plt.show()

    def _get_DFSLabels(self, edge):
        frmlbl = self.vertices[edge.frm].vlb
        edgelbl = edge.elb
        tolbl = self.vertices[edge.to].vlb
        return frmlbl, edgelbl, tolbl

class EnumeratedEdge(object):
    def __init__(self, originalGraphId, edgeId):
        """Initialize ProjectedEdge instance."""
        self.originalGraphId = originalGraphId
        self.edgeId = edgeId

    def __repr__(self):
        """Represent ProjectedEdge in string way."""
        return '(originalGraphId={}, edgeId={})'.format(
            self.originalGraphId, self.edgeId
        )

    def __eq__(self, other):
        """Check equivalence of ProjectedEdge."""
        return (self.originalGraphId == other.originalGraphId and
                self.edgeId == other.edgeId)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __hash__(self):
        return hash(hash(self.originalGraphId) + hash(self.edgeId))

# Start Close Graph specific classes
class DFSlabel(object):
    def __init__(self, frmlbl, edgelbl, tolbl):
        """Initialize DFSlabel instance."""
        self.frmlbl = min(frmlbl, tolbl)
        self.edgelbl = edgelbl
        self.tolbl = max(frmlbl, tolbl)

    def __repr__(self):
        """Represent DFSlabel in string way."""
        return '(frmlbl={}, edgelbl={}, tolbl={})'.format(
            self.frmlbl, self.edgelbl, self.tolbl
        )

    def __eq__(self, other):
        """Check equivalence of DFSlabels."""
        return ((self.frmlbl == other.frmlbl and self.edgelbl == other.edgelbl and self.tolbl == other.tolbl) or
               (self.frmlbl == other.tolbl and self.edgelbl == other.edgelbl and self.tolbl == other.frmlbl))

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __hash__(self):
        return hash(hash(self.frmlbl) + hash(self.edgelbl) + hash(self.tolbl))


class DatabaseGraph(Graph):
    def __init__(self,
                 gid=VACANT_GRAPH_ID,
                 is_undirected=True,
                 eid_auto_increment=True):
        super().__init__(gid, is_undirected, eid_auto_increment)
        self.enumerated_edges = dict()

    def add_edge(self, eid, frm, to, elb):
        eid = super(DatabaseGraph, self).add_edge(eid, frm, to , elb)
        self.enumerated_edges[eid] = EnumeratedEdge(self.gid, eid)

class FrequentGraph(Graph):
    def __init__(self,
                 edges_projection_sets,
                 where_graphs,
                 where_projections,
                 pdfs_edges_projection_set,
                 pdfs_edges_projection_list,
                 DFScode,
                 example_gid,
                 gid=VACANT_GRAPH_ID,
                 is_undirected=True,
                 eid_auto_increment=True):
        super().__init__(gid, is_undirected, eid_auto_increment)
        self.edges_projection_sets = edges_projection_sets
        self.where_graphs = where_graphs
        self.support_graphs = len(where_graphs)
        self.where_projections = sorted(where_projections)
        self.where_projections_set = set(where_projections)
        self.support_projections = len(where_projections)
        self.pdfs_edges_projection_set = pdfs_edges_projection_set
        self.pdfs_edges_projection_list = pdfs_edges_projection_list
        self.DFScode = DFScode
        self.example_gid = example_gid

    def is_supergraph_of_with_support_graphs(self, g):
        if self.support_graphs != g.support_graphs:
            return False

        if set(self.where_graphs) != set(g.where_graphs):
            return False

        if not self.edges_projection_sets.keys().issuperset(g.edges_projection_sets.keys()):
            return False

        if self.get_num_vertices() < g.get_num_vertices():
            return False

        for my_pdfs in self.example_pdfs_edges_projection_set:
            for other_pdfs in g.example_pdfs_edges_projection_set:
                if my_pdfs.issuperset(other_pdfs):
                    return True
            # we need to check only on one example
            return False

    def is_supergraph_of_with_support_projections(self, g):
        if self.support_projections < g.support_projections:
            return False, False, None

        if self.where_projections_set != g.where_projections_set:
            return False, False, None

        if not set(self.edges_projection_sets.keys()).issuperset(set(g.edges_projection_sets.keys())):
            return False, False, None
        return self.check_equivalent_occurrence(g.support_projections, g.where_projections, g.pdfs_edges_projection_list, g.dfs_code_edges_directions)

    def check_equivalent_occurrence(self, support_projections, where_projections, pdfs_edges_projection_list, edges_directions):
        if self.support_projections < support_projections:
            return False, False, None

        if set(self.where_projections) != set(where_projections):
            return False, False, None

        #if not edges_projection_sets is None:
        #    if not self.edges_projection_sets.keys().issuperset(edges_projection_sets.keys()):
        #        return False

        #if self.get_num_vertices() < num_vertices:
        #    return False

        possible_isomorphisms = self.find_possible_isomorphisms(pdfs_edges_projection_list[self.example_gid])
        isomorphism_found = False
        preserves_directions = False
        isomorphism = None
        for isomorphism in possible_isomorphisms:
            for gid in pdfs_edges_projection_list.keys():
                for other_edges_projections_list in pdfs_edges_projection_list[gid]:
                    isomorphism_found = True
                    for my_edges_projections_list in self.pdfs_edges_projection_list[gid]:
                        isomorphism_found = True
                        for other_index in isomorphism.keys():
                            my_index = isomorphism[other_index]
                            if other_edges_projections_list[other_index] != my_edges_projections_list[my_index]:
                                isomorphism_found = False
                                break
                        if isomorphism_found is True:
                            break
                    if isomorphism_found is False:
                        break
                if isomorphism_found is False:
                    break
            if isomorphism_found is True:
                preserves_directions = True
                for other_index in isomorphism.keys():
                    my_index = isomorphism[other_index]
                    if self.dfs_code_edges_directions[my_index] != edges_directions[other_index]:
                        preserves_directions = False
                        break
                break

        return isomorphism_found, preserves_directions, isomorphism

    def find_possible_isomorphisms(self, edges_projection_list):
        possible_isomorphisms = list()
        other_projected_edges = edges_projection_list[0]
        for my_projected_edges in self.pdfs_edges_projection_list[self.example_gid]:
            isomorphism = dict()
            for i, other_edge in enumerate(other_projected_edges):
                for j, my_edge in enumerate(my_projected_edges):
                    if other_edge == my_edge:
                        isomorphism[i] = j
                        break
            if len(isomorphism) == len(other_projected_edges):
                possible_isomorphisms.append(isomorphism)
        return possible_isomorphisms

    def can_be_descendant_of(self, support_projections, where_projections, pdfs_edges_projection_list):

        if not set(self.where_projections).issubset(set(where_projections)):
            return False

        #if not edges_projection_sets is None:
        #    if not self.edges_projection_sets.keys().issuperset(edges_projection_sets.keys()):
        #        return False

        #if self.get_num_vertices() < num_vertices:
        #    return False

        possible_isomorphisms = self.find_possible_isomorphisms_all(pdfs_edges_projection_list[self.example_gid])
        isomorphism_found = False

        for isomorphism in possible_isomorphisms:
            for gid in self.pdfs_edges_projection_list.keys():
                for my_edges_projections_list in self.pdfs_edges_projection_list[gid]:
                    isomorphism_found = True
                    for other_edges_projections_list in pdfs_edges_projection_list[gid]:
                        isomorphism_found = True
                        for other_index in isomorphism.keys():
                            my_index = isomorphism[other_index]
                            if other_edges_projections_list[other_index] != my_edges_projections_list[my_index]:
                                isomorphism_found = False
                                break
                        if isomorphism_found:
                            break
                    if not isomorphism_found:
                        break
                if not isomorphism_found:
                    break
            if isomorphism_found:
                break

        return isomorphism_found

    def find_possible_isomorphisms_all(self, edges_projection_list):
        possible_isomorphisms = list()
        for other_projected_edges in edges_projection_list:
            for my_projected_edges in self.pdfs_edges_projection_list[self.example_gid]:
                isomorphism = dict()
                for i, other_edge in enumerate(other_projected_edges):
                    for j, my_edge in enumerate(my_projected_edges):
                        if other_edge == my_edge:
                            isomorphism[i] = j
                            break
                if len(isomorphism) == len(other_projected_edges):
                    possible_isomorphisms.append(isomorphism)
        return possible_isomorphisms