# <div align = center>CloseGraph</div>

**CloseGraph** is an algorithm for mining maximal frequent subgraphs. This implementation of CloseGraph
is built using an existing implementation for gSpan.

**gSpan** is an algorithm for mining frequent subgraphs.

This program implements CloseGraph with Python. The repository on GitHub is [https://github.com/NaazS03/CloseGraph](https://github.com/NaazS03/CloseGraph)

The gSpan implementation referenced by this program can be found on GitHub at [https://github.com/betterenvi/gSpan](https://github.com/betterenvi/gSpan).

### Undirected Graphs
This program supports undirected graphs.

### How to install

This program supports **Python 3**.

##### Method 1

First, clone the project:

```sh
git clone https://github.com/NaazS03/CloseGraph
cd CloseGraph
```

### How to run

The command is:

```sh
python -m gspan_mining [-s min_support] [-n num_graph] [-l min_num_vertices] [-u max_num_vertices] [-d True/False] [-v True/False] [-p True/False] [-w True/False] [-h] database_file_name 
```


##### Some examples

- Read graph data from ./graphdata/graph.data, and mine undirected subgraphs given min support is 5000
```
python -m gspan_mining -s 5000 ./graphdata/graph.data
```

- Read graph data from ./graphdata/graph.data, mine undirected subgraphs given min support is 5000, and visualize these frequent subgraphs(matplotlib and networkx are required)
```
python -m gspan_mining -s 5000 -p True ./graphdata/graph.data
```

- Print help info
```
python -m gspan_mining -h
```

### Reference
- [CloseGraph Paper](https://sites.cs.ucsb.edu/~xyan/papers/CloseGraph.pdf)

CloseGraph: Mining Close Frequent Graph Patterns, by X. Yan and J.Han.

- [gSpan Paper](http://www.cs.ucsb.edu/~xyan/papers/gSpan-short.pdf)

gSpan: Graph-Based Substructure Pattern Mining, by X. Yan and J. Han. 
Proc. 2002 of Int. Conf. on Data Mining (ICDM'02). 