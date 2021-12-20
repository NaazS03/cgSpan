# <div align = center>cgSpan</div>

**cite** contains implementation for our [paper](https://arxiv.org/abs/2112.09573).  If you find this code useful in your research, please consider citing:

    @misc{shaul2021cgspan,
      title={cgSpan: Closed Graph-Based Substructure Pattern Mining}, 
      author={Zevin Shaul and Sheikh Naaz},
      year={2021},
      eprint={2112.09573},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
    }

**cgSpan** is an algorithm for mining closed frequent subgraphs. This implementation of cgSpan
is built using an existing implementation for gSpan.

**gSpan** is an algorithm for mining frequent subgraphs.

This program implements cgSpan with Python. The repository on GitHub is [https://github.com/NaazS03/cgSpan](https://github.com/NaazS03/cgSpan)

The gSpan implementation referenced by this program can be found on GitHub at [https://github.com/betterenvi/gSpan](https://github.com/betterenvi/gSpan).

### Undirected Graphs
This program supports undirected graphs.

### How to install

This program supports **Python 3**.

##### Method 1

Install this project using pip:
```sh
pip install cgspan-mining
```

##### Method 2

First, clone the project:

```sh
git clone https://github.com/NaazS03/cgSpan.git
cd cgSpan
```

You can ***optionally*** install this project as a third-party library so that you can run it under ***any*** path.

```sh
python setup.py install
```

### How to run

The command is:

```sh
python -m cgspan_mining [-s min_support] [-n num_graph] [-l min_num_vertices] [-u max_num_vertices] [-v True/False] [-p True/False] [-w True/False] [-h] database_file_name 
```


##### Some examples

- Read graph data from ./graphdata/graph.data, and mine closed subgraphs given min support is 5000

```
python -m cgspan_mining -s 5000 ./graphdata/graph.data
```

- Read graph data from ./graphdata/graph.data, mine closed subgraphs given min support is 5000, and visualize these frequent subgraphs(matplotlib and networkx are required)

```
python -m cgspan_mining -s 5000 -p True ./graphdata/graph.data
```

- Print help info

```
python -m cgspan_mining -h
```

### Reference
- [cgSpan Paper](https://arxiv.org/pdf/2112.09573.pdf)

cgSpan: Closed Graph-Based Substructure Pattern Mining, by Zevin Shaul and Sheikh Naaz

- [CloseGraph Paper](https://sites.cs.ucsb.edu/~xyan/papers/CloseGraph.pdf)

CloseGraph: Mining Close Frequent Graph Patterns, by X. Yan and J.Han.

- [gSpan Paper](http://www.cs.ucsb.edu/~xyan/papers/gSpan-short.pdf)

gSpan: Graph-Based Substructure Pattern Mining, by X. Yan and J. Han. 
Proc. 2002 of Int. Conf. on Data Mining (ICDM'02). 