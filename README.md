# panguff

A toolkit for pangenome stuff.


## Installation

```bash
git clone --recursive https://github.com/AndreaGuarracino/panguff.git
cd panguff
cargo build --release
```

## Commands

### `nearest` - Find nearest reference nodes

For each node in a pangenome graph, find the closest reference node by base pair distance through path connectivity.

**Usage:**
```bash
panguff nearest --gfa graph.gfa --references ref_paths.txt --output nearest.tsv
```

**Arguments:**
- `--gfa`: GFA v1.0 pangenome graph file with integer segment IDs (can be gzipped)
- `--references`: File with reference path names (one per line)
- `--output`: Output TSV file
- `--threads`: Number of parallel threads (default: 4)

**Output format:**
```
#node    ref.node    ref.position    ref.path    distance
1        1           0               chr1        -1
2        .           .               .           .
10       1           0               chr1        0
23       1           4500            chr1        1250
```

**Distance values:**
- `-1`: Node is a reference node itself
- `0`: Node is directly adjacent to a reference node
- `> 0`: Base pair distance to nearest reference node
- `.`: Node is unreachable from any reference path

## Author

Andrea Guarracino <aguarra1@uthsc.edu>
