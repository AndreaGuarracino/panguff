use flate2::read::GzDecoder;
use log::{error, info};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Lightweight GFA graph representation for nearest computation
pub struct Graph {
    pub segment_lengths: Vec<usize>,
    pub min_id: usize,
    pub paths: Vec<(String, Vec<u32>)>,
}

impl Graph {
    /// Parse GFA file efficiently (2-pass: segments, then paths)
    pub fn from_gfa(gfa_path: &str) -> std::io::Result<Self> {
        let path = Path::new(gfa_path);

        // First pass: Read segments
        info!("Parsing segments...");
        let mut reader = create_reader(path)?;
        let mut line = String::new();
        let mut seg_lens = Vec::new();
        let mut min_id = usize::MAX;
        let mut max_id = 0;

        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            if !line.starts_with('S') {
                continue;
            }

            let mut fields = line.split('\t');
            if let Some((id_str, seq)) = fields.nth(1).zip(fields.next()) {
                let id = id_str.parse::<usize>().map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid segment ID: {}", e),
                    )
                })?;
                min_id = min_id.min(id);
                max_id = max_id.max(id);
                seg_lens.push((id, seq.trim().len()));
            }
        }

        // Create dense vector for O(1) lookup
        let num_segments = max_id - min_id + 1;
        let mut segment_lengths = vec![0; num_segments];
        for (id, len) in seg_lens {
            segment_lengths[id - min_id] = len;
        }

        info!(
            "Parsed {} segments (ID range: {} - {})",
            num_segments, min_id, max_id
        );

        // Second pass: Read paths
        info!("Parsing paths...");
        let mut reader = create_reader(path)?;
        let mut paths = Vec::new();

        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            if !line.starts_with('P') {
                continue;
            }

            let mut fields = line.split('\t');
            if let Some((name, steps_str)) = fields.nth(1).zip(fields.next()) {
                let mut nodes = Vec::new();
                for step in steps_str.trim().split(',') {
                    if step.is_empty() {
                        continue;
                    }
                    // Remove orientation character (+/-)
                    let seg = step.trim_end_matches('+').trim_end_matches('-');
                    let seg_id = seg.parse::<usize>().map_err(|e| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Invalid segment in path '{}': {}", step, e),
                        )
                    })?;
                    nodes.push((seg_id - min_id) as u32);
                }
                paths.push((name.to_string(), nodes));
            }
        }

        info!("Parsed {} paths", paths.len());

        Ok(Graph {
            segment_lengths,
            min_id,
            paths,
        })
    }

    #[inline]
    pub fn segment_len(&self, node: u32) -> usize {
        self.segment_lengths[node as usize]
    }
}

/// Create reader handling gzip compression
pub fn create_reader(path: &Path) -> std::io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path
        .extension()
        .is_some_and(|ext| ext == "gz" || ext == "bgz")
    {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Load reference path names from file (one per line)
pub fn load_ref_paths(path: &str) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    reader.lines().collect()
}

/// Get all nodes in reference paths
pub fn get_ref_nodes(graph: &Graph, ref_paths: &FxHashSet<String>) -> FxHashSet<u32> {
    let mut ref_nodes = FxHashSet::default();
    for (path_name, nodes) in &graph.paths {
        if ref_paths.contains(path_name.as_str()) {
            ref_nodes.extend(nodes);
        }
    }
    ref_nodes
}

/// Find closest reference node for each node in the graph.
/// Algorithm: traverse each non-reference path forward+backward, tracking distance to nearest ref
pub fn find_nearest(graph: &Graph, ref_paths: &FxHashSet<String>) -> Vec<(u32, i64)> {
    let ref_nodes = get_ref_nodes(graph, ref_paths);
    info!("Reference nodes: {}", ref_nodes.len());

    // Process non-reference paths in parallel, collect all updates
    let all_updates: Vec<Vec<(u32, u32, i64)>> = graph
        .paths
        .par_iter()
        .filter(|(path_name, _)| !ref_paths.contains(path_name))
        .map(|(_, nodes)| {
            let mut updates: Vec<(u32, u32, i64)> = Vec::new();

            // Forward pass
            let mut distance = 0i64;
            let mut ref_node: Option<u32> = None;
            for &node in nodes {
                if ref_nodes.contains(&node) {
                    distance = 0;
                    ref_node = Some(node);
                } else if let Some(r) = ref_node {
                    updates.push((node, r, distance));
                    distance += graph.segment_len(node) as i64;
                }
            }

            // Backward pass
            distance = 0;
            ref_node = None;
            for &node in nodes.iter().rev() {
                if ref_nodes.contains(&node) {
                    distance = 0;
                    ref_node = Some(node);
                } else if let Some(r) = ref_node {
                    updates.push((node, r, distance));
                    distance += graph.segment_len(node) as i64;
                }
            }

            updates
        })
        .collect();

    // Initialize result Vec
    let num_nodes = graph.segment_lengths.len();
    let mut result: Vec<(u32, i64)> = Vec::with_capacity(num_nodes);
    for i in 0..num_nodes {
        result.push((i as u32, i64::MAX));
    }

    // Merge updates sequentially (cache-friendly, no lock overhead)
    for updates in all_updates {
        for (node, r, dist) in updates {
            let entry = &mut result[node as usize];
            if entry.1 > dist {
                *entry = (r, dist);
            }
        }
    }

    // Mark reference nodes with distance = -1
    for (path_name, nodes) in &graph.paths {
        if ref_paths.contains(path_name.as_str()) {
            for &node in nodes {
                result[node as usize] = (node, -1);
            }
        }
    }

    result
}

/// Get position of each reference node on its reference path
pub fn get_ref_positions(
    graph: &Graph,
    ref_paths: &FxHashSet<String>,
) -> FxHashMap<u32, Vec<(usize, String)>> {
    let mut positions: FxHashMap<u32, Vec<(usize, String)>> = FxHashMap::default();

    for (path_name, nodes) in &graph.paths {
        if !ref_paths.contains(path_name.as_str()) {
            continue;
        }

        let mut pos = 0;
        for &node in nodes {
            positions
                .entry(node)
                .or_default()
                .push((pos, path_name.clone()));
            pos += graph.segment_len(node);
        }
    }

    positions
}

/// Write results to TSV file or stdout
pub fn write_output(
    output_path: Option<&str>,
    graph: &Graph,
    result: &[(u32, i64)],
    ref_positions: &FxHashMap<u32, Vec<(usize, String)>>,
) -> std::io::Result<()> {
    let mut writer: Box<dyn Write> = match output_path {
        Some(path) => Box::new(std::io::BufWriter::new(File::create(path)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };

    writeln!(writer, "#node\tref.node\tref.position\tref.path\tdistance")?;

    // Create indexed vector (already in node ID order)
    for (node, &(ref_node, distance)) in result.iter().enumerate() {
        let node_id = node + graph.min_id;
        let ref_node_id = ref_node as usize + graph.min_id;

        // Handle unreachable nodes (never reached from any reference)
        if distance == i64::MAX {
            writeln!(writer, "{}\t.\t.\t.\t.", node_id)?;
            continue;
        }

        // Get all positions where this ref_node appears in reference paths
        if let Some(positions) = ref_positions.get(&ref_node) {
            for (pos, path_name) in positions {
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}",
                    node_id, ref_node_id, pos, path_name, distance
                )?;
            }
        } else {
            // This is a critical error - ref_node was found but has no position info
            error!(
                "Reference node {} has no position information in reference paths",
                ref_node_id
            );
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Reference node {} missing position information",
                    ref_node_id
                ),
            ));
        }
    }

    Ok(())
}
