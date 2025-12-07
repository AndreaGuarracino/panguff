//! Compute pairwise path/sample similarity from GFA files
//!
//! This module provides functionality equivalent to `odgi similarity`,
//! computing Jaccard, Cosine, Dice similarity, and Estimated Identity
//! based on node overlap weighted by node length (bp).

use crate::nearest::create_reader;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Mutex;

/// Path identifier (0-based index)
pub type PathId = u32;

/// Node identifier (0-based internal index)
pub type NodeId = u32;

/// Encoded pair of entity IDs
pub type EncodedPair = u64;

/// Information about a parsed path
struct PathInfo {
    name: String,
    steps: Vec<NodeId>,
}

/// Information about a group of paths
pub struct GroupInfo {
    pub name: String,
    pub path_ids: Vec<PathId>,
    pub total_length: u64,
}

/// Similarity index for efficient computation
pub struct SimilarityIndex {
    /// Node lengths (0-based internal indexing)
    #[allow(dead_code)]
    node_lens: Vec<u32>,
    /// Minimum node ID from GFA (for output)
    #[allow(dead_code)]
    min_id: usize,
    /// Path names in order
    pub path_names: Vec<String>,
    /// Total length of each path in bp
    path_lengths: Vec<u64>,
    /// Sparse mapping: node_id -> Vec<(path_id, bp_count)>
    node_to_paths: Vec<(NodeId, Vec<(PathId, u64)>)>,
}

impl SimilarityIndex {
    /// Parse GFA and build similarity index
    pub fn from_gfa(gfa_path: &str, show_progress: bool) -> std::io::Result<Self> {
        let path = Path::new(gfa_path);

        // First pass: Read segment lengths
        if show_progress {
            info!("Pass 1: Reading segments...");
        }
        let mut reader = create_reader(path)?;
        let mut line = String::new();
        let mut seg_lens: Vec<(usize, u32)> = Vec::new();
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
                seg_lens.push((id, seq.trim().len() as u32));
            }
        }

        // Create dense vector for O(1) lookup
        let num_segments = max_id - min_id + 1;
        let mut node_lens = vec![0u32; num_segments];
        for (id, len) in seg_lens {
            node_lens[id - min_id] = len;
        }

        if show_progress {
            info!(
                "Found {} segments (ID range: {} - {})",
                num_segments, min_id, max_id
            );
        }

        // Second pass: Read paths
        if show_progress {
            info!("Pass 2: Reading paths...");
        }
        let mut reader = create_reader(path)?;
        let mut paths: Vec<PathInfo> = Vec::new();

        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }

            match line.chars().next() {
                Some('P') => {
                    // GFA1 P-line: P <name> <segments> <overlaps>
                    let mut fields = line.split('\t');
                    if let Some((name, steps_str)) = fields.nth(1).zip(fields.next()) {
                        let mut steps = Vec::new();
                        for step in steps_str.trim().split(',') {
                            if step.is_empty() {
                                continue;
                            }
                            let seg = step.trim_end_matches('+').trim_end_matches('-');
                            if let Ok(seg_id) = seg.parse::<usize>() {
                                steps.push((seg_id - min_id) as NodeId);
                            }
                        }
                        if !steps.is_empty() {
                            paths.push(PathInfo {
                                name: name.to_string(),
                                steps,
                            });
                        }
                    }
                }
                Some('W') => {
                    // GFA1.1/2 W-line: W <sample> <hap> <seq> <start> <end> <walk>
                    let mut fields = line.split('\t');
                    fields.next(); // Skip 'W'
                    let sample = fields.next();
                    let haplotype = fields.next();
                    let seq_name = fields.next();
                    fields.next(); // start
                    fields.next(); // end
                    let walk = fields.next();

                    if let (Some(sample), Some(hap), Some(seq), Some(walk)) =
                        (sample, haplotype, seq_name, walk)
                    {
                        let name = format!("{}#{}#{}", sample, hap, seq);
                        let mut steps = Vec::new();
                        let walk = walk.trim();

                        // Parse walk: >seg1<seg2>seg3...
                        let mut i = 0;
                        let walk_bytes = walk.as_bytes();
                        while i < walk_bytes.len() {
                            // Skip orientation marker
                            if walk_bytes[i] == b'>' || walk_bytes[i] == b'<' {
                                i += 1;
                            }
                            // Find end of segment name
                            let start = i;
                            while i < walk_bytes.len()
                                && walk_bytes[i] != b'>'
                                && walk_bytes[i] != b'<'
                            {
                                i += 1;
                            }
                            if start < i {
                                if let Ok(seg_id) = std::str::from_utf8(&walk_bytes[start..i])
                                    .ok()
                                    .and_then(|s| s.parse::<usize>().ok())
                                    .ok_or(())
                                {
                                    steps.push((seg_id - min_id) as NodeId);
                                }
                            }
                        }
                        if !steps.is_empty() {
                            paths.push(PathInfo { name, steps });
                        }
                    }
                }
                _ => continue,
            }
        }

        if show_progress {
            info!("Found {} paths", paths.len());
            info!("Building similarity index...");
        }

        // Compute path lengths and node->paths mapping
        let mut path_names: Vec<String> = Vec::with_capacity(paths.len());
        let mut path_lengths: Vec<u64> = Vec::with_capacity(paths.len());
        let mut node_paths_map: FxHashMap<NodeId, FxHashMap<PathId, u64>> = FxHashMap::default();

        for (path_idx, path) in paths.iter().enumerate() {
            let path_id = path_idx as PathId;
            path_names.push(path.name.clone());

            let mut path_len: u64 = 0;
            let mut node_visits: FxHashMap<NodeId, u64> = FxHashMap::default();

            for &node_id in &path.steps {
                let node_len = node_lens.get(node_id as usize).copied().unwrap_or(0) as u64;
                path_len += node_len;
                *node_visits.entry(node_id).or_default() += node_len;
            }

            path_lengths.push(path_len);

            for (node_id, bp_count) in node_visits {
                node_paths_map
                    .entry(node_id)
                    .or_default()
                    .insert(path_id, bp_count);
            }
        }

        // Convert to Vec format
        let node_to_paths: Vec<(NodeId, Vec<(PathId, u64)>)> = node_paths_map
            .into_iter()
            .map(|(node_id, path_map)| {
                let path_vec: Vec<(PathId, u64)> = path_map.into_iter().collect();
                (node_id, path_vec)
            })
            .collect();

        if show_progress {
            info!(
                "Index built: {} paths, {} nodes with paths",
                path_names.len(),
                node_to_paths.len()
            );
        }

        Ok(Self {
            node_lens,
            min_id,
            path_names,
            path_lengths,
            node_to_paths,
        })
    }

    /// Get number of nodes with at least one path
    pub fn node_count(&self) -> usize {
        self.node_to_paths.len()
    }
}

/// Group paths by a delimiter character
pub fn group_paths(path_names: &[String], delim: char, delim_pos: u16) -> Vec<GroupInfo> {
    let mut group_map: FxHashMap<String, Vec<PathId>> = FxHashMap::default();

    for (idx, name) in path_names.iter().enumerate() {
        let group_name = extract_group_name(name, delim, delim_pos);
        group_map
            .entry(group_name)
            .or_default()
            .push(idx as PathId);
    }

    let mut groups: Vec<GroupInfo> = group_map
        .into_iter()
        .map(|(name, path_ids)| GroupInfo {
            name,
            path_ids,
            total_length: 0,
        })
        .collect();

    groups.sort_by(|a, b| a.name.cmp(&b.name));
    groups
}

/// Extract group name from path name using delimiter
fn extract_group_name(path_name: &str, delim: char, delim_pos: u16) -> String {
    let target_pos = delim_pos.saturating_sub(1) as usize;
    let mut occurrences = 0;

    for (idx, c) in path_name.char_indices() {
        if c == delim {
            if occurrences == target_pos {
                return path_name[..idx].to_string();
            }
            occurrences += 1;
        }
    }

    // Delimiter not found enough times
    if occurrences > 0 {
        for (idx, c) in path_name.char_indices().rev() {
            if c == delim {
                return path_name[..idx].to_string();
            }
        }
    }

    path_name.to_string()
}

/// Update group lengths from path lengths
fn update_group_lengths(groups: &mut [GroupInfo], path_lengths: &[u64]) {
    for group in groups.iter_mut() {
        group.total_length = group
            .path_ids
            .iter()
            .map(|&pid| path_lengths.get(pid as usize).copied().unwrap_or(0))
            .sum();
    }
}

/// Encode two IDs into a single u64
#[inline]
fn encode_pair(a: u32, b: u32) -> EncodedPair {
    ((a as u64) << 32) | (b as u64)
}

/// Decode a pair from u64
#[inline]
fn decode_pair(encoded: EncodedPair) -> (u32, u32) {
    ((encoded >> 32) as u32, (encoded & 0xFFFFFFFF) as u32)
}

/// Result of similarity computation
pub struct SimilarityResult {
    pub intersections: FxHashMap<EncodedPair, u64>,
    pub entity_lengths: Vec<u64>,
    pub entity_names: Vec<String>,
}

/// Compute pairwise similarity between paths or groups
pub fn compute_similarity(
    index: &SimilarityIndex,
    groups: Option<&mut [GroupInfo]>,
    all_pairs: bool,
    show_progress: bool,
) -> SimilarityResult {
    // Prepare entity info (paths or groups)
    let (entity_lengths, entity_names, path_to_entity) = if let Some(groups) = groups {
        update_group_lengths(groups, &index.path_lengths);

        let lengths: Vec<u64> = groups.iter().map(|g| g.total_length).collect();
        let names: Vec<String> = groups.iter().map(|g| g.name.clone()).collect();

        let mut path_to_group: FxHashMap<PathId, u32> = FxHashMap::default();
        for (group_idx, group) in groups.iter().enumerate() {
            for &path_id in &group.path_ids {
                path_to_group.insert(path_id, group_idx as u32);
            }
        }

        (lengths, names, Some(path_to_group))
    } else {
        let lengths = index.path_lengths.clone();
        let names = index.path_names.clone();
        (lengths, names, None)
    };

    let num_entities = entity_names.len();

    // Pre-populate with zeros if requested
    let mut intersections: FxHashMap<EncodedPair, u64> = if all_pairs {
        let mut map = FxHashMap::default();
        for i in 0..num_entities as u32 {
            for j in 0..num_entities as u32 {
                map.insert(encode_pair(i, j), 0);
            }
        }
        map
    } else {
        FxHashMap::default()
    };

    // Use limited thread-local maps
    let num_threads = rayon::current_num_threads();
    let max_local_maps = num_threads.min(4);
    let local_maps: Vec<Mutex<FxHashMap<EncodedPair, u64>>> = (0..max_local_maps)
        .map(|_| Mutex::new(FxHashMap::default()))
        .collect();

    let total_nodes = index.node_count();

    // Setup progress bar
    let progress = if show_progress {
        let pb = ProgressBar::new(total_nodes as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[panguff::similarity] {msg} [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );
        pb.set_message("Computing intersections");
        Some(pb)
    } else {
        None
    };

    // Process nodes in parallel
    index
        .node_to_paths
        .par_iter()
        .enumerate()
        .for_each(|(node_idx, (_node_id, path_visits))| {
            // Convert path visits to entity visits
            let entity_visits: Vec<(u32, u64)> = if let Some(ref p2e) = path_to_entity {
                let mut group_visits: FxHashMap<u32, u64> = FxHashMap::default();
                for &(path_id, bp_count) in path_visits.iter() {
                    if let Some(&group_id) = p2e.get(&path_id) {
                        *group_visits.entry(group_id).or_default() += bp_count;
                    }
                }
                group_visits.into_iter().collect()
            } else {
                path_visits
                    .iter()
                    .map(|&(path_id, bp_count)| (path_id, bp_count))
                    .collect()
            };

            // Compute pairwise contributions
            let mut local_contributions: FxHashMap<EncodedPair, u64> = FxHashMap::default();
            for i in 0..entity_visits.len() {
                let (entity_a, bp_a) = entity_visits[i];
                for j in 0..entity_visits.len() {
                    let (entity_b, bp_b) = entity_visits[j];
                    let contribution = bp_a.min(bp_b);
                    let key = encode_pair(entity_a, entity_b);
                    *local_contributions.entry(key).or_default() += contribution;
                }
            }

            // Merge into thread-local map
            let map_idx = rayon::current_thread_index().unwrap_or(0) % max_local_maps;
            let mut map = local_maps[map_idx].lock().unwrap();
            for (key, value) in local_contributions {
                *map.entry(key).or_default() += value;
            }

            // Update progress
            if let Some(ref pb) = progress {
                if node_idx % 10000 == 0 {
                    pb.set_position(node_idx as u64);
                }
            }
        });

    if let Some(pb) = progress {
        pb.finish_with_message("Done");
    }

    // Merge all thread-local maps
    for mutex in local_maps {
        let local_map = mutex.into_inner().unwrap();
        for (key, value) in local_map {
            *intersections.entry(key).or_default() += value;
        }
    }

    SimilarityResult {
        intersections,
        entity_lengths,
        entity_names,
    }
}

/// Compute Jaccard similarity
#[inline]
fn jaccard(intersection: u64, len_a: u64, len_b: u64) -> f64 {
    let union = len_a + len_b - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Compute Cosine similarity
#[inline]
fn cosine(intersection: u64, len_a: u64, len_b: u64) -> f64 {
    let product = len_a * len_b;
    if product == 0 {
        0.0
    } else {
        intersection as f64 / (product as f64).sqrt()
    }
}

/// Compute Dice similarity
#[inline]
fn dice(intersection: u64, len_a: u64, len_b: u64) -> f64 {
    let sum = len_a + len_b;
    if sum == 0 {
        0.0
    } else {
        2.0 * (intersection as f64 / sum as f64)
    }
}

/// Compute estimated identity from Jaccard
#[inline]
fn estimated_identity(jaccard: f64) -> f64 {
    if jaccard == 0.0 {
        0.0
    } else {
        2.0 * jaccard / (1.0 + jaccard)
    }
}

/// Write similarity results to TSV
pub fn write_output(
    output_path: Option<&str>,
    result: &SimilarityResult,
    distances: bool,
) -> std::io::Result<()> {
    let mut writer: Box<dyn Write> = match output_path {
        Some(path) => Box::new(std::io::BufWriter::new(File::create(path)?)),
        None => Box::new(std::io::BufWriter::new(std::io::stdout())),
    };

    // Write header
    if distances {
        writeln!(
            writer,
            "group.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\t\
             jaccard.distance\tcosine.distance\tdice.distance\t\
             estimated.difference.rate\teuclidean.distance\tmanhattan.distance"
        )?;
    } else {
        writeln!(
            writer,
            "group.a\tgroup.b\tgroup.a.length\tgroup.b.length\tintersection\t\
             jaccard.similarity\tcosine.similarity\tdice.similarity\testimated.identity"
        )?;
    }

    // Chunked buffering
    const BUFFER_CHUNK_SIZE: usize = 100000;
    let mut output_buffer = String::with_capacity(BUFFER_CHUNK_SIZE * 200);
    let mut lines_written = 0usize;

    for (&encoded_pair, &intersection) in &result.intersections {
        let (id_a, id_b) = decode_pair(encoded_pair);

        let len_a = result.entity_lengths.get(id_a as usize).copied().unwrap_or(0);
        let len_b = result.entity_lengths.get(id_b as usize).copied().unwrap_or(0);
        let name_a = result
            .entity_names
            .get(id_a as usize)
            .map(|s| s.as_str())
            .unwrap_or("");
        let name_b = result
            .entity_names
            .get(id_b as usize)
            .map(|s| s.as_str())
            .unwrap_or("");

        let jaccard_val = jaccard(intersection, len_a, len_b);
        let cosine_val = cosine(intersection, len_a, len_b);
        let dice_val = dice(intersection, len_a, len_b);
        let est_id_val = estimated_identity(jaccard_val);

        use std::fmt::Write as FmtWrite;
        if distances {
            let euclidean = (((len_a + len_b - intersection) - intersection) as f64).sqrt();
            let manhattan = (len_a + len_b - intersection) - intersection;
            let _ = writeln!(
                output_buffer,
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}",
                name_a,
                name_b,
                len_a,
                len_b,
                intersection,
                1.0 - jaccard_val,
                1.0 - cosine_val,
                1.0 - dice_val,
                1.0 - est_id_val,
                euclidean,
                manhattan
            );
        } else {
            let _ = writeln!(
                output_buffer,
                "{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                name_a,
                name_b,
                len_a,
                len_b,
                intersection,
                jaccard_val,
                cosine_val,
                dice_val,
                est_id_val
            );
        }

        lines_written += 1;
        if lines_written % BUFFER_CHUNK_SIZE == 0 {
            writer.write_all(output_buffer.as_bytes())?;
            output_buffer.clear();
        }
    }

    if !output_buffer.is_empty() {
        writer.write_all(output_buffer.as_bytes())?;
    }

    Ok(())
}
